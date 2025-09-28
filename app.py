import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import os
import requests
from bs4 import BeautifulSoup
import re
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import io # Import for handling the Excel output

# ======================================================================================
# CONFIGURATION & HEADER
# ======================================================================================
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Stock Analyzer | NS    T R A D E R")
st.markdown("Select an industry from your Excel file to get a consolidated analysis, including financial ratios from **Screener.in** and a detailed **Swing Trading** recommendation with Fibonacci levels.")

# --- API KEY CONFIGURATION ---
# IMPORTANT: Replace the placeholder below with your actual NewsAPI.org API Key
NEWSAPI_KEY = "517d56d684f04f0bba8a65bcc478727c"

# ======================================================================================
# DATA FETCHING & CALCULATION FUNCTIONS
# ======================================================================================

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker):
    """Fetches all necessary data for a stock from yfinance."""
    # Ensure ticker format for NSE
    stock = yf.Ticker(f"{ticker}.NS")
    history_weekly = stock.history(period="3y", interval="1wk")
    history_daily = stock.history(period="2y", interval="1d")
    info = stock.info
    financials = stock.financials
    return history_weekly, info, financials, history_daily

@st.cache_data(ttl=3600)
def get_price_on_date(daily_history, target_date_str):
    """Finds the closest closing price and the actual date to a target date from daily history."""
    try:
        target_date = pd.to_datetime(target_date_str)
        # Use 'nearest' method to find the closest trading day
        closest_date_index = daily_history.index.get_indexer([target_date], method='nearest')[0]
        actual_date = daily_history.index[closest_date_index]
        price = daily_history.iloc[closest_date_index]['Close']
        return price, actual_date
    except Exception:
        return None, None

@st.cache_data(ttl=3600)
def scrape_screener_data(ticker):
    """Scrapes key financial data for a given ticker from screener.in."""
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None, "Failed to load page"
    
    soup = BeautifulSoup(response.content, 'html.parser')
    data = {}
    ratio_list = soup.select_one('#top-ratios')
    if not ratio_list: return None, "Ratios not found"
    for li in ratio_list.find_all('li'):
        name = li.select_one('.name').get_text(strip=True) if li.select_one('.name') else ''
        value = li.select_one('.nowrap.value .number').get_text(strip=True) if li.select_one('.nowrap.value .number') else ''
        if name and value: data[name] = value
    return data, "Success"

@st.cache_data(ttl=3600)
def get_newsapi_articles(ticker):
    """Fetches recent news articles for a given ticker from NewsAPI.org."""
    if NEWSAPI_KEY == "REPLACE_WITH_YOUR_NEWSAPI_KEY":
        return {"status": "Error", "articles": [], "message": "NewsAPI key not set. Please update NEWSAPI_KEY."}
    
    # Use the raw ticker (e.g., RELIANCE) for news search relevance
    # Added "AND stock" to improve search relevance
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker} AND stock&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"pageSize=5&" 
        f"apiKey={NEWSAPI_KEY}"
    )
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get("articles", [])
        if not articles:
            return {"status": "Success", "articles": [], "message": f"No recent news found for {ticker}."}

        # Filter out essential fields
        cleaned_articles = [
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name"),
                "publishedAt": pd.to_datetime(article.get("publishedAt")).strftime('%Y-%m-%d')
            } for article in articles
        ]
        return {"status": "Success", "articles": cleaned_articles, "message": "Successfully fetched news."}

    except requests.RequestException as e:
        return {"status": "Error", "articles": [], "message": f"NewsAPI request failed: {e}. Check key or connection."}
    except Exception:
        return {"status": "Error", "articles": [], "message": "Failed to process NewsAPI response."}


def calculate_graham_intrinsic_value(info, financials, bond_yield=7.5):
    """Calculates the intrinsic value of a stock using Benjamin Graham's formula."""
    try:
        eps = info.get('trailingEps')
        if not eps or eps <= 0: return None
        net_income = financials.loc['Net Income']
        if net_income.isnull().all() or len(net_income.dropna()) < 2: return None
        growth_rates = net_income.pct_change().dropna()
        avg_growth_rate = np.mean(growth_rates)
        g = min(avg_growth_rate * 100, 15.0) 
        if g < 0: g = 0
        return (eps * (8.5 + 2 * g) * 4.4) / bond_yield
    except (KeyError, IndexError, TypeError):
        return None

# --- NEW: 52-DAY AVERAGE CALCULATION ---
def calculate_52_day_average(daily_history):
    """Calculates the average closing price over the last 52 trading days."""
    if daily_history.empty or len(daily_history) < 52:
        return None
    # Calculate 52-day Simple Moving Average (SMA)
    # Use a rolling mean on the last 52 data points
    avg_52 = daily_history['Close'].iloc[-52:].mean()
    return avg_52

# --- FIBONACCI RETRACEMENT ANALYSIS ---
def calculate_fibonacci_levels(history):
    """Calculates Fibonacci retracement levels and provides a signal."""
    lookback_period = history.tail(52) # Look at the last 52 weeks (1 year)
    high_price = lookback_period['High'].max()
    low_price = lookback_period['Low'].min()
    price_range = high_price - low_price
    current_price = history['Close'].iloc[-1]

    # Determine trend
    is_uptrend = current_price > lookback_period['Close'].iloc[0]

    levels = {}
    if is_uptrend:
        levels['23.6%'] = high_price - (price_range * 0.236)
        levels['38.2%'] = high_price - (price_range * 0.382)
        levels['50.0%'] = high_price - (price_range * 0.500)
        levels['61.8%'] = high_price - (price_range * 0.618)
    else: # Downtrend
        levels['23.6%'] = low_price + (price_range * 0.236)
        levels['38.2%'] = low_price + (price_range * 0.382)
        levels['50.0%'] = low_price + (price_range * 0.500)
        levels['61.8%'] = low_price + (price_range * 0.618)
    
    # Generate signal
    signal = "Neutral"
    if is_uptrend:
        if current_price > levels['38.2%'] and current_price < high_price:
            signal = f"Finding support above the 38.2% level (₹{levels['38.2%']:.2f}). Potential continuation of uptrend."
        elif current_price <= levels['61.8%']:
            signal = "Trend weakening, has broken below the 61.8% support."
    else: # Downtrend
        if current_price < levels['61.8%'] and current_price > low_price:
            signal = f"Facing resistance below the 61.8% level (₹{levels['61.8%']:.2f}). Potential continuation of downtrend."
        elif current_price >= levels['61.8%']:
            signal = "Potential trend reversal, has broken above the 61.8% resistance."
            
    return levels, signal, is_uptrend, high_price, low_price

def calculate_swing_trade_analysis(history):
    """Calculates swing trading indicators and generates a recommendation."""
    if len(history) < 52:
        # Return structure compatible with the caller when data is insufficient
        return None, "Insufficient Data", "Not enough weekly data for full analysis."

    close = history['Close']
    price = close.iloc[-1]
    
    # Calculate indicators
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    rsi_14 = RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_indicator = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    obv_indicator = OnBalanceVolumeIndicator(close, history['Volume'])
    obv_slope = obv_indicator.on_balance_volume().diff().rolling(window=5).mean().iloc[-1]

    indicators = {"20W SMA": sma_20, "50W SMA": sma_50, "RSI (14)": rsi_14, "MACD Line": macd_line, "MACD Signal": macd_signal}

    score = 0
    reasons = []
    
    # Scoring Logic
    if price > sma_20: score += 2; reasons.append("✅ Price > 20W SMA (Short-term trend is up).")
    else: reasons.append("❌ Price < 20W SMA (Short-term trend is down).")
    if sma_20 > sma_50: score += 2; reasons.append("✅ 20W SMA > 50W SMA (Golden Cross).")
    else: reasons.append("❌ 20W SMA < 50W SMA (Death Cross).")
    if macd_line > macd_signal: score += 1; reasons.append("✅ MACD > Signal (Bullish momentum).")
    else: reasons.append("❌ MACD < Signal (Bearish momentum).")
    if 45 < rsi_14 < 68: score += 2; reasons.append(f"✅ RSI is healthy at {rsi_14:.1f}.")
    else: reasons.append(f"⚠️ RSI is {rsi_14:.1f} (Not in optimal range).")
    if obv_slope > 0: score += 2; reasons.append("✅ OBV trend is positive (Volume confirms trend).")
    else: reasons.append("❌ OBV trend is negative (Volume does not confirm).")
    
    # Final Recommendation
    if score >= 7: recommendation = "Strong Buy"
    elif score >= 5: recommendation = "Buy"
    elif score >= 3: recommendation = "Hold / Monitor"
    else: recommendation = "Sell / Avoid"
        
    return indicators, recommendation, "\n\n".join(reasons)

# --- NEW CACHED FUNCTION TO PREPARE EXPORT DATA ---
@st.cache_data(ttl=3600)
def prepare_export_data(df_full, ticker_col, company_col):
    """
    Collects all required financial and technical metrics for the Excel export.
    """
    tickers = df_full[ticker_col].dropna().unique()
    export_list = []
    
    # Create a mapping for company names
    company_map = df_full.set_index(ticker_col)[company_col].to_dict()

    for ticker in tickers: 
        try:
            history_weekly, info, _, history_daily = get_stock_data(ticker)
            
            if history_weekly.empty or len(history_weekly) < 52:
                # Append placeholder data if fetching or history is insufficient
                export_list.append({
                    'Ticker': ticker,
                    'Signal': 'N/A (Data Error)',
                    'Current_Price': np.nan,
                    'Intrinsic_Value': np.nan,
                    'Avg_52_Day': np.nan,
                    'Buy_Price_20W_SMA': np.nan,
                    'Market_Cap': np.nan,
                    'Stock_PE': np.nan,
                    'RSI_Value': np.nan,
                })
                continue
                
            # Calculations
            indicators, recommendation, _ = calculate_swing_trade_analysis(history_weekly)
            intrinsic_value = calculate_graham_intrinsic_value(info, info, bond_yield=7.5) # info for financials placeholder
            avg_52_day = calculate_52_day_average(history_daily)
            
            # Data extraction
            current_price = history_weekly['Close'].iloc[-1]
            buy_price = indicators.get("20W SMA")
            rsi_value = indicators.get("RSI (14)")
            market_cap = info.get('marketCap')
            stock_pe = info.get('trailingPE')
            
            export_list.append({
                'Ticker': ticker,
                'Signal': recommendation,
                'Current_Price': current_price,
                'Intrinsic_Value': intrinsic_value,
                'Avg_52_Day': avg_52_day,
                'Buy_Price_20W_SMA': buy_price,
                'Market_Cap': market_cap,
                'Stock_PE': stock_pe,
                'RSI_Value': rsi_value,
            })
        except Exception: 
            export_list.append({'Ticker': ticker, 'Signal': 'N/A (Fetch Error)', 
                                'Current_Price': np.nan, 'Intrinsic_Value': np.nan, 'Avg_52_Day': np.nan,
                                'Buy_Price_20W_SMA': np.nan, 'Market_Cap': np.nan, 'Stock_PE': np.nan, 'RSI_Value': np.nan})
            continue
            
    df_export = pd.DataFrame(export_list)
    
    # Merge Company Name
    df_export['Company Name'] = df_export['Ticker'].map(company_map).fillna(df_export['Ticker'])
    
    # Final column selection and renaming
    df_export.insert(0, 'S. No.', range(1, 1 + len(df_export)))
    df_export = df_export.rename(columns={
        'Current_Price': 'Current Market Price',
        'Intrinsic_Value': 'Intrinsic Value',
        'Avg_52_Day': 'Last 52 Day Avg Price',
        'Buy_Price_20W_SMA': 'Buy Price (20W SMA)',
        'Market_Cap': 'Market Cap',
        'Stock_PE': 'Stock P/E',
        'RSI_Value': 'RSI Value',
        'Signal': 'Swing Signal'
    })
    
    # Select and reorder the 10 final columns
    final_columns = [
        'S. No.', 'Company Name', 'Current Market Price', 'Intrinsic Value', 
        'Last 52 Day Avg Price', 'Buy Price (20W SMA)', 'Market Cap', 'Stock P/E', 
        'RSI Value', 'Swing Signal'
    ]
    
    return df_export[final_columns]

# Helper function to convert DataFrame to Excel bytes
def to_excel(df):
    """Converts a pandas DataFrame to an Excel file stored in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Stock_Data_Export')
    return output.getvalue()


@st.cache_data(ttl=3600)
def calculate_all_stock_signals(df, ticker_col):
    """
    Calculates swing signals and necessary ranking metrics (Current Price, 20W SMA) for ALL available stocks.
    """
    tickers = df[ticker_col].dropna().unique()
    total_stocks = len(tickers)
    all_signals = []
    analyzed_count = 0
    
    for ticker in tickers: 
        try:
            history, _, _, _ = get_stock_data(ticker)
            if not history.empty and len(history) < 52:
                # Handle insufficient data cleanly for ranking/signal display
                all_signals.append({'Ticker': ticker, 'Signal': 'N/A (Data Error)', 'Price_Diff': np.nan})
                continue
            
            indicators, recommendation, _ = calculate_swing_trade_analysis(history)
            current_price = history['Close'].iloc[-1]
            buy_price_20w_sma = indicators.get('20W SMA')

            # Calculate the difference: Current Price - Recommended Buy Price (for ranking)
            price_diff = current_price - buy_price_20w_sma if buy_price_20w_sma is not None else np.nan

            all_signals.append({
                'Ticker': ticker, 
                'Signal': recommendation,
                'Current_Price': current_price,
                'Buy_Price_20W_SMA': buy_price_20w_sma,
                'Price_Diff': price_diff
            })
            analyzed_count += 1
        except Exception: 
            all_signals.append({'Ticker': ticker, 'Signal': 'N/A (Fetch Error)', 'Price_Diff': np.nan})
            continue
    
    return pd.DataFrame(all_signals), total_stocks, analyzed_count

def display_stock_analysis(ticker):
    """Displays analysis for a single stock."""
    try:
        # Fetching data is cached for speed
        history, info, financials, daily_history = get_stock_data(ticker)
        if history.empty:
            st.warning(f"Could not fetch price history for **{ticker}**. Skipping.")
            return

        swing_indicators, swing_recommendation, _ = calculate_swing_trade_analysis(history)
        intrinsic_value = calculate_graham_intrinsic_value(info, financials)
        
        current_price = history['Close'].iloc[-1]
        
        # --- NEW: 52-DAY AVERAGE METRIC ---
        avg_52_day = calculate_52_day_average(daily_history)
        avg_52_day_display = f"₹{avg_52_day:,.2f}" if avg_52_day is not None else "N/A"
        
        # Determine delta (color) based on comparison with Current Price
        avg_delta_color = "off" # Default gray
        avg_delta_text = "N/A"

        if avg_52_day is not None and avg_52_day != 0:
            price_diff_abs = abs(current_price - avg_52_day)
            price_diff_percent = ((current_price - avg_52_day) / avg_52_day) * 100
            
            if current_price > avg_52_day:
                # Current price is ABOVE 52D Avg -> Show RED percentage AND absolute diff (User Request)
                # Color: Red (inverse)
                avg_delta_color = "inverse" 
                avg_delta_text = f"↑ {price_diff_percent:,.2f}% (+₹{price_diff_abs:,.2f})"
            elif current_price < avg_52_day:
                # Current price is BELOW 52D Avg -> Show GREEN absolute diff AND percentage (User Request)
                # Color: Green (normal)
                avg_delta_color = "normal" 
                avg_delta_text = f"↓ -₹{price_diff_abs:,.2f} ({price_diff_percent:,.2f}%)"
            else:
                avg_delta_text = "At 52D Avg"
        elif avg_52_day == 0:
            avg_delta_text = "Avg Price is Zero"

        # --- Swing Signal Text Color Enhancement (Reduced Font Size) ---
        def get_signal_style(signal):
            if 'Buy' in signal:
                return "🟢 <span style='color: #2ECC71; font-weight: bold;'>{signal}</span>"
            elif 'Hold' in signal:
                return "🟡 <span style='color: #F39C12; font-weight: bold;'>{signal}</span>"
            else:
                return "🔴 <span style='color: #E74C3C; font-weight: bold;'>{signal}</span>"
        
        styled_signal = get_signal_style(swing_recommendation).format(signal=swing_recommendation)

        # --- NEW: Calculate and Format Price Variation for Header ---
        price_variation_text = ""
        buy_price = swing_indicators.get('20W SMA') if swing_indicators else None
        
        if buy_price is not None:
            variation = current_price - buy_price
            sign = "+" if variation >= 0 else ""
            price_variation_text = f", [{sign}{variation:,.2f} Rs from Recommended Price]"
            
        # FIX: Update the header with the calculated price variation
        st.header(f"Analysis for: {info.get('shortName', ticker)} ({ticker}){price_variation_text}", divider='rainbow')


        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Current Price", f"₹{current_price:,.2f}")
        
        # FIX 1: Display Swing Signal using markdown with a size similar to other metrics (1.5em)
        m_col2.markdown(f"**Swing Signal**", unsafe_allow_html=True)
        m_col2.markdown(f"<p style='font-size: 1.5em;'>{styled_signal}</p>", unsafe_allow_html=True) 
        
        m_col3.metric("Intrinsic Value", f"₹{intrinsic_value:,.2f}" if intrinsic_value else "N/A")
        
        # FIX 2 & 3: Last 52 Days Average (with combined delta text for absolute and percent)
        m_col4.metric(label="Last 52 Days Average", value=avg_52_day_display, delta=avg_delta_text, delta_color=avg_delta_color)
        m_col5.metric("Buy Price (≈20W SMA)", f"₹{swing_indicators['20W SMA']:,.2f}" if swing_indicators else "N/A")
        
        st.divider()

        # ===============================================
        # CHART AND ANALYSIS SECTION
        # ===============================================
        chart_col, analysis_col = st.columns([2, 1])

        with chart_col:
            st.subheader("Weekly Price Chart with Fibonacci Retracement")
            fib_levels, _, is_uptrend, high_price, low_price = calculate_fibonacci_levels(history)
            
            # Setup Plotly Candlestick Chart
            fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='Price')])
            history['SMA_20W'] = history['Close'].rolling(window=20).mean()
            history['SMA_50W'] = history['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(x=history.index, y=history['SMA_20W'], mode='lines', name='20W SMA', line=dict(color='orange', width=1.5)))
            fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50W'], mode='lines', name='50W SMA', line=dict(color='purple', width=1.5)))
            
            # Add Fibonacci lines to chart
            colors = ['red', 'orange', 'yellow', 'green']
            for i, (level, price) in enumerate(fib_levels.items()):
                fig.add_hline(y=price, line_width=1, line_dash="dash", line_color=colors[i], annotation_text=f"Fib {level}", annotation_position="bottom right")

            fig.update_layout(height=600, yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        
        with analysis_col:
            st.subheader("Swing Signal Reasoning")
            _, _, swing_reasoning = calculate_swing_trade_analysis(history)
            st.info(swing_reasoning)

            st.subheader("Fibonacci Retracement Analysis")
            _, fib_signal, _, _, _ = calculate_fibonacci_levels(history)
            st.info(fib_signal)

            st.subheader("Key Financial Ratios (Screener.in)")
            screener_data, screener_status = scrape_screener_data(ticker)
            if screener_status == "Success":
                df_screener = pd.DataFrame(screener_data.items(), columns=['Ratio', 'Value'])
                st.dataframe(df_screener, use_container_width=True, hide_index=True)
            else:
                st.warning(f"Could not scrape data ({screener_status}).")

    except Exception as e:
        st.error(f"An error occurred while processing **{ticker}**: {e}")
    
    # ===============================================
    # NEW: RECENT NEWS ARTICLES DISPLAY (Full Width)
    # ===============================================
    st.subheader("Recent News Articles (NewsAPI.org)", divider='red')
    news_result = get_newsapi_articles(ticker)
    
    if news_result['status'] == 'Success' and news_result['articles']:
        for article in news_result['articles']:
            st.markdown(
                f"**[{article['publishedAt']}]** [**{article['title']}**]({article['url']}) "
                f"— *Source: {article['source']}*", 
                unsafe_allow_html=True
            )
    elif news_result['status'] == 'Success':
        st.info(f"No recent articles found for {ticker} using NewsAPI.")
    else:
        st.warning(news_result['message'])
        
    # NSE Website Link (as requested previously)
    st.markdown(f"**🔗 External Link:** [View {ticker} on NSE India](https://www.nseindia.com/get-quotes/equity?symbol={ticker})")


# ======================================================================================
# STREAMLIT UI & LOGIC
# ======================================================================================
EXCEL_FILE_PATH = "SELECTED STOCKS 22FEB2025.xlsx"
TICKER_COLUMN_NAME = "NSE SYMBOL"
INDUSTRY_COLUMN_NAME = "INDUSTRY"
COMPANY_COLUMN_NAME = "COMPANY" # Assuming a 'COMPANY' column exists in your Excel

# Initialize Session State Variables
if 'current_stock_index' not in st.session_state: st.session_state.current_stock_index = 0
if 'ticker_list' not in st.session_state: st.session_state.ticker_list = []
if 'selected_industry' not in st.session_state: st.session_state.selected_industry = "All Industries"
if 'all_signals_df' not in st.session_state: st.session_state.all_signals_df = pd.DataFrame()
if 'total_stocks' not in st.session_state: st.session_state.total_stocks = 0
if 'analyzed_stocks' not in st.session_state: st.session_state.analyzed_stocks = 0

if not os.path.exists(EXCEL_FILE_PATH):
    st.error(f"Error: The file '{EXCEL_FILE_PATH}' was not found.")
else:
    try:
        df_full = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
        industries = ["All Industries"] + sorted(df_full[INDUSTRY_COLUMN_NAME].dropna().unique().tolist())
    except Exception as e:
        st.error(f"Could not read the Excel file. Error: {e}")
        st.stop()
    
    with st.sidebar:
        st.header("⚙️ Analysis Filter")
        
        # 1. Industry Selection
        new_selected_industry = st.selectbox("Select an Industry:", industries, key='sidebar_select')
        
        # Check if selection changed to trigger a refresh
        if new_selected_industry != st.session_state.selected_industry:
            st.session_state.selected_industry = new_selected_industry

        button_clicked = st.button("🚀 Analyze Selection", type="primary")

        # 2. Comprehensive Signal Calculation (Cached and runs once initially)
        if st.session_state.all_signals_df.empty or button_clicked:
            with st.spinner("Calculating swing signals and ranking metrics..."):
                # Run the cached function to get all signals and prices
                df_signals, total_stocks, analyzed_stocks = calculate_all_stock_signals(df_full, TICKER_COLUMN_NAME)
                st.session_state.all_signals_df = df_signals
                st.session_state.total_stocks = total_stocks
                st.session_state.analyzed_stocks = analyzed_stocks
                
                if button_clicked:
                    # Filter the main analysis list based on the selected industry
                    if st.session_state.selected_industry != "All Industries":
                        df_filtered = df_full[df_full[INDUSTRY_COLUMN_NAME] == st.session_state.selected_industry]
                    else:
                        df_filtered = df_full
                        
                    # --- NEW RANKING LOGIC (Applied only for specific industries) ---
                    if st.session_state.selected_industry != "All Industries":
                        
                        # Filter signals for the selected industry
                        industry_signals = st.session_state.all_signals_df[st.session_state.all_signals_df['Ticker'].isin(df_filtered[TICKER_COLUMN_NAME].unique())].copy()
                        
                        # Clean up N/A values for ranking
                        df_rankable = industry_signals.dropna(subset=['Price_Diff']).copy()
                        df_non_rankable = industry_signals[industry_signals['Price_Diff'].isna()].copy()

                        # Sort: Ascending order of Price_Diff (Current Price - 20W SMA)
                        # The smallest/most negative difference (Current Price is far below 20W SMA) is Rank #1
                        df_rankable = df_rankable.sort_values(by='Price_Diff', ascending=True)

                        # Concatenate and assign ranks
                        ranked_df = pd.concat([df_rankable, df_non_rankable], ignore_index=True)
                        ranked_df['Rank'] = np.nan
                        ranked_df.loc[:len(df_rankable)-1, 'Rank'] = range(1, len(df_rankable) + 1)
                        
                        # Update the ticker list to reflect the new rank order
                        st.session_state.ticker_list = ranked_df['Ticker'].tolist()
                    else:
                        st.session_state.ticker_list = df_filtered[TICKER_COLUMN_NAME].dropna().unique().tolist()
                        
                    st.session_state.current_stock_index = 0
                    st.rerun() # Rerun to display the first stock in the newly filtered list

        # 3. Conditional Sidebar Display
        if not st.session_state.all_signals_df.empty:
            
            # --- DISPLAY: ALL INDUSTRIES (Quick Snapshot) ---
            if st.session_state.selected_industry == "All Industries":
                st.subheader("Quick Signals Snapshot", divider='rainbow')
                st.markdown(f"**Market Coverage:** Analyzed **{st.session_state.analyzed_stocks}** of **{st.session_state.total_stocks}** tickers.")
                
                buy_signals = st.session_state.all_signals_df[st.session_state.all_signals_df['Signal'].isin(['Strong Buy', 'Buy'])].head(3)
                sell_signals = st.session_state.all_signals_df[st.session_state.all_signals_df['Signal'] == 'Sell / Avoid'].head(3)

                st.markdown("**Top 3 Buy Signals**")
                if not buy_signals.empty:
                    for _, row in buy_signals.iterrows(): st.success(f"**{row['Ticker']}**: {row['Signal']}")
                else: st.info("No strong buy signals found.")
                
                st.markdown("**Top 3 Sell Signals**")
                if not sell_signals.empty:
                    for _, row in sell_signals.iterrows(): st.error(f"**{row['Ticker']}**: {row['Signal']}")
                else: st.info("No strong sell signals found.")
                
            # --- DISPLAY: SPECIFIC INDUSTRY (Full List with New Rank and Styling) ---
            else:
                st.subheader(f"{st.session_state.selected_industry} Signals (Ranked by Buy Opportunity)", divider='rainbow')
                
                # Retrieve the current ranked list from session state
                current_ranked_tickers = st.session_state.ticker_list
                
                if current_ranked_tickers:
                    # Re-filter and prepare for display based on the session state order
                    industry_signals = st.session_state.all_signals_df[st.session_state.all_signals_df['Ticker'].isin(current_ranked_tickers)].copy()
                    
                    # 1. Re-sort the display DF using the session state order
                    industry_signals = industry_signals.set_index('Ticker').reindex(current_ranked_tickers).reset_index()
                    
                    # 2. Assign Rank (based on the index of non-NaN Price_Diff values)
                    df_rankable_count = industry_signals['Price_Diff'].notna().sum()
                    industry_signals['Rank'] = np.nan
                    industry_signals.loc[:df_rankable_count-1, 'Rank'] = range(1, df_rankable_count + 1)
                    
                    # Final Rank Cleanup
                    industry_signals['Rank'] = industry_signals['Rank'].fillna(0).astype(int)
                    industry_signals.loc[industry_signals['Rank'] == 0, 'Rank'] = '-'
                    industry_signals['Rank'] = industry_signals['Rank'].astype(str)

                    # 3. Format the Price Difference for display (including sign)
                    industry_signals['Difference'] = industry_signals['Price_Diff'].apply(
                        lambda x: f"₹{x:,.2f}" if pd.notna(x) else '-'
                    )

                    # Define the styling function for coloring the 'Recommendation' column (Rich Colors)
                    def color_signals(s):
                        if s == 'Strong Buy' or s == 'Buy':
                            return 'background-color: #2ECC71; color: white; font-weight: bold;' 
                        elif s == 'Hold / Monitor':
                            return 'background-color: #F39C12; color: #333333;' 
                        elif s == 'Sell / Avoid':
                            return 'background-color: #E74C3C; color: white; font-weight: bold;' 
                        else:
                            return 'background-color: #BDC3C7; color: #383d41;'

                    # Define the styling function for the difference column
                    def style_difference_cell(val):
                        if val == '-':
                            return 'background-color: #BDC3C7; color: #383d41;' # Gray for N/A

                        # Extract the numeric part safely (handling signs and currency)
                        try:
                            # Clean the string by removing non-numeric characters except for the leading sign
                            numeric_str = val.replace('₹', '').replace(',', '').strip()
                            numeric_val = float(numeric_str)
                        except ValueError:
                            return '' # Default style if parsing fails

                        if numeric_val <= 0:
                            # Negative difference (Good/Neutral Opportunity) -> Green
                            return 'background-color: #d4edda; color: #155724; font-weight: bold;' 
                        else:
                            # Positive difference (Run Up) -> Red
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'

                    # Prepare for display, including the new 'Difference' column
                    styled_df = industry_signals[['Rank', 'Ticker', 'Signal', 'Difference']].rename(columns={
                        'Ticker': 'Stock', 
                        'Signal': 'Recommendation',
                        'Difference': 'Diff (Curr - SMA)'
                    })
                    
                    st.caption(f"Showing {len(industry_signals)} stocks in this industry. Rank based on proximity to 20W SMA.")

                    # Apply styling and display the DataFrame
                    st.dataframe(
                        styled_df.style
                            .applymap(color_signals, subset=['Recommendation'])
                            .applymap(style_difference_cell, subset=['Diff (Curr - SMA)']),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info(f"No valid signal data found for stocks in the {st.session_state.selected_industry} industry.")
        
        # --- NEW EXPORT MODULE ---
        st.markdown("---")
        st.subheader("Data Export")
        
        # Check if df_full is available and proceed to prepare export data
        if 'df_full' in locals():
            with st.spinner("Preparing export data..."):
                df_export_ready = prepare_export_data(df_full, TICKER_COLUMN_NAME, COMPANY_COLUMN_NAME)
                excel_data = to_excel(df_export_ready)

            st.download_button(
                label="⬇️ EXPORT All Stock Data to Excel",
                data=excel_data,
                file_name=f"Stock_Analysis_Export_{pd.Timestamp('today').strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.warning("Cannot export data: Excel file not loaded.")
        
        # --- COPYRIGHT NOTICE (Mild Color, Small Font) ---
        st.markdown(
            """
            <div style='font-size: 0.7rem; color: #a9a9a9; margin-top: 50px; text-align: center; padding-top: 10px; border-top: 1px solid #33333340;'>
                © 2025 Mock Test Platform, by Sivasethupathi. All Rights Reserved and strictly for internal purpose.
            </div>
            """, 
            unsafe_allow_html=True
        )


    # --- NEW: TOP RIGHT SEARCH BAR LOGIC ---
    # Create the combined list for search suggestions
    all_tickers_names = []
    df_search_mapping = df_full[[TICKER_COLUMN_NAME, COMPANY_COLUMN_NAME]].dropna()
    for _, row in df_search_mapping.iterrows():
        # Format: "COMPANY NAME (TICKER)" for easy selection and ticker extraction
        all_tickers_names.append(f"{row[COMPANY_COLUMN_NAME]} ({row[TICKER_COLUMN_NAME]})")
    
    # We place the search box in two columns to give it a "top right" feel.
    search_main_col, search_box_col = st.columns([1, 4])
    
    with search_box_col:
        selected_search = st.selectbox(
            "Quick Stock Search:",
            options=[''] + sorted(all_tickers_names),
            index=0,
            key='quick_search_input',
            label_visibility="collapsed",
            placeholder="Search Company Name or Ticker..."
        )
    
    # Handle search selection immediately
    if selected_search:
        # Extract the Ticker from the selected string
        match = re.search(r'\((.*?)\)', selected_search)
        if match:
            search_ticker = match.group(1).strip()
            
            # Check if we need to update the session state
            if not st.session_state.ticker_list or st.session_state.ticker_list[st.session_state.current_stock_index] != search_ticker:
                # Overwrite the ticker list and index to display the search result
                st.session_state.ticker_list = [search_ticker]
                st.session_state.current_stock_index = 0
                st.session_state.selected_industry = "Search Result" # Indicate a single search result is active
                st.rerun()

    # --- MAIN CONTENT LAYOUT ---
    if st.session_state.ticker_list:
        current_ticker = st.session_state.ticker_list[st.session_state.current_stock_index]
        
        # Navigation Buttons logic
        is_single_stock_result = len(st.session_state.ticker_list) == 1 and st.session_state.selected_industry == "Search Result"
        
        # Only show navigation buttons if it's NOT a single search result
        if not is_single_stock_result:
            col1, col2, col3 = st.columns([1.5, 5, 1.5])
            with col1:
                if st.button("⬅️ Previous Stock", use_container_width=True, disabled=(st.session_state.current_stock_index == 0)):
                    st.session_state.current_stock_index -= 1; st.rerun()
            with col2:
                st.markdown(f"<p style='text-align: center; font-size: 1.1em;'>Displaying <b>{st.session_state.current_stock_index + 1}</b> of <b>{len(st.session_state.ticker_list)}</b> stocks in <b>{st.session_state.selected_industry}</b></p>", unsafe_allow_html=True)
            with col3:
                if st.button("Next Stock ➡️", use_container_width=True, disabled=(st.session_state.current_stock_index >= len(st.session_state.ticker_list) - 1)):
                    st.session_state.current_stock_index += 1; st.rerun()
        else:
             st.markdown(f"<p style='text-align: center; font-size: 1.1em;'>Displaying Search Result: <b>{current_ticker}</b></p>", unsafe_allow_html=True)

        # Display the detailed analysis
        display_stock_analysis(current_ticker)
    else:
        st.info("Select an industry from the sidebar and click the 'Analyze Selection' button to view the stock list and detailed analysis, or use the Quick Stock Search above.")
