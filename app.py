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

# ======================================================================================
# CONFIGURATION & HEADER
# ======================================================================================
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Stock Analyzer | S N   T R A D E R")
st.markdown("Select an industry from your Excel file to get a consolidated analysis, including financial ratios from **Screener.in** and a detailed **Swing Trading** recommendation with Fibonacci levels.")

# ======================================================================================
# DATA FETCHING & CALCULATION FUNCTIONS
# ======================================================================================

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker):
    """Fetches all necessary data for a stock from yfinance."""
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

# --- NEW: FIBONACCI RETRACEMENT ANALYSIS ---
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
        return None, "Insufficient Data", "Not enough weekly data for full analysis."

    close = history['Close']
    price = close.iloc[-1]
    
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    rsi_14 = RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_indicator = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    macd_hist = macd_indicator.macd_diff().iloc[-1]
    bb_indicator = BollingerBands(close, window=20, window_dev=2)
    bb_high = bb_indicator.bollinger_hband().iloc[-1]
    bb_low = bb_indicator.bollinger_lband().iloc[-1]
    obv_indicator = OnBalanceVolumeIndicator(close, history['Volume'])
    obv_slope = obv_indicator.on_balance_volume().diff().rolling(window=5).mean().iloc[-1]
    atr_14 = AverageTrueRange(history['High'], history['Low'], close, window=14).average_true_range().iloc[-1]

    indicators = {"20W SMA": sma_20, "50W SMA": sma_50, "RSI (14)": rsi_14, "MACD Line": macd_line, "MACD Signal": macd_signal}

    score = 0
    reasons = []
    
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
    
    if score >= 7: recommendation = "Strong Buy"
    elif score >= 5: recommendation = "Buy"
    elif score >= 3: recommendation = "Hold / Monitor"
    else: recommendation = "Sell / Avoid"
        
    return indicators, recommendation, "\n\n".join(reasons)

@st.cache_data(ttl=3600)
def calculate_quick_signals(df, ticker_col):
    """Analyzes all stocks to find top buy/sell signals."""
    tickers = df[ticker_col].dropna().unique()
    all_signals = []
    for ticker in tickers[:50]: 
        try:
            history, _, _, _ = get_stock_data(ticker)
            if not history.empty and len(history) >= 52:
                _, recommendation, _ = calculate_swing_trade_analysis(history)
                all_signals.append({'Ticker': ticker, 'Signal': recommendation})
        except Exception: continue
    if not all_signals: return pd.DataFrame(), pd.DataFrame()
    signals_df = pd.DataFrame(all_signals)
    buy_signals = signals_df[signals_df['Signal'].isin(['Strong Buy', 'Buy'])].head(3)
    sell_signals = signals_df[signals_df['Signal'] == 'Sell / Avoid'].head(3)
    return buy_signals, sell_signals

def display_stock_analysis(ticker):
    """Displays analysis for a single stock."""
    try:
        history, info, financials, daily_history = get_stock_data(ticker)
        if history.empty:
            st.warning(f"Could not fetch price history for **{ticker}**. Skipping.")
            return

        st.header(f"Analysis for: {info.get('shortName', ticker)} ({ticker})", divider='rainbow')

        swing_indicators, swing_recommendation, _ = calculate_swing_trade_analysis(history)
        intrinsic_value = calculate_graham_intrinsic_value(info, financials)
        
        current_price = history['Close'].iloc[-1]
        price_mar_28, date_mar_28 = get_price_on_date(daily_history, '2025-03-28')
        move_fy_percent = None
        fy_delta_text = "N/A"
        if price_mar_28 and current_price:
            move_fy_percent = ((current_price - price_mar_28) / price_mar_28) * 100
            fy_delta_text = f"from ₹{price_mar_28:,.2f} on {date_mar_28.strftime('%d-%b-%Y')}"

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Current Price", f"₹{current_price:,.2f}")
        m_col2.metric("Swing Signal", swing_recommendation)
        m_col3.metric("Intrinsic Value", f"₹{intrinsic_value:,.2f}" if intrinsic_value else "N/A")
        m_col4.metric(label="Move within FY", value=f"{move_fy_percent:.2f}%" if move_fy_percent is not None else "N/A", delta=fy_delta_text)
        m_col5.metric("Buy Price (≈20W SMA)", f"₹{swing_indicators['20W SMA']:,.2f}" if swing_indicators else "N/A")

        st.divider()

        chart_col, analysis_col = st.columns([2, 1])

        with chart_col:
            st.subheader("Weekly Price Chart with Fibonacci Retracement")
            fib_levels, _, is_uptrend, high_price, low_price = calculate_fibonacci_levels(history)
            
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

            # --- NEW: Fibonacci section ---
            st.subheader("Fibonacci Retracement Analysis")
            _, fib_signal, _, _, _ = calculate_fibonacci_levels(history)
            st.info(fib_signal)

            st.subheader("Key Financial Ratios")
            screener_data, screener_status = scrape_screener_data(ticker)
            if screener_status == "Success":
                df_screener = pd.DataFrame(screener_data.items(), columns=['Ratio', 'Value'])
                st.dataframe(df_screener, use_container_width=True, hide_index=True)
            else:
                st.warning(f"Could not scrape data ({screener_status}).")

    except Exception as e:
        st.error(f"An error occurred while processing **{ticker}**: {e}")

# ======================================================================================
# STREAMLIT UI & LOGIC
# ======================================================================================
EXCEL_FILE_PATH = "SELECTED STOCKS 22FEB2025.xlsx"
TICKER_COLUMN_NAME = "NSE SYMBOL"
INDUSTRY_COLUMN_NAME = "INDUSTRY"

if 'current_stock_index' not in st.session_state: st.session_state.current_stock_index = 0
if 'ticker_list' not in st.session_state: st.session_state.ticker_list = []
if 'quick_signals_calculated' not in st.session_state: st.session_state.quick_signals_calculated = False

if not os.path.exists(EXCEL_FILE_PATH):
    st.error(f"Error: The file '{EXCEL_FILE_PATH}' was not found.")
else:
    with st.sidebar:
        st.header("⚙️ Analysis Filter")
        try:
            df_full = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
            industries = ["All Industries"] + sorted(df_full[INDUSTRY_COLUMN_NAME].dropna().unique().tolist())
            selected_industry = st.selectbox("Select an Industry:", industries)
            
            if st.button("🚀 Analyze Selected Industry", type="primary"):
                df_filtered = df_full[df_full[INDUSTRY_COLUMN_NAME] == selected_industry] if selected_industry != "All Industries" else df_full
                st.session_state.ticker_list = df_filtered[TICKER_COLUMN_NAME].dropna().unique().tolist()
                st.session_state.current_stock_index = 0
                st.session_state.quick_signals_calculated = True
                
            if st.session_state.quick_signals_calculated:
                with st.spinner("Calculating market snapshot..."):
                    buy_signals, sell_signals = calculate_quick_signals(df_full, TICKER_COLUMN_NAME)
                
                st.subheader("Quick Signals Snapshot", divider='rainbow')
                st.markdown("**Top 3 Buy Signals**")
                if not buy_signals.empty:
                    for _, row in buy_signals.iterrows(): st.success(f"**{row['Ticker']}**: {row['Signal']}")
                else: st.info("No strong buy signals found.")
            
                st.markdown("**Top 3 Sell Signals**")
                if not sell_signals.empty:
                    for _, row in sell_signals.iterrows(): st.error(f"**{row['Ticker']}**: {row['Signal']}")
                else: st.info("No strong sell signals found.")
        except Exception as e:
            st.error(f"Could not read the Excel file. Error: {e}")

    if st.session_state.ticker_list:
        current_ticker = st.session_state.ticker_list[st.session_state.current_stock_index]
        
        col1, col2, col3 = st.columns([1.5, 5, 1.5])
        with col1:
            if st.button("⬅️ Previous Stock", use_container_width=True, disabled=(st.session_state.current_stock_index == 0)):
                st.session_state.current_stock_index -= 1; st.rerun()
        with col2:
            st.markdown(f"<p style='text-align: center; font-size: 1.1em;'>Displaying <b>{st.session_state.current_stock_index + 1}</b> of <b>{len(st.session_state.ticker_list)}</b> stocks</p>", unsafe_allow_html=True)
        with col3:
            if st.button("Next Stock ➡️", use_container_width=True, disabled=(st.session_state.current_stock_index >= len(st.session_state.ticker_list) - 1)):
                st.session_state.current_stock_index += 1; st.rerun()

        display_stock_analysis(current_ticker)
    else:
        st.info("Select an industry from the sidebar and click the 'Analyze' button to begin.")

