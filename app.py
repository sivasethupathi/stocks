import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import plotly.graph_objects as go
import os
import requests
from bs4 import BeautifulSoup
import re

# ======================================================================================
# CONFIGURATION & HEADER
# ======================================================================================
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Automated Stock Analyzer")
st.markdown("""
This tool offers two modes of analysis:
1.  **Intrinsic Value Analyzer**: Calculates intrinsic value using Benjamin Graham's formula and technicals.
2.  **Screener Data Extractor**: Scrapes key financial ratios for your stocks from `screener.in`.
""")

# ======================================================================================
# NEW: SCREENER.IN SCRAPING FUNCTIONS
# ======================================================================================

def clean_value(value_str):
    """Cleans the scraped string value and converts it to a float."""
    if value_str is None:
        return None
    # Remove commas, Cr., %, and other characters, then convert to float
    cleaned_str = re.sub(r'[^\d.]', '', value_str)
    try:
        return float(cleaned_str)
    except (ValueError, TypeError):
        return None

def scrape_screener_data(ticker):
    """Scrapes key financial data for a given ticker from screener.in."""
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None, f"Failed to load page (Status: {response.status_code})"
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # --- Extract data from the top ratios section ---
    data = {}
    ratio_list = soup.select_one('#top-ratios')
    if not ratio_list:
        return None, "Could not find the ratios section."
        
    for li in ratio_list.find_all('li'):
        name_span = li.select_one('.name')
        value_span = li.select_one('.nowrap.value .number')
        if name_span and value_span:
            name = name_span.get_text(strip=True)
            value = value_span.get_text(strip=True)
            data[name] = value

    # --- Map scraped data to the desired fields ---
    output = {
        'Ticker': ticker,
        'Market Cap': clean_value(data.get('Market Cap')),
        'Current Price': clean_value(data.get('Current Price')),
        'High / Low': data.get('High / Low'), # Kept as string
        'Stock P/E': clean_value(data.get('Stock P/E')),
        'Book Value': clean_value(data.get('Book Value')),
        'Dividend Yield': clean_value(data.get('Dividend Yield')),
        'ROCE': clean_value(data.get('ROCE')),
        'ROE': clean_value(data.get('ROE')),
        'Face Value': clean_value(data.get('Face Value')),
        'EV/EBITDA': clean_value(data.get('EV/EBITDA')),
        'Debt to equity': clean_value(data.get('Debt to equity'))
    }
    
    # --- Extract other details that might be outside the main list ---
    # Note: This part can be brittle and may need updates if screener.in changes layout.
    # We will get Enterprise Value from the Quarterly Results table if possible.
    try:
        # This is a more complex scrape and might not always work.
        # For this version, we focus on the reliable top ratios.
        output['Enterprise Value'] = 'N/A'
        output['Current Ratio'] = 'N/A'
        output['PEG Ratio'] = 'N/A'
    except Exception:
        pass

    return output, "Success"


# ======================================================================================
# CORE CALCULATION FUNCTIONS (from previous version)
# ======================================================================================

def calculate_graham_intrinsic_value(info, financials, bond_yield):
    """
    Calculates the intrinsic value of a stock using Benjamin Graham's formula.
    """
    try:
        eps = info.get('trailingEps')
        if not eps or eps <= 0: return None, "EPS is zero or negative."
        net_income = financials.loc['Net Income']
        if net_income.isnull().all() or len(net_income.dropna()) < 2: return None, "Not enough Net Income data."
        growth_rates = net_income.pct_change().dropna()
        avg_growth_rate = np.mean(growth_rates)
        g = min(avg_growth_rate * 100, 15.0) 
        if g < 0: g = 0
        intrinsic_value = (eps * (8.5 + 2 * g) * 4.4) / bond_yield
        return intrinsic_value, "Success"
    except (KeyError, IndexError, TypeError) as e:
        return None, f"Missing data for calculation: {e}"

def get_technical_indicators(history):
    """Calculates key technical indicators (50D SMA, 200D SMA, 14D RSI)."""
    try:
        if len(history) < 200: return {}, "Insufficient data (<200 days)."
        sma_50 = SMAIndicator(close=history['Close'], window=50).sma_indicator().iloc[-1]
        sma_200 = SMAIndicator(close=history['Close'], window=200).sma_indicator().iloc[-1]
        rsi_14 = RSIIndicator(close=history['Close'], window=14).rsi().iloc[-1]
        return {'SMA_50': sma_50, 'SMA_200': sma_200, 'RSI_14': rsi_14}, "Success"
    except Exception as e:
        return {}, f"Error in technical calculation: {e}"

def generate_signal(row):
    """Generates a trading signal based on valuation and technicals."""
    iv = row['Intrinsic Value']
    price = row['Current Price']
    sma_50 = row['SMA_50']
    sma_200 = row['SMA_200']
    if any(pd.isna(val) for val in [iv, price, sma_50, sma_200]): return "Not Available"
    is_undervalued = price < iv
    in_uptrend = price > sma_50 and price > sma_200
    if is_undervalued and in_uptrend: return "Strong Buy"
    if is_undervalued and price > sma_50: return "Buy"
    if is_undervalued and not in_uptrend: return "Monitor (Undervalued, but in Downtrend)"
    if not is_undervalued and in_uptrend: return "Hold (Overvalued, but in Uptrend)"
    if not is_undervalued and not in_uptrend: return "Sell"
    return "Hold"

# ======================================================================================
# STREAMLIT UI & LOGIC
# ======================================================================================

# --- Sidebar is now common for both tabs ---
with st.sidebar:
    st.header("⚙️ File Configuration")
    EXCEL_FILE_PATH = "nse_tickers.xlsx"
    TICKER_COLUMN_NAME = "Ticker"
    st.info(f"Reading tickers from local file:\n`{EXCEL_FILE_PATH}`")
    st.write(f"Expecting a column named: `{TICKER_COLUMN_NAME}`")
    file_exists = os.path.exists(EXCEL_FILE_PATH)
    if not file_exists:
        st.error(f"Error: The file '{EXCEL_FILE_PATH}' was not found in the app's root directory.")

# --- Create Tabs for different functionalities ---
tab1, tab2 = st.tabs(["Screener Data Extractor", "Intrinsic Value Analyzer"])

# --- TAB 1: SCREENER DATA EXTRACTOR ---
with tab1:
    st.header("📊 Screener.in Data Extractor")
    st.markdown("This tool scrapes the latest financial ratios from `screener.in` for each stock in your Excel file.")
    
    extract_button = st.button("🚀 Extract Data from Screener", disabled=(not file_exists))

    if extract_button:
        try:
            df = pd.read_excel(EXCEL_FILE_PATH)
            if TICKER_COLUMN_NAME not in df.columns:
                st.error(f"The column '{TICKER_COLUMN_NAME}' was not found in '{EXCEL_FILE_PATH}'.")
            else:
                tickers = df[TICKER_COLUMN_NAME].dropna().unique()
                scraped_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, ticker in enumerate(tickers):
                    status_text.text(f"Scraping {i+1}/{len(tickers)}: {ticker}")
                    data, status = scrape_screener_data(ticker)
                    if data:
                        scraped_results.append(data)
                    else:
                        scraped_results.append({'Ticker': ticker, 'Market Cap': status}) # Show error status
                    progress_bar.progress((i + 1) / len(tickers))
                
                status_text.success("Scraping complete!")
                
                if scraped_results:
                    results_df = pd.DataFrame(scraped_results).set_index('Ticker')
                    st.dataframe(results_df, use_container_width=True)
                    
                    # --- Add a download button ---
                    @st.cache_data
                    def convert_df_to_excel(df):
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=True, sheet_name='ScreenerData')
                        processed_data = output.getvalue()
                        return processed_data

                    excel_data = convert_df_to_excel(results_df)
                    st.download_button(
                        label="📥 Download Data as Excel",
                        data=excel_data,
                        file_name="screener_stock_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"An error occurred during extraction: {e}")

# --- TAB 2: INTRINSIC VALUE ANALYZER ---
with tab2:
    st.header("⚖️ Intrinsic Value & Technical Analyzer")
    with st.sidebar:
        st.header("Intrinsic Value Settings")
        bond_yield = st.number_input(
            "Current AAA Corporate Bond Yield (%)", 
            min_value=0.1, max_value=15.0, value=7.5, step=0.1,
            help="This is 'Y' in Graham's formula."
        )
    
    analyze_button = st.button("🚀 Run Intrinsic Value Analysis", disabled=(not file_exists))

    if analyze_button:
        try:
            df = pd.read_excel(EXCEL_FILE_PATH)
            if TICKER_COLUMN_NAME not in df.columns:
                st.error(f"The column '{TICKER_COLUMN_NAME}' was not found in '{EXCEL_FILE_PATH}'.")
            else:
                tickers = df[TICKER_COLUMN_NAME].dropna().unique()
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, ticker in enumerate(tickers):
                    full_ticker = f"{ticker}.NS"
                    status_text.text(f"Analyzing {i+1}/{len(tickers)}: {full_ticker}")
                    try:
                        stock = yf.Ticker(full_ticker)
                        info, financials, history = stock.info, stock.financials, stock.history(period="2y")
                        if history.empty or financials.empty: raise ValueError("No data.")
                        current_price = info.get('currentPrice', history['Close'].iloc[-1])
                        iv, iv_status = calculate_graham_intrinsic_value(info, financials, bond_yield)
                        tech, tech_status = get_technical_indicators(history)
                        results.append({'Ticker': ticker, 'Current Price': current_price, 'Intrinsic Value': iv, 'IV Status': iv_status, 'SMA_50': tech.get('SMA_50'), 'SMA_200': tech.get('SMA_200'), 'RSI_14': tech.get('RSI_14'), 'Tech Status': tech_status})
                    except Exception as e:
                        results.append({'Ticker': ticker, 'Current Price': None, 'IV Status': f"Error: {e}"})
                    progress_bar.progress((i + 1) / len(tickers))
                
                status_text.success("Analysis complete!")
                
                if results:
                    results_df = pd.DataFrame(results).set_index('Ticker')
                    results_df['Signal'] = results_df.apply(generate_signal, axis=1)
                    st.subheader("📊 Analysis Summary")
                    st.dataframe(results_df, use_container_width=True)
                    st.subheader("🔍 Detailed Stock View")
                    selected_stock = st.selectbox("Select a stock:", results_df.index)
                    if selected_stock:
                        stock_data = results_df.loc[selected_stock]
                        stock_history = yf.Ticker(f"{selected_stock}.NS").history(period="2y")
                        st.write(f"**Signal for {selected_stock}: {stock_data['Signal']}**")
                        # ... (rest of the detailed view code remains the same)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"₹{stock_data['Current Price']:.2f}")
                        col2.metric("Intrinsic Value", f"₹{stock_data['Intrinsic Value']:.2f}" if pd.notna(stock_data['Intrinsic Value']) else "N/A")
                        col3.metric("RSI (14)", f"{stock_data['RSI_14']:.1f}" if pd.notna(stock_data['RSI_14']) else "N/A")
                        fig = go.Figure(data=[go.Candlestick(x=stock_history.index, open=stock_history['Open'], high=stock_history['High'], low=stock_history['Low'], close=stock_history['Close'], name='Price')])
                        fig.add_trace(go.Scatter(x=stock_history.index, y=stock_history['Close'].rolling(window=50).mean(), mode='lines', name='50-Day SMA', line=dict(color='orange', width=1.5)))
                        fig.add_trace(go.Scatter(x=stock_history.index, y=stock_history['Close'].rolling(window=200).mean(), mode='lines', name='200-Day SMA', line=dict(color='purple', width=1.5)))
                        fig.update_layout(title=f'{selected_stock} Price Chart with Indicators', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

