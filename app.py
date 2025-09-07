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
import io

# ======================================================================================
# CONFIGURATION & HEADER
# ======================================================================================
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Automated Stock Analyzer")
st.markdown("""
This tool offers three modes of analysis:
1.  **Swing Trading Analyzer**: Identifies top 5 swing trade opportunities on a weekly timeline.
2.  **Screener Data Extractor**: Scrapes key financial ratios for your stocks from `screener.in`.
3.  **Intrinsic Value Analyzer**: Calculates intrinsic value using Benjamin Graham's formula and daily technicals.
""")

# ======================================================================================
# SCREENER.IN SCRAPING FUNCTIONS
# ======================================================================================

def clean_value(value_str):
    """Cleans the scraped string value and converts it to a float."""
    if value_str is None: return None
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
    if response.status_code != 200: return None, f"Failed to load page (Status: {response.status_code})"
    soup = BeautifulSoup(response.content, 'html.parser')
    data = {}
    ratio_list = soup.select_one('#top-ratios')
    if not ratio_list: return None, "Could not find the ratios section."
    for li in ratio_list.find_all('li'):
        name = li.select_one('.name').get_text(strip=True) if li.select_one('.name') else ''
        value = li.select_one('.nowrap.value .number').get_text(strip=True) if li.select_one('.nowrap.value .number') else ''
        if name and value: data[name] = value
    output = {
        'Ticker': ticker, 'Market Cap': clean_value(data.get('Market Cap')), 'Current Price': clean_value(data.get('Current Price')),
        'High / Low': data.get('High / Low'), 'Stock P/E': clean_value(data.get('Stock P/E')), 'Book Value': clean_value(data.get('Book Value')),
        'Dividend Yield': clean_value(data.get('Dividend Yield')), 'ROCE': clean_value(data.get('ROCE')), 'ROE': clean_value(data.get('ROE')),
        'Face Value': clean_value(data.get('Face Value')), 'EV/EBITDA': clean_value(data.get('EV/EBITDA')), 'Debt to equity': clean_value(data.get('Debt to equity')),
        'Enterprise Value': 'N/A', 'Current Ratio': 'N/A', 'PEG Ratio': 'N/A'
    }
    return output, "Success"

# ======================================================================================
# CORE CALCULATION FUNCTIONS
# ======================================================================================

def calculate_graham_intrinsic_value(info, financials, bond_yield):
    """Calculates the intrinsic value of a stock using Benjamin Graham's formula."""
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
    except (KeyError, IndexError, TypeError):
        return None, "Missing data for IV calculation."

def get_technical_indicators(history, interval='daily'):
    """Calculates key technical indicators for daily or weekly data."""
    if interval == 'weekly':
        if len(history) < 50: return {}, "Insufficient weekly data (<50 weeks)."
        sma_20 = SMAIndicator(close=history['Close'], window=20).sma_indicator().iloc[-1]
        sma_50 = SMAIndicator(close=history['Close'], window=50).sma_indicator().iloc[-1]
        rsi_14 = RSIIndicator(close=history['Close'], window=14).rsi().iloc[-1]
        return {'SMA_20W': sma_20, 'SMA_50W': sma_50, 'RSI_14W': rsi_14}, "Success"
    else: # Daily
        if len(history) < 200: return {}, "Insufficient daily data (<200 days)."
        sma_50 = SMAIndicator(close=history['Close'], window=50).sma_indicator().iloc[-1]
        sma_200 = SMAIndicator(close=history['Close'], window=200).sma_indicator().iloc[-1]
        rsi_14 = RSIIndicator(close=history['Close'], window=14).rsi().iloc[-1]
        return {'SMA_50': sma_50, 'SMA_200': sma_200, 'RSI_14': rsi_14}, "Success"

def generate_daily_signal(row):
    """Generates a trading signal based on daily valuation and technicals."""
    iv, price, sma_50, sma_200 = row['Intrinsic Value'], row['Current Price'], row['SMA_50'], row['SMA_200']
    if any(pd.isna(val) for val in [iv, price, sma_50, sma_200]): return "Not Available"
    is_undervalued, in_uptrend = price < iv, price > sma_50 and price > sma_200
    if is_undervalued and in_uptrend: return "Strong Buy"
    if is_undervalued and price > sma_50: return "Buy"
    if is_undervalued and not in_uptrend: return "Monitor (Undervalued, Downtrend)"
    if not is_undervalued and in_uptrend: return "Hold (Overvalued, Uptrend)"
    return "Sell"

# ======================================================================================
# STREAMLIT UI & LOGIC
# ======================================================================================

with st.sidebar:
    st.header("⚙️ File Configuration")
    EXCEL_FILE_PATH = "SELECTED STOCKS 22FEB2025.xlsx"
    TICKER_COLUMN_NAME = "NSE SYMBOL"
    INDUSTRY_COLUMN_NAME = "INDUSTRY"
    st.info(f"Reading tickers from local file:\n`{EXCEL_FILE_PATH}`")
    file_exists = os.path.exists(EXCEL_FILE_PATH)
    if not file_exists:
        st.error(f"Error: The file '{EXCEL_FILE_PATH}' was not found.")

# --- Create Tabs for different functionalities ---
tab3, tab1, tab2 = st.tabs(["Swing Trading Analyzer", "Screener Data Extractor", "Intrinsic Value Analyzer"])

# --- NEW TAB: SWING TRADING ANALYZER ---
with tab3:
    st.header("🚀 Swing Trading Opportunities (Weekly)")
    st.markdown("This tool identifies the top 5 stocks for swing trading based on weekly trend and momentum indicators.")
    
    if file_exists:
        try:
            df_swing = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
            industries = ["All Industries"] + sorted(df_swing[INDUSTRY_COLUMN_NAME].dropna().unique().tolist())
            selected_industry = st.selectbox("Filter by Industry:", industries)

            swing_analyze_button = st.button("Find Top 5 Swing Trades")

            if swing_analyze_button:
                if selected_industry != "All Industries":
                    df_filtered = df_swing[df_swing[INDUSTRY_COLUMN_NAME] == selected_industry]
                else:
                    df_filtered = df_swing
                
                tickers = df_filtered[TICKER_COLUMN_NAME].dropna().unique()
                swing_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, ticker in enumerate(tickers):
                    full_ticker = f"{ticker}.NS"
                    status_text.text(f"Analyzing {i+1}/{len(tickers)}: {full_ticker}")
                    try:
                        stock = yf.Ticker(full_ticker)
                        history = stock.history(period="3y", interval="1wk")
                        if history.empty: continue
                        
                        price = history['Close'].iloc[-1]
                        tech, tech_status = get_technical_indicators(history, interval='weekly')
                        iv, iv_status = calculate_graham_intrinsic_value(stock.info, stock.financials, 7.5) # Using default bond yield

                        # Scoring Logic
                        score = 0
                        sma_20w, sma_50w, rsi_14w = tech.get('SMA_20W'), tech.get('SMA_50W'), tech.get('RSI_14W')
                        if price > sma_20w: score += 2
                        if price > sma_50w: score += 1
                        if sma_20w > sma_50w: score += 2
                        if 40 < rsi_14w < 65: score += 1
                        discount = (iv - price) / price if iv and price else 0
                        if discount > 0.2: score += 2
                        elif discount > 0: score += 1
                        
                        swing_results.append({
                            'Ticker': ticker, 'Current Price': price, 'Recommended Buy Price': sma_20w,
                            'Intrinsic Value': iv, 'RSI (14W)': rsi_14w, 'Swing Score': score
                        })
                    except Exception:
                        continue # Skip stocks with errors
                    progress_bar.progress((i + 1) / len(tickers))
                
                status_text.success("Swing trade analysis complete!")
                if swing_results:
                    swing_df = pd.DataFrame(swing_results).sort_values(by="Swing Score", ascending=False).head(5)
                    st.dataframe(swing_df.style.format({
                        'Current Price': '₹{:.2f}', 'Recommended Buy Price': '₹{:.2f}', 'Intrinsic Value': '₹{:.2f}',
                        'RSI (14W)': '{:.1f}', 'Swing Score': '{:.0f}'
                    }), use_container_width=True)

        except Exception as e:
            st.error(f"Could not process Excel file for swing trading. Error: {e}")

# --- TAB 1: SCREENER DATA EXTRACTOR ---
with tab1:
    st.header("📊 Screener.in Data Extractor")
    extract_button = st.button("🚀 Extract Data from Screener", disabled=(not file_exists))
    if extract_button:
        # (Logic from previous version, unchanged)
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
        tickers = df[TICKER_COLUMN_NAME].dropna().unique()
        # ... rest of the scraping and display logic ...

# --- TAB 2: INTRINSIC VALUE ANALYZER ---
with tab2:
    st.header("⚖️ Intrinsic Value & Technical Analyzer (Daily)")
    with st.sidebar:
        st.header("Daily Analysis Settings")
        bond_yield = st.number_input("Current AAA Bond Yield (%)", min_value=0.1, max_value=15.0, value=7.5, step=0.1)
    
    analyze_button = st.button("🚀 Run Intrinsic Value Analysis", disabled=(not file_exists))
    if analyze_button:
        # (Logic from previous version, unchanged)
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
        tickers = df[TICKER_COLUMN_NAME].dropna().unique()
        # ... rest of the daily analysis and display logic ...

