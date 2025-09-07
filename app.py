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

st.title("📈 Integrated Stock Analyzer")
st.markdown("Select an industry from your Excel file to get a consolidated analysis, including financial ratios from **Screener.in** and a detailed **Swing Trading** recommendation.")

# ======================================================================================
# DATA FETCHING & CALCULATION FUNCTIONS
# ======================================================================================

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker):
    """Fetches all necessary data for a stock from yfinance."""
    stock = yf.Ticker(f"{ticker}.NS")
    history = stock.history(period="3y", interval="1wk")
    info = stock.info
    financials = stock.financials
    return history, info, financials

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

# --- NEW: SWING TRADING ALGORITHM FUNCTION ---
def calculate_swing_trade_analysis(history):
    """
    Calculates a comprehensive set of swing trading indicators and generates a recommendation.
    """
    if len(history) < 52: # Need at least a year of weekly data
        return None, "Insufficient weekly data for full analysis."

    # --- 1. Calculate all indicators ---
    close = history['Close']
    # Moving Averages
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    # RSI
    rsi_14 = RSIIndicator(close, window=14).rsi().iloc[-1]
    # MACD
    macd_indicator = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    macd_hist = macd_indicator.macd_diff().iloc[-1]
    # Bollinger Bands
    bb_indicator = BollingerBands(close, window=20, window_dev=2)
    bb_high = bb_indicator.bollinger_hband().iloc[-1]
    bb_low = bb_indicator.bollinger_lband().iloc[-1]
    # Volume
    obv_indicator = OnBalanceVolumeIndicator(close, history['Volume'])
    obv_slope = obv_indicator.on_balance_volume().diff().rolling(window=5).mean().iloc[-1]
    # Volatility
    atr_14 = AverageTrueRange(history['High'], history['Low'], close, window=14).average_true_range().iloc[-1]

    indicators = {
        "20-Week SMA": sma_20, "50-Week SMA": sma_50, "RSI (14)": rsi_14,
        "MACD Line": macd_line, "MACD Signal": macd_signal, "MACD Histogram": macd_hist,
        "Bollinger High": bb_high, "Bollinger Low": bb_low, "OBV Trend": obv_slope, "ATR (14)": atr_14
    }

    # --- 2. Algorithm for Buy/Sell Recommendation ---
    score = 0
    price = close.iloc[-1]
    
    # Trend Signals (Max 5 points)
    if price > sma_20: score += 2
    if sma_20 > sma_50: score += 2
    if macd_line > macd_signal: score += 1
    
    # Momentum Signals (Max 3 points)
    if 45 < rsi_14 < 68: score += 2 # Strong, but not overbought
    if macd_hist > 0: score += 1

    # Volume Confirmation (Max 2 points)
    if obv_slope > 0: score += 2

    # Entry Point Signal (Bonus for pullbacks)
    if price < sma_20 and price > sma_50 and rsi_14 < 55: score += 1 # Dip buy opportunity
    if price <= bb_low: score += 1 # Near support

    # --- 3. Generate Final Recommendation ---
    if score >= 7:
        recommendation = "Strong Buy"
    elif score >= 5:
        recommendation = "Buy"
    elif score >= 3:
        recommendation = "Hold / Monitor"
    else:
        recommendation = "Sell / Avoid"
        
    return indicators, recommendation

# ======================================================================================
# STREAMLIT UI & LOGIC
# ======================================================================================
EXCEL_FILE_PATH = "SELECTED STOCKS 22FEB2025.xlsx"
TICKER_COLUMN_NAME = "NSE SYMBOL"
INDUSTRY_COLUMN_NAME = "INDUSTRY"

if not os.path.exists(EXCEL_FILE_PATH):
    st.error(f"Error: The file '{EXCEL_FILE_PATH}' was not found.")
else:
    with st.sidebar:
        st.header("⚙️ Analysis Filter")
        try:
            df_full = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
            industries = ["All Industries"] + sorted(df_full[INDUSTRY_COLUMN_NAME].dropna().unique().tolist())
            selected_industry = st.selectbox("Select an Industry:", industries)
            analyze_button = st.button("🚀 Analyze Selected Industry", type="primary")
        except Exception as e:
            st.error(f"Could not read the Excel file. Error: {e}")
            selected_industry = None; analyze_button = False

    if analyze_button and selected_industry:
        df_filtered = df_full[df_full[INDUSTRY_COLUMN_NAME] == selected_industry] if selected_industry != "All Industries" else df_full
        tickers = df_filtered[TICKER_COLUMN_NAME].dropna().unique()
        st.header(f"Analysis for: {selected_industry}")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(tickers):
            status_text.text(f"Processing {i+1}/{len(tickers)}: {ticker}")
            try:
                history, info, financials = get_stock_data(ticker)
                if history.empty: continue

                with st.expander(f"▶️ **{ticker}** | Current Price: ₹{history['Close'].iloc[-1]:.2f}", expanded=(i==0)):
                    col1, col2 = st.columns([1, 2])
                    
                    # --- Column 1: Ratios, Valuation, and Swing Analysis ---
                    with col1:
                        # --- Swing Trade Analysis ---
                        st.subheader("Swing Trade Analysis (Weekly)")
                        swing_indicators, swing_recommendation = calculate_swing_trade_analysis(history)
                        
                        if swing_indicators:
                            st.metric("Recommendation", swing_recommendation)
                            st.dataframe(pd.DataFrame(swing_indicators.items(), columns=['Indicator', 'Value']).set_index('Indicator'))
                        else:
                            st.warning("Not enough data for swing analysis.")
                        
                        # --- Valuation ---
                        st.subheader("Valuation")
                        intrinsic_value = calculate_graham_intrinsic_value(info, financials)
                        st.metric("Intrinsic Value (Graham)", f"₹{intrinsic_value:.2f}" if intrinsic_value else "N/A")
                        st.metric("Recommended Swing Buy Price (≈20W SMA)", f"₹{swing_indicators['20-Week SMA']:.2f}" if swing_indicators else "N/A")

                        # --- Screener Ratios ---
                        st.subheader("Financial Ratios (from Screener.in)")
                        screener_data, screener_status = scrape_screener_data(ticker)
                        if screener_status == "Success":
                            st.table(pd.DataFrame(screener_data.items(), columns=['Ratio', 'Value']))
                        else:
                            st.warning(f"Could not scrape data ({screener_status}).")

                    # --- Column 2: Chart ---
                    with col2:
                        st.subheader("Weekly Price Chart")
                        fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='Price')])
                        history['SMA_20W'] = history['Close'].rolling(window=20).mean()
                        history['SMA_50W'] = history['Close'].rolling(window=50).mean()
                        fig.add_trace(go.Scatter(x=history.index, y=history['SMA_20W'], mode='lines', name='20-Week SMA', line=dict(color='orange', width=1.5)))
                        fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50W'], mode='lines', name='50-Week SMA', line=dict(color='purple', width=1.5)))
                        fig.update_layout(yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while processing **{ticker}**: {e}")
            progress_bar.progress((i + 1) / len(tickers))
        status_text.success("Analysis Complete!")
    else:
        st.info("Select an industry from the sidebar and click the 'Analyze' button to begin.")

