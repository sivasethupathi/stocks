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

st.title("Stock Analyzer by SIVASETHUPATHI")
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

# --- UPDATED: SWING TRADING ALGORITHM NOW PROVIDES REASONING ---
def calculate_swing_trade_analysis(history):
    """
    Calculates swing trading indicators and generates a recommendation with reasoning.
    """
    if len(history) < 52:
        return None, "Insufficient Data", "Not enough weekly data for full analysis."

    close = history['Close']
    price = close.iloc[-1]
    
    # --- 1. Calculate indicators ---
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

    indicators = {
        "20-Week SMA": sma_20, "50-Week SMA": sma_50, "RSI (14)": rsi_14,
        "MACD Line": macd_line, "MACD Signal": macd_signal, "MACD Histogram": macd_hist,
        "Bollinger High": bb_high, "Bollinger Low": bb_low, "OBV Trend": obv_slope, "ATR (14)": atr_14
    }

    # --- 2. Scoring Algorithm with Reasoning ---
    score = 0
    reasons = []
    
    # Trend Signals
    if price > sma_20: score += 2; reasons.append("✅ Price is above the 20-week SMA (Strong short-term trend).")
    else: reasons.append("❌ Price is below the 20-week SMA (Bearish short-term trend).")
    if sma_20 > sma_50: score += 2; reasons.append("✅ 20-week SMA is above the 50-week SMA (Golden Cross).")
    else: reasons.append("❌ 20-week SMA is below the 50-week SMA (Death Cross).")
    if macd_line > macd_signal: score += 1; reasons.append("✅ MACD line is above the signal line (Bullish momentum).")
    else: reasons.append("❌ MACD line is below the signal line (Bearish momentum).")
    
    # Momentum Signals
    if 45 < rsi_14 < 68: score += 2; reasons.append(f"✅ RSI is healthy at {rsi_14:.1f} (Not overbought/oversold).")
    elif rsi_14 >= 68: reasons.append(f"⚠️ RSI is high at {rsi_14:.1f} (Approaching overbought).")
    else: reasons.append(f"❌ RSI is weak at {rsi_14:.1f} (Bearish momentum).")
    if macd_hist > 0: score += 1; reasons.append("✅ MACD histogram is positive (Increasing bullish momentum).")
    
    # Volume Confirmation
    if obv_slope > 0: score += 2; reasons.append("✅ On-Balance Volume trend is positive (Volume confirms price trend).")
    else: reasons.append("❌ On-Balance Volume trend is negative (Volume does not confirm price trend).")

    # Entry Point Signal
    if price <= bb_low: score += 1; reasons.append("💡 Price is near the lower Bollinger Band (Potential bounce/support).")
    
    # --- 3. Generate Final Recommendation ---
    if score >= 7: recommendation = "Strong Buy"
    elif score >= 5: recommendation = "Buy"
    elif score >= 3: recommendation = "Hold / Monitor"
    else: recommendation = "Sell / Avoid"
        
    return indicators, recommendation, "\n\n".join(reasons)

# --- Function to display analysis for a single stock ---
def display_stock_analysis(ticker):
    try:
        history, info, financials = get_stock_data(ticker)
        if history.empty:
            st.warning(f"Could not fetch price history for **{ticker}**. Skipping.")
            return

        st.header(f"Analysis for: {ticker}", divider='rainbow')

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Swing Trade Analysis (Weekly)")
            swing_indicators, swing_recommendation, swing_reasoning = calculate_swing_trade_analysis(history)
            
            if swing_indicators:
                st.metric("Recommendation", swing_recommendation)
                st.info(swing_reasoning)
                st.dataframe(pd.DataFrame(swing_indicators.items(), columns=['Indicator', 'Value']).set_index('Indicator'))
            else:
                st.warning("Not enough data for swing analysis.")
            
            st.subheader("Valuation")
            intrinsic_value = calculate_graham_intrinsic_value(info, financials)
            st.metric("Intrinsic Value (Graham)", f"₹{intrinsic_value:.2f}" if intrinsic_value else "N/A")
            st.metric("Recommended Swing Buy Price (≈20W SMA)", f"₹{swing_indicators['20-Week SMA']:.2f}" if swing_indicators else "N/A")

            st.subheader("Financial Ratios (from Screener.in)")
            screener_data, screener_status = scrape_screener_data(ticker)
            if screener_status == "Success":
                st.table(pd.DataFrame(screener_data.items(), columns=['Ratio', 'Value']))
            else:
                st.warning(f"Could not scrape data ({screener_status}).")

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

# ======================================================================================
# STREAMLIT UI & LOGIC
# ======================================================================================
EXCEL_FILE_PATH = "SELECTED STOCKS 22FEB2025.xlsx"
TICKER_COLUMN_NAME = "NSE SYMBOL"
INDUSTRY_COLUMN_NAME = "INDUSTRY"

# --- Initialize session state for navigation ---
if 'current_stock_index' not in st.session_state:
    st.session_state.current_stock_index = 0
if 'ticker_list' not in st.session_state:
    st.session_state.ticker_list = []

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

        except Exception as e:
            st.error(f"Could not read the Excel file. Error: {e}")

    # --- Main Display Area ---
    if st.session_state.ticker_list:
        current_ticker = st.session_state.ticker_list[st.session_state.current_stock_index]
        
        # --- Navigation Buttons ---
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            if st.button("⬅️ Previous Stock", use_container_width=True, disabled=(st.session_state.current_stock_index == 0)):
                st.session_state.current_stock_index -= 1
                st.rerun()
        with col2:
            st.write(f"Displaying **{st.session_state.current_stock_index + 1}** of **{len(st.session_state.ticker_list)}** stocks")
        with col3:
            if st.button("Next Stock ➡️", use_container_width=True, disabled=(st.session_state.current_stock_index >= len(st.session_state.ticker_list) - 1)):
                st.session_state.current_stock_index += 1
                st.rerun()

        # Display the analysis for the current stock
        display_stock_analysis(current_ticker)
    else:
        st.info("Select an industry from the sidebar and click the 'Analyze' button to begin.")

