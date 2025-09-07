import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import os
import requests
from bs4 import BeautifulSoup
import re

# ======================================================================================
# CONFIGURATION & HEADER
# ======================================================================================
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Integrated Stock Analyzer")
st.markdown("Select an industry from your Excel file to get a consolidated analysis, including financial ratios from **Screener.in** and weekly trading charts.")

# ======================================================================================
# DATA FETCHING & CALCULATION FUNCTIONS
# ======================================================================================

@st.cache_data(ttl=3600) # Cache data for 1 hour to reduce redundant calls
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
        name_span = li.select_one('.name')
        value_span = li.select_one('.nowrap.value .number')
        if name_span and value_span:
            name = name_span.get_text(strip=True)
            value = value_span.get_text(strip=True)
            data[name] = value

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

# ======================================================================================
# STREAMLIT UI & LOGIC
# ======================================================================================

# --- File Configuration ---
EXCEL_FILE_PATH = "SELECTED STOCKS 22FEB2025.xlsx"
TICKER_COLUMN_NAME = "NSE SYMBOL"
INDUSTRY_COLUMN_NAME = "INDUSTRY"

if not os.path.exists(EXCEL_FILE_PATH):
    st.error(f"Error: The file '{EXCEL_FILE_PATH}' was not found. Please place it in the same directory as the app.")
else:
    # --- Sidebar for Industry Selection ---
    with st.sidebar:
        st.header("⚙️ Analysis Filter")
        try:
            df_full = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
            industries = ["All Industries"] + sorted(df_full[INDUSTRY_COLUMN_NAME].dropna().unique().tolist())
            selected_industry = st.selectbox("Select an Industry:", industries)
            
            analyze_button = st.button("🚀 Analyze Selected Industry", type="primary")

        except Exception as e:
            st.error(f"Could not read the Excel file. Make sure the sheet is named 'Sheet1' and columns are correct. Error: {e}")
            selected_industry = None
            analyze_button = False

    if analyze_button and selected_industry:
        # Filter stocks based on selected industry
        if selected_industry != "All Industries":
            df_filtered = df_full[df_full[INDUSTRY_COLUMN_NAME] == selected_industry]
        else:
            df_filtered = df_full
        
        tickers = df_filtered[TICKER_COLUMN_NAME].dropna().unique()
        
        st.header(f"Analysis for: {selected_industry}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(tickers):
            status_text.text(f"Processing {i+1}/{len(tickers)}: {ticker}")
            
            try:
                # --- Fetch all data for the ticker ---
                history, info, financials = get_stock_data(ticker)
                screener_data, screener_status = scrape_screener_data(ticker)
                
                if history.empty:
                    st.warning(f"Could not fetch price history for **{ticker}**. Skipping.")
                    continue

                # --- Calculations ---
                intrinsic_value = calculate_graham_intrinsic_value(info, financials)
                
                # Weekly technicals for swing trading
                history['SMA_20W'] = history['Close'].rolling(window=20).mean()
                history['SMA_50W'] = history['Close'].rolling(window=50).mean()
                recommended_buy_price = history['SMA_20W'].iloc[-1]
                
                # --- Display Stock Information in an Expander ---
                with st.expander(f"▶️ **{ticker}** | Current Price: ₹{history['Close'].iloc[-1]:.2f}", expanded=(i==0)):
                    col1, col2 = st.columns([1, 2])
                    
                    # --- Column 1: Ratios and Key Values ---
                    with col1:
                        st.subheader("Financial Ratios")
                        if screener_status == "Success":
                            st.table(pd.DataFrame(screener_data.items(), columns=['Ratio', 'Value']))
                        else:
                            st.warning(f"Could not scrape data from Screener.in ({screener_status}).")

                        st.subheader("Valuation")
                        st.metric("Intrinsic Value (Graham)", f"₹{intrinsic_value:.2f}" if intrinsic_value else "N/A")
                        st.metric("Recommended Swing Buy Price (≈20W SMA)", f"₹{recommended_buy_price:.2f}" if recommended_buy_price else "N/A")

                    # --- Column 2: Chart ---
                    with col2:
                        st.subheader("Weekly Price Chart")
                        fig = go.Figure()
                        # Candlestick chart
                        fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'],
                                                     low=history['Low'], close=history['Close'], name='Price'))
                        # Moving Averages
                        fig.add_trace(go.Scatter(x=history.index, y=history['SMA_20W'], mode='lines', 
                                                 name='20-Week SMA', line=dict(color='orange', width=1.5)))
                        fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50W'], mode='lines', 
                                                 name='50-Week SMA', line=dict(color='purple', width=1.5)))
                        
                        fig.update_layout(
                            yaxis_title='Price (INR)',
                            xaxis_rangeslider_visible=False,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while processing **{ticker}**: {e}")

            progress_bar.progress((i + 1) / len(tickers))
        
        status_text.success("Analysis Complete!")

    else:
        st.info("Select an industry from the sidebar and click the 'Analyze' button to begin.")

