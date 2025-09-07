import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import plotly.graph_objects as go
import os

# ======================================================================================
# CONFIGURATION & HEADER
# ======================================================================================
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Intrinsic Value & Technical Stock Analyzer")
st.markdown("""
This tool analyzes stocks from a local Excel file to determine their intrinsic value based on Benjamin Graham's formula 
and evaluates their technical strength using moving averages and RSI.
""")

# ======================================================================================
# CORE CALCULATION FUNCTIONS
# ======================================================================================

def calculate_graham_intrinsic_value(info, financials, bond_yield):
    """
    Calculates the intrinsic value of a stock using Benjamin Graham's formula.
    Formula: V = (EPS * (8.5 + 2g) * 4.4) / Y
    V = Intrinsic Value
    EPS = Trailing 12-month Earnings Per Share
    8.5 = P/E base for a no-growth company
    g = Estimated earnings growth rate (we'll estimate this over 5 years)
    4.4 = The average yield of high-grade corporate bonds in Graham's time
    Y = The current yield on AAA corporate bonds (user input)
    """
    try:
        # --- Get EPS ---
        eps = info.get('trailingEps')
        if not eps or eps <= 0:
            return None, "EPS is zero or negative."

        # --- Estimate Growth Rate (g) ---
        # We use Net Income growth over the last 4-5 years
        net_income = financials.loc['Net Income']
        if net_income.isnull().all() or len(net_income.dropna()) < 2:
            return None, "Not enough Net Income data for growth calculation."

        # Calculate year-over-year growth and get the average
        growth_rates = net_income.pct_change().dropna()
        # Cap growth rate at a reasonable level (e.g., 15%) to be conservative
        avg_growth_rate = np.mean(growth_rates)
        g = min(avg_growth_rate * 100, 15.0) 

        # --- Apply Graham's Formula ---
        # If growth is negative, we consider it zero for a more conservative valuation.
        if g < 0:
            g = 0

        intrinsic_value = (eps * (8.5 + 2 * g) * 4.4) / bond_yield
        return intrinsic_value, "Success"

    except (KeyError, IndexError, TypeError) as e:
        return None, f"Missing data for calculation: {e}"

def get_technical_indicators(history):
    """Calculates key technical indicators (50D SMA, 200D SMA, 14D RSI)."""
    try:
        if len(history) < 200:
            return {}, "Insufficient data (<200 days)."
            
        # --- SMAs ---
        sma_50 = SMAIndicator(close=history['Close'], window=50).sma_indicator().iloc[-1]
        sma_200 = SMAIndicator(close=history['Close'], window=200).sma_indicator().iloc[-1]

        # --- RSI ---
        rsi_14 = RSIIndicator(close=history['Close'], window=14).rsi().iloc[-1]
        
        return {
            'SMA_50': sma_50,
            'SMA_200': sma_200,
            'RSI_14': rsi_14
        }, "Success"
    except Exception as e:
        return {}, f"Error in technical calculation: {e}"

def generate_signal(row):
    """Generates a trading signal based on valuation and technicals."""
    iv = row['Intrinsic Value']
    price = row['Current Price']
    sma_50 = row['SMA_50']
    sma_200 = row['SMA_200']

    # Check for missing data
    if any(pd.isna(val) for val in [iv, price, sma_50, sma_200]):
        return "Not Available"

    is_undervalued = price < iv
    in_uptrend = price > sma_50 and price > sma_200

    if is_undervalued and in_uptrend:
        return "Strong Buy"
    if is_undervalued and price > sma_50:
        return "Buy"
    if is_undervalued and not in_uptrend:
        return "Monitor (Undervalued, but in Downtrend)"
    if not is_undervalued and in_uptrend:
        return "Hold (Overvalued, but in Uptrend)"
    if not is_undervalued and not in_uptrend:
        return "Sell"
    return "Hold"

# ======================================================================================
# STREAMLIT SIDEBAR & INPUTS
# ======================================================================================

with st.sidebar:
    st.header("⚙️ Analysis Configuration")
    
    # --- MODIFIED: Hardcoded local file path and ticker column name ---
    EXCEL_FILE_PATH = "nse_tickers.xlsx"
    TICKER_COLUMN_NAME = "Ticker"

    st.info(f"Reading tickers from local file:\n`{EXCEL_FILE_PATH}`")
    st.write(f"Expecting a column named: `{TICKER_COLUMN_NAME}`")

    bond_yield = st.number_input(
        "Current AAA Corporate Bond Yield (%)", 
        min_value=0.1, max_value=15.0, value=7.5, step=0.1,
        help="This is 'Y' in Graham's formula. A higher yield leads to a more conservative (lower) intrinsic value."
    )
    
    # --- MODIFIED: Button is disabled if the local file does not exist ---
    file_exists = os.path.exists(EXCEL_FILE_PATH)
    if not file_exists:
        st.error(f"Error: The file '{EXCEL_FILE_PATH}' was not found in the app's root directory.")
    
    analyze_button = st.button("🚀 Run Analysis", disabled=(not file_exists))

# ======================================================================================
# MAIN APP LOGIC & DISPLAY
# ======================================================================================

if analyze_button:
    try:
        # --- MODIFIED: Reads from the local Excel file path ---
        df = pd.read_excel(EXCEL_FILE_PATH)
        if TICKER_COLUMN_NAME not in df.columns:
            st.error(f"The column '{TICKER_COLUMN_NAME}' was not found in '{EXCEL_FILE_PATH}'. Please check the file.")
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
                    
                    # --- Fetch Data ---
                    info = stock.info
                    financials = stock.financials
                    history = stock.history(period="2y") # 2 years of data for indicators

                    if history.empty or financials.empty:
                        raise ValueError("No historical or financial data found.")

                    current_price = info.get('currentPrice', history['Close'].iloc[-1])
                    
                    # --- Calculations ---
                    iv, iv_status = calculate_graham_intrinsic_value(info, financials, bond_yield)
                    tech, tech_status = get_technical_indicators(history)

                    results.append({
                        'Ticker': ticker,
                        'Current Price': current_price,
                        'Intrinsic Value': iv,
                        'IV Status': iv_status,
                        'SMA_50': tech.get('SMA_50'),
                        'SMA_200': tech.get('SMA_200'),
                        'RSI_14': tech.get('RSI_14'),
                        'Tech Status': tech_status
                    })

                except Exception as e:
                    results.append({'Ticker': ticker, 'Current Price': None, 'IV Status': f"Error: {e}"})

                progress_bar.progress((i + 1) / len(tickers))

            status_text.text("Analysis complete!")
            
            # --- Display Results Table ---
            if results:
                results_df = pd.DataFrame(results).set_index('Ticker')
                results_df['Signal'] = results_df.apply(generate_signal, axis=1)
                
                # --- Styling for the results table ---
                def style_signal(val):
                    color = 'gray'
                    if 'Buy' in val: color = 'green'
                    elif 'Sell' in val: color = 'red'
                    elif 'Monitor' in val: color = 'orange'
                    return f'color: {color}; font-weight: bold;'

                st.subheader("📊 Analysis Summary")
                st.dataframe(results_df.style.applymap(style_signal, subset=['Signal'])
                                           .format({
                                               'Current Price': '₹{:.2f}',
                                               'Intrinsic Value': '₹{:.2f}',
                                               'SMA_50': '{:.2f}',
                                               'SMA_200': '{:.2f}',
                                               'RSI_14': '{:.1f}'
                                           }), use_container_width=True)
                
                # --- Detailed View Section ---
                st.subheader("🔍 Detailed Stock View")
                selected_stock = st.selectbox("Select a stock to view details:", results_df.index)
                
                if selected_stock:
                    stock_data = results_df.loc[selected_stock]
                    stock_history = yf.Ticker(f"{selected_stock}.NS").history(period="2y")
                    
                    # Display metrics
                    st.write(f"**Signal for {selected_stock}: {stock_data['Signal']}**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"₹{stock_data['Current Price']:.2f}")
                    col2.metric("Intrinsic Value", f"₹{stock_data['Intrinsic Value']:.2f}" if pd.notna(stock_data['Intrinsic Value']) else "N/A")
                    col3.metric("RSI (14)", f"{stock_data['RSI_14']:.1f}" if pd.notna(stock_data['RSI_14']) else "N/A")

                    # Create Plotly chart
                    fig = go.Figure(data=[go.Candlestick(x=stock_history.index,
                                    open=stock_history['Open'],
                                    high=stock_history['High'],
                                    low=stock_history['Low'],
                                    close=stock_history['Close'], name='Price')])

                    fig.add_trace(go.Scatter(x=stock_history.index, y=stock_history['Close'].rolling(window=50).mean(), mode='lines', name='50-Day SMA', line=dict(color='orange', width=1.5)))
                    fig.add_trace(go.Scatter(x=stock_history.index, y=stock_history['Close'].rolling(window=200).mean(), mode='lines', name='200-Day SMA', line=dict(color='purple', width=1.5)))
                    fig.update_layout(title=f'{selected_stock} Price Chart with Indicators',
                                      yaxis_title='Price (INR)',
                                      xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

else:
    st.info("Ensure `nse_tickers.xlsx` is in the same directory and click 'Run Analysis' to begin.")

