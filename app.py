import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import os
import requests
from bs4 import BeautifulSoup
import re
import json
import time 
from typing import Dict, Any, List 
from io import BytesIO 
from docx import Document 
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches # Added for document layout

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# ======================================================================================
# CONFIGURATION & HEADER
# ======================================================================================
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Stock Analyzer | NS   T R A D E R")
st.markdown("Select an industry from your Excel file to get a consolidated analysis, including financial ratios from **Screener.in** and a detailed **Swing Trading** recommendation with Fibonacci levels.")

# --- Web Scraping Configuration ---
NSE_BASE_URL = "https://www.nseindia.com/" # Kept for consistency, not used in export now.

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/555.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/555.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
}
# --- End Web Scraping Configuration ---

# ======================================================================================
# DATA FETCHING & CALCULATION FUNCTIONS (Dashboard UI)
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
    """Scrapes key financial data for a given ticker from screener.in (used for UI ratios)."""
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
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

# --- FIBONACCI RETRACEMENT ANALYSIS ---
def calculate_fibonacci_levels(history):
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

# ======================================================================================
# WORD DOCUMENT GENERATION MODULE: NEW SCREENER-BASED CRAWLER
# ======================================================================================

def scrape_all_screener_data(ticker: str) -> Dict[str, Any]:
    """
    Scrapes all structured data tables from the Screener page for a given ticker.
    Returns a dictionary where keys are section titles and values are DataFrames or dicts.
    """
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    scraped_data = {}
    time.sleep(1) # Be polite when scraping
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- 1. Top Ratios/Key Metrics ---
        # Note: This list also contains company name and basic industry
        key_metrics = {}
        ratio_list = soup.select_one('#top-ratios')
        if ratio_list:
            for li in ratio_list.find_all('li'):
                name_tag = li.select_one('.name')
                value_tag = li.select_one('.nowrap.value .number')
                
                if name_tag and value_tag:
                    name = name_tag.get_text(strip=True)
                    value = value_tag.get_text(strip=True)
                    key_metrics[name] = value
        
        # Get Company Name and Industry
        company_name = soup.select_one('.heading h1').get_text(strip=True) if soup.select_one('.heading h1') else ticker
        industry = soup.select_one('.company-industry a').get_text(strip=True) if soup.select_one('.company-industry a') else 'N/A'
        
        scraped_data['0. Company Summary'] = {
            'Name': company_name,
            'Ticker': ticker,
            'Industry': industry,
            'Latest Metrics': pd.DataFrame(key_metrics.items(), columns=['Metric', 'Value']).set_index('Metric').T
        }
        
        # --- 2. HTML Tables (Quarterly, P&L, BS, CF, Shareholding) ---
        table_sections = [
            ('1. Quarterly Results', '#quarterly', 'Quarterly'),
            ('2. Profit & Loss', '#profit-loss', 'Annual'),
            ('3. Balance Sheet', '#balance-sheet', 'Annual'),
            ('4. Cash Flow', '#cash-flow', 'Annual'),
            ('5. Shareholding Pattern', '#shareholding', 'Latest')
        ]
        
        for title, selector, transpose_mode in table_sections:
            section_tag = soup.select_one(selector)
            if section_tag:
                try:
                    df_list = pd.read_html(str(section_tag))
                    if df_list:
                        df = df_list[0].fillna('')
                        # Transpose logic for P&L, BS, CF, and Shareholding (Rows as metrics, Columns as dates)
                        if transpose_mode in ['Annual', 'Quarterly']:
                            df = df.set_index(df.columns[0])
                            df.index.name = df.columns.name
                        
                        # Fix multi-index columns for Shareholding if present
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
                        
                        scraped_data[title] = df
                except Exception as e:
                    scraped_data[title] = pd.DataFrame([{'Error': f'Failed to parse table: {e}'}])
        
    except requests.RequestException as e:
        scraped_data['Error'] = f"Failed to access Screener.in: {e}"
    except Exception as e:
        scraped_data['Error'] = f"An unexpected error occurred during scraping: {e}"
        
    return scraped_data

def add_dataframe_to_word(document, df: pd.DataFrame, table_style: str = 'Table Grid'):
    """Helper function to convert a Pandas DataFrame to a Word table."""
    document.add_paragraph()
    has_index = df.index.name is not None or not pd.RangeIndex(start=0, stop=len(df.index)).equals(df.index)

    # Ensure index column is present if we are treating the index as part of the data
    rows, cols = df.shape
    num_cols = cols + (1 if has_index else 0)
    
    # Calculate the number of header rows
    num_header_rows = 1

    table = document.add_table(rows + num_header_rows, num_cols)
    table.style = table_style
    
    # --- Write Header Row ---
    hdr_cells = table.rows[0].cells
    current_col = 0
    
    if has_index:
        hdr_cells[0].text = str(df.index.name or 'Metric')
        current_col = 1
        
    for j, col_name in enumerate(df.columns):
        hdr_cells[current_col + j].text = str(col_name)
        hdr_cells[current_col + j].paragraphs[0].runs[0].font.bold = True
        hdr_cells[current_col + j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # --- Write Data Rows ---
    for i, (index_name, row) in enumerate(df.iterrows()):
        row_cells = table.rows[i + num_header_rows].cells
        current_col = 0
        
        if has_index:
            row_cells[0].text = str(index_name)
            current_col = 1
            
        for j, value in enumerate(row):
            text_value = str(value).replace('\u2010', '-') # Replace hyphen-like character
            
            # Formatting (e.g., thousands separator)
            try:
                # Attempt to format only numeric values that are not percentages (which contain %)
                if text_value.replace(',', '').replace('.', '', 1).isdigit() and '%' not in text_value:
                    float_value = float(text_value.replace(',', ''))
                    # Format as comma-separated number
                    text_value = f"{float_value:,.2f}" 
            except:
                pass

            row_cells[current_col + j].text = text_value
            
            # Align numeric columns to the right
            if any(char.isdigit() for char in text_value) and not any(char.isalpha() for char in text_value):
                 row_cells[current_col + j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                 row_cells[current_col + j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

def generate_word_document_bytes(ticker: str, isin: str = "") -> BytesIO:
    """Fetches all Screener data, generates the Word document, and returns it as a BytesIO object."""
    
    # Data is fetched entirely from Screener now
    scraped_data = scrape_all_screener_data(ticker)

    document = Document()
    
    # Set document style
    style = document.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    # --- Main Title ---
    company_name = scraped_data.get('0. Company Summary', {}).get('Name', ticker)
    document.add_heading(f"Comprehensive Stock Analysis Report: {company_name} ({ticker})", 0)
    document.add_paragraph(f"Source: Screener.in | Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    document.add_paragraph("---")
    
    if 'Error' in scraped_data:
        document.add_paragraph(f"FATAL ERROR: Could not generate report due to scraping failure: {scraped_data['Error']}")
        doc_io = BytesIO()
        document.save(doc_io)
        doc_io.seek(0)
        return doc_io

    # --- Process and Add Sections to Document ---
    
    # 1. Company Summary (Name, Ticker, Industry, Latest Metrics)
    summary = scraped_data.pop('0. Company Summary', {})
    document.add_heading("1. Company Overview and Key Ratios", level=1)
    
    document.add_paragraph(f"Industry: {summary.get('Industry', 'N/A')}")
    document.add_paragraph(f"Key Financial Ratios (Latest Trailing Twelve Months/Yearly Data):")
    
    # Add the metrics table (transposed)
    if 'Latest Metrics' in summary and not summary['Latest Metrics'].empty:
        add_dataframe_to_word(document, summary['Latest Metrics'].T.reset_index().rename(columns={'index': 'Metric'}), table_style='Light List Accent 1')
    
    document.add_paragraph("---")

    # 2. Financial Tables (Quarterly, P&L, BS, CF, Shareholding)
    
    section_index = 2
    for title_key, df in scraped_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            document.add_heading(f"{section_index}. {title_key}", level=1)
            
            # For shareholding, ensure index is named correctly if applicable
            if 'Shareholding' in title_key:
                 df.index.name = 'Investor Category'

            add_dataframe_to_word(document, df, table_style='Table Grid')
            section_index += 1
            document.add_paragraph("---")

    # Save document to a BytesIO stream
    doc_io = BytesIO()
    document.save(doc_io)
    doc_io.seek(0)
    return doc_io

# ======================================================================================
# STREAMLIT UI & LOGIC (Updated display_stock_analysis)
# ======================================================================================

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
        
        # --- Export Button Section (Placed below the columns) ---
        st.divider()
        
        export_col, _ = st.columns([1.5, 5.5])
        with export_col:
            with st.spinner(f"Preparing comprehensive report for {ticker} from Screener.in..."):
                # Generate the document bytes dynamically
                # NOTE: ISIN is no longer necessary as the Cogencis URL is removed.
                doc_bytes = generate_word_document_bytes(ticker)
            
            st.download_button(
                label="⬇️ Export Full Report (DOCX)",
                data=doc_bytes,
                file_name=f"{ticker}_Screener_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True
            )
        # --- End Export Section ---


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
