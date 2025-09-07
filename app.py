import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go

# --- Technical Analysis Libraries ---
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# --- AutoGen Agent ---
# The AssistantAgent is the primary agent we will use.
from autogen_agentchat.agents import AssistantAgent

# =========================
# ENV / PAGE CONFIGURATION
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="📈 Stock Analyst Agent", page_icon="📈", layout="wide")
st.title("📈 Stock Analyst Agent (Real Data + AutoGen)")
st.caption("Enter a ticker symbol (e.g., AAPL, MSFT, TSLA). I’ll fetch data, analyze it, and the agent will recommend BUY / HOLD / SELL.")

if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY in the app secrets or a .env file.", icon="⚠️")

# =========================
# SESSION STATE
# =========================
def _init_state():
    # Set up the asyncio event loop for Streamlit.
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    # We no longer need to store model_client or team in session state.
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "latest_json" not in st.session_state:
        st.session_state.latest_json = None

_init_state()

# =========================
# DATA FETCHING & INDICATORS
# =========================
@st.cache_data(ttl=timedelta(minutes=15))
def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """Pulls price history and other info for a stock ticker."""
    tk = yf.Ticker(symbol)
    end = datetime.utcnow()
    start = end - timedelta(days=400) # Buffer for 200-day SMA
    hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
    if hist.empty:
        return {"hist": pd.DataFrame(), "fast": {}, "fin": {}}

    fast = tk.fast_info or {}
    return {"hist": hist, "fast": fast}


def compute_indicators(hist: pd.DataFrame) -> Dict[str, Any]:
    """Calculates technical indicators from the price history."""
    if hist.empty:
        return {"error": "No price history"}

    close = hist["Close"].dropna()
    if len(close) < 200: # Ensure enough data for all indicators
        return {"error": "Insufficient history (need ~200+ days)"}

    # Standard indicators
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = RSIIndicator(close, window=14).rsi()
    macd_calc = MACD(close, window_slow=26, window_fast=12, window_sign=9)

    # Additional indicators
    bollinger = BollingerBands(close, window=20, window_dev=2)
    atr = AverageTrueRange(hist['High'], hist['Low'], close, window=14)
    obv = OnBalanceVolumeIndicator(close, hist['Volume'])
    obv_series = obv.on_balance_volume()

    last = float(close.iloc[-1])
    vol = hist["Volume"].dropna()
    vol_avg_20 = float(vol.tail(20).mean())
    vol_last = float(vol.iloc[-1])

    return {
        "last_price": last,
        "sma20": float(sma20.iloc[-1]),
        "sma50": float(sma50.iloc[-1]),
        "sma200": float(sma200.iloc[-1]),
        "rsi14": float(rsi.iloc[-1]),
        "macd": float(macd_calc.macd().iloc[-1]),
        "macd_signal": float(macd_calc.macd_signal().iloc[-1]),
        "macd_hist": float(macd_calc.macd_diff().iloc[-1]),
        "bollinger_hband": float(bollinger.bollinger_hband().iloc[-1]),
        "bollinger_lband": float(bollinger.bollinger_lband().iloc[-1]),
        "atr14": float(atr.average_true_range().iloc[-1]),
        "obv_slope_20d": (obv_series.iloc[-1] - obv_series.iloc[-20]) / 20,
        "52w_high": float(close.tail(252).max()),
        "52w_low": float(close.tail(252).min()),
        "above_sma50": last > sma50.iloc[-1],
        "above_sma200": last > sma200.iloc[-1],
        "volume_ratio": vol_last / vol_avg_20 if vol_avg_20 else None,
    }


def extract_fundamentals(fast: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts key fundamental ratios from yfinance fast_info."""
    def safe(k, default=None):
        v = fast.get(k, default)
        try:
            return float(v)
        except (ValueError, TypeError):
            return v

    return {
        "market_cap": safe("marketCap"),
        "pe_ratio": safe("trailingPE"),
        "forward_pe": safe("forwardPE"),
        "pb_ratio": safe("priceToBook"),
        "dividend_yield": safe("dividendYield"),
    }


# =========================
# AUTOGEN AGENT SETUP
# =========================
def build_agent() -> AssistantAgent:
    """Creates and configures the AutoGen AssistantAgent."""
    # UPDATE: Use the standard llm_config dictionary for model configuration.
    llm_config = {
        "model": "gpt-4o-mini",
        "api_key": OPENAI_API_KEY,
        "temperature": 0.3,
    }

    system_message = (
        "You are a disciplined equity research assistant. You will be given technical and fundamental data.\n\n"
        "Output strict JSON with the following fields:\n"
        "{\n"
        '  "reasoning": "First, I analyzed the technicals... Next, I considered the fundamentals... Finally, combining these points, my recommendation is...",\n'
        '  "action": "BUY" | "HOLD" | "SELL",\n'
        '  "confidence": 0-100,\n'
        '  "technical_summary": "Summary of key technical signals.",\n'
        '  "fundamental_summary": "Summary of key fundamental signals.",\n'
        '  "risks": ["List of potential risks.", "..."],\n'
        '  "notes": "Any other important notes or context."\n'
        "}\n\n"
        "Rules:\n"
        "- First, write your step-by-step thought process in the 'reasoning' field.\n"
        "- Combine both technical + fundamental signals for your final decision.\n"
        "- Favor BUY if the trend is positive (price > SMA50 & SMA200, MACD>=0, RSI 45–65) and valuation is not excessive.\n"
        "- Favor SELL if the trend is negative (price < SMA200, MACD<0, RSI<40) or valuation/risks are severe.\n"
        "- Your entire response must be a single JSON object. No markdown, no extra text."
    )

    return AssistantAgent(
        name="stock_agent",
        system_message=system_message,
        llm_config=llm_config,
    )


async def _ask_agent_async(agent: AssistantAgent, payload: Dict[str, Any]) -> str:
    """Sends the data payload to the agent and gets a response."""
    # For a single response, we can use the generate_chat_response method.
    chat_response = await agent.a_generate_chat_response(
        messages=[{"role": "user", "content": json.dumps(payload)}]
    )
    # The response is an AIMessage object; we extract its content.
    return chat_response.chat_info.get("summary", "") if chat_response else ""

def ask_agent(agent: AssistantAgent, payload: Dict[str, Any]) -> str:
    """Wrapper to run the async agent call within Streamlit's sync environment."""
    loop = st.session_state.loop
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(_ask_agent_async(agent, payload))


# =========================
# STREAMLIT UI
# =========================
col_main = st.columns([2, 1, 1])
with col_main[0]:
    symbol = st.text_input("Ticker symbol", value="NVDA", placeholder="e.g., AAPL")
with col_main[1]:
    lookback = st.selectbox("Price lookback", ["1y", "6mo", "3mo"], index=0)
with col_main[2]:
    st.write("")
    st.write("")
    run_btn = st.button("Analyze", type="primary", use_container_width=True, disabled=(not symbol.strip() or not OPENAI_API_KEY))

st.divider()

if run_btn:
    ticker = symbol.strip().upper()
    with st.spinner(f"Fetching and analyzing {ticker}..."):
        try:
            # --- Data Processing ---
            data = fetch_stock_data(ticker)
            hist = data["hist"]

            if hist.empty:
                st.error(f"No price data found for symbol '{ticker}'. Please check the ticker.")
            else:
                inds = compute_indicators(hist)
                if "error" in inds:
                    st.error(f"Could not compute indicators for {ticker}: {inds['error']}")
                else:
                    fins = extract_fundamentals(data["fast"])
                    payload = {
                        "symbol": ticker,
                        "as_of": datetime.utcnow().isoformat() + "Z",
                        "technical": inds,
                        "fundamental": fins,
                    }

                    # --- Display Metrics & Chart ---
                    st.subheader(f"{ticker} – Key Metrics")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.json(inds, expanded=False)
                    with c2:
                        st.json(fins, expanded=False)

                    # Trim history for display
                    days = {"1y": 252, "6mo": 126, "3mo": 63}
                    hist_disp = hist.tail(days[lookback])

                    fig = go.Figure(data=[go.Candlestick(x=hist_disp.index,
                                    open=hist_disp['Open'], high=hist_disp['High'],
                                    low=hist_disp['Low'], close=hist_disp['Close'])])
                    fig.update_layout(title=f'{ticker} Price History', xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Agent Analysis ---
                    agent = build_agent()
                    raw_response = ask_agent(agent, payload)
                    decision = None
                    try:
                        # Find the JSON object within the response string
                        start = raw_response.find("{")
                        end = raw_response.rfind("}") + 1
                        if start != -1 and end != 0:
                            decision = json.loads(raw_response[start:end])
                    except (json.JSONDecodeError, IndexError):
                        st.error("Agent returned an invalid JSON response. Showing raw output:")
                        st.code(raw_response, language="text")

                    if decision and isinstance(decision, dict) and "action" in decision:
                        st.session_state.latest_json = decision
                        action = decision.get("action", "HOLD")
                        conf = decision.get("confidence", 50)

                        if action == "BUY":
                            st.success(f"**Recommendation: {action}** (Confidence: {conf}/100)")
                        elif action == "SELL":
                            st.error(f"**Recommendation: {action}** (Confidence: {conf}/100)")
                        else:
                            st.info(f"**Recommendation: {action}** (Confidence: {conf}/100)")

                        st.markdown("**Agent's Reasoning**")
                        st.info(decision.get("reasoning", "No reasoning provided."))

                        c1_res, c2_res = st.columns(2)
                        with c1_res:
                            st.markdown("**Technical Summary**")
                            st.write(decision.get("technical_summary", ""))
                        with c2_res:
                            st.markdown("**Fundamental Summary**")
                            st.write(decision.get("fundamental_summary", ""))

                        if decision.get("risks"):
                            st.markdown("**Risks**")
                            st.warning("- " + "\n- ".join(decision["risks"]))

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.divider()
st.caption("Data via yfinance. This is for educational purposes and not financial advice.")
