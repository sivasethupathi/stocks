import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD

# AutoGen v0.4 (AgentChat)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# =========================
# ENV / PAGE
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="📈 Stock Analyst Agent", page_icon="📈", layout="wide")
st.title("📈 Stock Analyst Agent (Real Data + AutoGen)")
st.caption("Enter a ticker symbol (e.g., AAPL, MSFT, TSLA). I’ll fetch data, analyze it, and the agent will recommend BUY / HOLD / SELL.")

if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in your environment or a .env file.", icon="⚠️")


# =========================
# Session State
# =========================
def _init_state():
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    if "model_client" not in st.session_state:
        st.session_state.model_client: Optional[OpenAIChatCompletionClient] = None

    if "agent" not in st.session_state:
        st.session_state.agent: Optional[AssistantAgent] = None

    if "team" not in st.session_state:
        st.session_state.team: Optional[RoundRobinGroupChat] = None

    if "latest_json" not in st.session_state:
        st.session_state.latest_json: Optional[Dict[str, Any]] = None

_init_state()


# =========================
# Data / Indicator Helpers
# =========================
def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """
    Pull price history (1y daily), fast info, and basic financial ratios if available.
    """
    tk = yf.Ticker(symbol)

    # Price history
    end = datetime.utcnow()
    start = end - timedelta(days=365 + 30)  # little buffer
    hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)

    # Fast info (robust vs legacy .info)
    fast = {}
    try:
        fast = tk.fast_info or {}
    except Exception:
        fast = {}

    # Fundamentals (best-effort; yfinance varies by ticker)
    fin = {}
    try:
        # yfinance v0.2+ exposes .get_financials, .get_income_stmt, etc. as DataFrames
        income = tk.income_stmt
        bal = tk.balance_sheet
        cash = tk.cashflow
        fin = {
            "income_stmt_cols": list(income.columns) if isinstance(income, pd.DataFrame) else [],
            "balance_sheet_cols": list(bal.columns) if isinstance(bal, pd.DataFrame) else [],
            "cashflow_cols": list(cash.columns) if isinstance(cash, pd.DataFrame) else [],
        }
    except Exception:
        pass

    return {"hist": hist, "fast": fast, "fin": fin}


def compute_indicators(hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty:
        return {"error": "No price history"}

    close = hist["Close"].dropna()
    if len(close) < 60:
        return {"error": "Insufficient history (need ~60+ days)"}

    # Simple MAs
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # RSI
    rsi = RSIIndicator(close, window=14).rsi()

    # MACD
    macd_calc = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd = macd_calc.macd()
    macd_signal = macd_calc.macd_signal()
    macd_diff = macd_calc.macd_diff()

    # 52w high/low
    last_252 = close.tail(252)
    high_52w = float(last_252.max())
    low_52w = float(last_252.min())

    last = float(close.iloc[-1])
    above_50 = last > float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else False
    above_200 = last > float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else False

    # Volume trend
    vol = hist["Volume"].dropna()
    vol_avg_20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol_last = float(vol.iloc[-1]) if len(vol) else np.nan
    vol_ratio = (vol_last / vol_avg_20) if vol_avg_20 else np.nan

    return {
        "last_price": last,
        "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None,
        "sma50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None,
        "rsi14": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else None,
        "macd_signal": float(macd_signal.iloc[-1]) if not np.isnan(macd_signal.iloc[-1]) else None,
        "macd_hist": float(macd_diff.iloc[-1]) if not np.isnan(macd_diff.iloc[-1]) else None,
        "52w_high": high_52w,
        "52w_low": low_52w,
        "above_sma50": above_50,
        "above_sma200": above_200,
        "volume_last": vol_last,
        "volume_avg20": vol_avg_20,
        "volume_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else None,
    }


def extract_fundamentals(fast: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull whatever is reliably present in .fast_info (yfinance’s newer API).
    Fields may be missing per ticker.
    """
    def safe(k, default=None):
        v = fast.get(k, default)
        try:
            return float(v)
        except Exception:
            return v

    out = {
        "market_cap": safe("market_cap"),
        "pe_ratio": safe("trailing_pe"),
        "forward_pe": safe("forward_pe"),
        "pb_ratio": safe("price_to_book"),
        "dividend_yield": safe("dividend_yield"),
    }
    return out


# =========================
# Agent (LLM) Setup
# =========================
def build_agent() -> AssistantAgent:
    if st.session_state.model_client is None:
        st.session_state.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.3,  # more deterministic for decisions
        )

    system_message = (
        "You are a disciplined equity research assistant. You will be given:\n"
        "1) Technical indicators (RSI, MACD, SMAs, 52w range, volume ratio)\n"
        "2) Fundamental signals (PE, PB, market cap, dividend yield)\n\n"
        "Output strict JSON with fields:\n"
        "{\n"
        '  "action": "BUY" | "HOLD" | "SELL",\n'
        '  "confidence": 0-100,\n'
        '  "technical_summary": "…",\n'
        '  "fundamental_summary": "…",\n'
        '  "risks": ["…", "…"],\n'
        '  "notes": "…"\n'
        "}\n\n"
        "Rules:\n"
        "- Combine both technical + fundamental signals.\n"
        "- Favor BUY if trend is positive (price > SMA50 & SMA200, MACD>=0, RSI 45–65) and valuation not excessive (PE or PB reasonable vs sector).\n"
        "- Favor SELL if trend is negative (price < SMA200, MACD<0, RSI<40) or valuation/risks are severe.\n"
        "- Otherwise HOLD.\n"
        "- Be conservative if data is missing.\n"
        "- JSON only. No markdown, no extra text."
    )
    agent = AssistantAgent(
        name="stock_agent",
        system_message=system_message,
        model_client=st.session_state.model_client,
    )
    return agent


async def _ask_agent_async(agent: AssistantAgent, payload: Dict[str, Any]) -> str:
    # We use a 1-agent "team" just to reuse the same run loop style if you later add more tools/agents
    team = RoundRobinGroupChat([agent], max_turns=1)
    msg = json.dumps(payload)
    result = await team.run(task=msg)
    # get last message
    if result.messages:
        return result.messages[-1].content
    return ""


def ask_agent(agent: AssistantAgent, payload: Dict[str, Any]) -> str:
    loop = st.session_state.loop
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(_ask_agent_async(agent, payload))


# =========================
# UI
# =========================
col = st.columns([2, 1, 1])
with col[0]:
    symbol = st.text_input("Ticker symbol", value="", placeholder="e.g., AAPL")

with col[1]:
    lookback = st.selectbox("Price lookback", ["1y", "6mo", "3mo"], index=0)

with col[2]:
    run_btn = st.button("Analyze", type="primary", use_container_width=True, disabled=(not symbol.strip()))

st.divider()

if run_btn:
    ticker = symbol.strip().upper()
    with st.spinner(f"Fetching and analyzing {ticker}…"):
        try:
            data = fetch_stock_data(ticker)
            hist = data["hist"]
            fast = data["fast"]

            if hist is None or hist.empty:
                st.error("No price data found for this symbol.")
            else:
                # Trim lookback for display
                if lookback == "6mo":
                    hist_disp = hist.tail(126)  # ~126 trading days
                elif lookback == "3mo":
                    hist_disp = hist.tail(63)
                else:
                    hist_disp = hist

                inds = compute_indicators(hist)
                fins = extract_fundamentals(fast)
                payload = {
                    "symbol": ticker,
                    "as_of": datetime.utcnow().isoformat() + "Z",
                    "technical": inds,
                    "fundamental": fins,
                }

                # Show raw metrics
                st.subheader(f"{ticker} – Key Metrics")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Technical**")
                    st.json(inds, expanded=False)
                with c2:
                    st.markdown("**Fundamental**")
                    st.json(fins, expanded=False)

                # Chart
                st.markdown("**Price (Adj Close)**")
                st.line_chart(hist_disp["Close"])

                # Call agent for decision
                agent = build_agent()
                raw = ask_agent(agent, payload)

                # Try parse
                decision = None
                try:
                    decision = json.loads(raw)
                except Exception:
                    # Sometimes models add stray characters—try a crude fix:
                    try:
                        start = raw.find("{")
                        end = raw.rfind("}")
                        if start != -1 and end != -1:
                            decision = json.loads(raw[start:end+1])
                    except Exception:
                        decision = None

                if not decision or not isinstance(decision, dict) or "action" not in decision:
                    st.error("Agent returned an invalid response. Showing raw output.")
                    st.code(raw)
                else:
                    st.session_state.latest_json = decision
                    a = decision.get("action", "HOLD")
                    conf = decision.get("confidence", 50)
                    st.success(f"**Recommendation: {a}** (confidence: {conf}/100)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Technical Summary**")
                        st.write(decision.get("technical_summary", ""))
                    with c2:
                        st.markdown("**Fundamental Summary**")
                        st.write(decision.get("fundamental_summary", ""))

                    if decision.get("risks"):
                        st.markdown("**Risks**")
                        st.write("- " + "\n- ".join(decision["risks"]))
                    if decision.get("notes"):
                        st.markdown("**Notes**")
                        st.write(decision["notes"])

        except Exception as e:
            st.error(f"Error: {e}")

# Footer / debug
st.divider()
st.caption("Data via yfinance. This is educational, not financial advice.")
