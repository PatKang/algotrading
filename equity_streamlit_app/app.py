"""
app.py — Equity Analysis Streamlit App
=======================================
Entry point.  Run with:   streamlit run app.py

Tabs
----
  🏠 Overview      — holistic snapshot per ticker across all three modules
  📊 Fundamentals  — Module 1: snapshot metrics + YoY growth charts
  📈 Technical     — Module 2: interactive charts, signals, Monte Carlo
  📰 Sentiment     — Module 3: VADER-scored headlines + Alpha Vantage

Adding a new module
-------------------
1. Create modules/your_module.py with a  render_your_module(tickers) function.
2. Add a new st.tab in the TABS section below and call your render function.
3. Optionally surface a summary card in _render_overview().
"""
from __future__ import annotations

# Load .env from this folder or any parent (picks up Equity_Analysis/.env locally;
# on Streamlit Community Cloud the file won't exist and this is a silent no-op).
from dotenv import load_dotenv
load_dotenv(override=False)   # won't overwrite vars already set in the environment

import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Equity Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── module imports ────────────────────────────────────────────────────────────
from modules.fundamentals import render_fundamentals
from modules.technical    import render_technical, _compute_signals
from modules.sentiment    import render_sentiment, _score_headlines, _get_vader
from utils.data           import get_info, get_history, get_infos
from utils.formatting     import fmt_val, fmt_large, fmt_pct, signal_badge

import numpy as np
import plotly.graph_objects as go


# ── default tickers ───────────────────────────────────────────────────────────
_DEFAULTS = ["ACMR", "GXO", "SMCI", "AAPL", "MSFT"]

_SUGGESTIONS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "AVGO", "PEP", "KO", "LLY", "TMO", "COST", "CSCO",
    "ACN", "MCD", "BAC", "NFLX", "CRM", "ADBE", "AMD", "INTC", "QCOM",
    "GXO", "ACMR", "SMCI", "BN", "NEM", "GS", "MS", "SPGI", "BLK",
]


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Equity Analysis")
    st.markdown(
        "Analyse and compare any publicly listed equity using live data from "
        "[yfinance](https://github.com/ranaroussi/yfinance)."
    )
    st.markdown("---")

    # Initialise session state on first load
    if "tickers" not in st.session_state:
        st.session_state["tickers"] = _DEFAULTS.copy()

    st.markdown("**Tickers**")

    # ── add a ticker ──────────────────────────────────────────────────────────
    add_col, btn_col = st.columns([3, 1])
    with add_col:
        new_ticker = st.text_input(
            "Add ticker:",
            value="",
            placeholder="e.g. NVDA",
            label_visibility="collapsed",
            key="ticker_input",
        ).strip().upper()
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)   # vertical alignment
        add_clicked = st.button("Add", use_container_width=True)

    if add_clicked and new_ticker:
        if new_ticker not in st.session_state["tickers"]:
            st.session_state["tickers"].append(new_ticker)
        # clear the input by resetting its key
        st.rerun()

    # ── quick-add from popular suggestions ───────────────────────────────────
    with st.expander("Quick-add popular tickers"):
        available = [t for t in _SUGGESTIONS if t not in st.session_state["tickers"]]
        chosen = st.multiselect(
            "Select to add:",
            options=available,
            default=[],
            key="quick_add",
            label_visibility="collapsed",
        )
        if chosen:
            for t in chosen:
                if t not in st.session_state["tickers"]:
                    st.session_state["tickers"].append(t)
            st.rerun()

    # ── current ticker list (removable) ──────────────────────────────────────
    st.markdown("**Active tickers** — click × to remove")
    to_remove = []
    cols = st.columns(3)
    for i, tkr in enumerate(st.session_state["tickers"]):
        with cols[i % 3]:
            if st.button(f"{tkr} ×", key=f"rm_{tkr}", use_container_width=True):
                to_remove.append(tkr)
    if to_remove:
        for t in to_remove:
            st.session_state["tickers"].remove(t)
        st.rerun()

    if not st.session_state["tickers"]:
        st.error("Add at least one ticker.")
        st.stop()

    tickers: list[str] = st.session_state["tickers"]

    st.markdown("---")
    st.markdown(
        "**Modules**\n"
        "- 🏠 Overview — quick cross-module snapshot\n"
        "- 📊 Fundamentals — ratios & YoY growth\n"
        "- 📈 Technical — charts, signals, MC forecast\n"
        "- 📰 Sentiment — VADER-scored headlines\n"
    )
    st.markdown("---")
    st.caption(
        "Data is cached for 15–60 min. "
        "Prices from yfinance · Sentiment via VADER · "
        "Optional enriched news via Alpha Vantage."
    )


# ── overview helpers ──────────────────────────────────────────────────────────
_OVERVIEW_METRICS = [
    ("Market Cap",    "marketCap"),
    ("Trailing P/E",  "trailingPE"),
    ("EV/EBITDA",     "enterpriseToEbitda"),
    ("Profit Margin", "profitMargins"),
    ("Revenue (TTM)", "totalRevenue"),
    ("Beta",          "beta"),
]


def _spark_fig(close) -> go.Figure:
    """Return a minimal sparkline figure for an overview card."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(close))),
        y=close.values,
        mode="lines",
        line=dict(width=1.5, color="#1f77b4"),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.15)",
    ))
    fig.update_layout(
        height=80,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _render_overview(tickers: list[str]) -> None:
    st.markdown(
        "A high-level, cross-module snapshot for every ticker. "
        "Dive into each tab for full analysis."
    )

    # ── fetch data ────────────────────────────────────────────────────────────
    with st.spinner("Loading overview data…"):
        infos     = get_infos(tickers)
        histories = {t: get_history(t, "3mo") for t in tickers}

        # sentiment aggregate (compound mean per ticker)
        vader_ok = _get_vader() is not None
        sent_scores: dict[str, float | None] = {}
        if vader_ok:
            sent_df = _score_headlines(tickers)
            if not sent_df.empty:
                sent_scores = (
                    sent_df.groupby("Ticker")["Compound"].mean().to_dict()
                )

        # technical signals
        tech_signals: dict[str, dict] = {}
        for tkr in tickers:
            h = histories.get(tkr)
            if h is not None and not h.empty:
                try:
                    tech_signals[tkr] = _compute_signals(tkr, h)
                except Exception:
                    pass

    # ── one card per ticker ───────────────────────────────────────────────────
    for tkr in tickers:
        info = infos.get(tkr, {})
        hist = histories.get(tkr)
        sig  = tech_signals.get(tkr)

        company = info.get("longName") or info.get("shortName") or tkr
        sector  = info.get("sector", "—")
        price   = info.get("currentPrice") or info.get("regularMarketPrice")
        prev    = info.get("regularMarketPreviousClose") or info.get("previousClose")
        day_chg = ((price - prev) / prev) if price and prev else None

        with st.expander(f"**{tkr}** — {company}  |  {sector}", expanded=True):
            top_cols = st.columns([1, 2, 1, 1])

            # ── price + sparkline ─────────────────────────────────────────────
            with top_cols[0]:
                if price:
                    sign  = "+" if day_chg and day_chg >= 0 else ""
                    delta = f"{sign}{day_chg*100:.2f}% today" if day_chg else ""
                    st.metric("Price", f"${price:.2f}", delta)
                else:
                    st.metric("Price", "—")

                if hist is not None and not hist.empty:
                    close_3m = hist["Close"].dropna().iloc[-60:]
                    st.plotly_chart(
                        _spark_fig(close_3m),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            # ── fundamentals snapshot ─────────────────────────────────────────
            with top_cols[1]:
                st.markdown("**Key Fundamentals**")
                fund_data = {
                    label: fmt_val(key, info.get(key))
                    for label, key in _OVERVIEW_METRICS
                }
                fund_df = (
                    pd.DataFrame.from_dict(fund_data, orient="index", columns=["Value"])
                )
                st.dataframe(fund_df, use_container_width=True, height=220)

            # ── technical signal ──────────────────────────────────────────────
            with top_cols[2]:
                st.markdown("**Technical Signal**")
                if sig:
                    st.markdown(signal_badge(sig["bias"]), unsafe_allow_html=True)
                    st.markdown(
                        f"🟢 {sig['n_bull']} bullish&nbsp;&nbsp;"
                        f"🔴 {sig['n_bear']} bearish",
                        unsafe_allow_html=True,
                    )
                    if sig["rsi"] is not None:
                        st.markdown(f"RSI: **{sig['rsi']:.1f}**")
                else:
                    st.caption("No data")

            # ── sentiment score ───────────────────────────────────────────────
            with top_cols[3]:
                st.markdown("**Headline Sentiment**")
                score = sent_scores.get(tkr)
                if score is not None:
                    color = (
                        "#4caf50" if score >= 0.05
                        else "#f44336" if score <= -0.05
                        else "#ffc107"
                    )
                    label = (
                        "Positive" if score >= 0.05
                        else "Negative" if score <= -0.05
                        else "Neutral"
                    )
                    st.markdown(
                        f'<span style="color:{color};font-size:1.6em;font-weight:700;">'
                        f"{score:+.3f}</span><br/>"
                        f'<span style="color:{color};">{label}</span>',
                        unsafe_allow_html=True,
                    )
                elif not vader_ok:
                    st.caption("Install vaderSentiment")
                else:
                    st.caption("No headlines found")


# ── needs pandas import in app.py for the overview DataFrame ──────────────────
import pandas as pd


# ── tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_fundamentals, tab_technical, tab_sentiment = st.tabs([
    "🏠 Overview",
    "📊 Fundamentals",
    "📈 Technical",
    "📰 Sentiment",
])

with tab_overview:
    _render_overview(tickers)

with tab_fundamentals:
    render_fundamentals(tickers)

with tab_technical:
    render_technical(tickers)

with tab_sentiment:
    render_sentiment(tickers)
