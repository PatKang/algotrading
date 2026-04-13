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
# Local run vs running on streamlit
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
        "Analyse and compare any publicly listed equity using live data from"
        "[yfinance](https://github.com/ranaroussi/yfinance) and new sentiment analysis via Alpha Vantage" \
        "This is not financial advice - but it shall give you a leg up in doing your own research and make some money"
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


def _mini_price_fig(close) -> go.Figure:
    """
    Small but readable 3-month price chart for the overview card.
    - Real dates on the x-axis (abbreviated month labels)
    - Price range on the y-axis (min / max ticks only)
    - Hover tooltip with full date + price
    - No zoom/pan — static so no reset needed
    """
    dates  = close.index
    prices = close.values
    lo, hi = float(prices.min()), float(prices.max())

    # colour by direction (last vs first)
    color = "#4caf50" if prices[-1] >= prices[0] else "#ef5350"
    fill  = color.replace(")", ",0.15)").replace("rgb", "rgba") if color.startswith("rgb") else (
        "rgba(76,175,80,0.15)" if color == "#4caf50" else "rgba(239,83,80,0.15)"
    )

    fig = go.Figure(go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        line=dict(width=1.5, color=color),
        fill="tozeroy",
        fillcolor=fill,
        hovertemplate="%{x|%d %b %Y}<br>$%{y:.2f}<extra></extra>",
    ))

    pad = (hi - lo) * 0.08 or 1.0
    fig.update_layout(
        height=110,
        margin=dict(t=4, b=4, l=4, r=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickformat="%b",          # "Jan", "Feb", …
            nticks=4,
            tickfont=dict(size=9, color="#888"),
            tickcolor="#888",
            linecolor="#444",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#2a2a2a",
            zeroline=False,
            tickformat="$.0f",
            nticks=3,
            range=[lo - pad, hi + pad],
            tickfont=dict(size=9, color="#888"),
            tickcolor="#888",
            linecolor="#444",
        ),
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

            # ── price + 52w range + sparkline ────────────────────────────────
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
                        _mini_price_fig(close_3m),
                        use_container_width=True,
                        config={
                            "displayModeBar": False,
                            "scrollZoom": False,
                            "doubleClick": False,
                            "staticPlot": False,
                        },
                    )

                # 52-week range bar (below the chart)
                low52  = info.get("fiftyTwoWeekLow")
                high52 = info.get("fiftyTwoWeekHigh")
                if price and low52 and high52 and high52 > low52:
                    pct_pos = (price - low52) / (high52 - low52)
                    pct_pos = max(0.0, min(1.0, pct_pos))
                    bar_pct = f"{pct_pos * 100:.1f}%"
                    st.markdown(
                        f"<div style='font-size:0.72em;color:#888;margin-top:2px;'>"
                        f"52W Range</div>"
                        f"<div style='display:flex;align-items:center;gap:4px;"
                        f"font-size:0.72em;margin-bottom:2px;'>"
                        f"<span style='white-space:nowrap'>${low52:.0f}</span>"
                        f"<div style='flex:1;background:#333;border-radius:4px;"
                        f"height:6px;position:relative;'>"
                        f"<div style='position:absolute;left:0;width:{bar_pct};"
                        f"height:100%;background:#1f77b4;border-radius:4px;'></div>"
                        f"<div style='position:absolute;left:{bar_pct};transform:"
                        f"translateX(-50%);top:-3px;width:12px;height:12px;"
                        f"background:#fff;border:2px solid #1f77b4;"
                        f"border-radius:50%;'></div>"
                        f"</div>"
                        f"<span style='white-space:nowrap'>${high52:.0f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
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
tab_overview, tab_fundamentals, tab_technical, tab_sentiment, tab_glossary = st.tabs([
    "🏠 Overview",
    "📊 Fundamentals",
    "📈 Technical",
    "📰 Sentiment",
    "📖 Glossary",
])

with tab_overview:
    _render_overview(tickers)

with tab_fundamentals:
    render_fundamentals(tickers)

with tab_technical:
    render_technical(tickers)

with tab_sentiment:
    render_sentiment(tickers)

with tab_glossary:
    st.markdown("## 📖 Glossary")
    st.markdown(
        "Reference guide for every indicator and methodology used in this app. "
        "Click a section to expand it."
    )

    # ── Technical Indicators ──────────────────────────────────────────────────
    st.markdown("### Technical Indicators")

    with st.expander("MA50 / MA200 — Moving Averages"):
        st.markdown("""
📎 *Sources: [Simple Moving Average — Investopedia](https://www.investopedia.com/terms/s/sma.asp) · [Golden Cross — Investopedia](https://www.investopedia.com/terms/g/goldencross.asp) · [Death Cross — Investopedia](https://www.investopedia.com/terms/d/deathcross.asp)*

**What it is**
A simple moving average (SMA) smooths price by averaging the closing price over a
rolling window.  MA50 covers the last 50 trading days (~10 weeks); MA200 covers the
last 200 trading days (~10 months).

**How it is calculated**
$$MA_n = \\frac{1}{n}\\sum_{i=0}^{n-1} P_{t-i}$$

**How to read it**
| Condition | Interpretation |
|---|---|
| Price > MA50 | Short-term uptrend; bullish signal |
| Price < MA50 | Short-term downtrend; bearish signal |
| Price > MA200 | Long-term uptrend; bullish signal |
| Price < MA200 | Long-term downtrend; bearish signal |
| MA50 > MA200 | **Golden Cross** — historically bullish |
| MA50 < MA200 | **Death Cross** — historically bearish |

**Limitations**  Moving averages are *lagging* — they react after the price has
already moved.  They work best in trending markets and can produce false signals in
sideways, choppy conditions.
""")

    with st.expander("RSI (14) — Relative Strength Index"):
        st.markdown("""
📎 *Source: [Relative Strength Index (RSI) — Investopedia](https://www.investopedia.com/terms/r/rsi.asp)*

**What it is**
RSI measures the speed and size of recent price changes to indicate whether an
asset is overbought or oversold.  This app uses a 14-period exponentially weighted
version (Wilder's smoothing).

**How it is calculated**
$$RSI = 100 - \\frac{100}{1 + RS}, \\quad RS = \\frac{\\text{Avg Gain}}{\\text{Avg Loss}}$$

Gains and losses are smoothed with an exponential moving average using
$\\alpha = 1/14$ (equivalent to Wilder's original method).

**How to read it**
| RSI value | Signal used in this app |
|---|---|
| < 30 | **Oversold** — potential reversal upward; bullish |
| 30 – 50 | Below mid-range; mild bearish lean |
| 50 – 70 | Above mid-range; mild bullish lean |
| > 70 | **Overbought** — potential reversal downward; bearish |

**Limitations**  RSI can remain overbought for extended periods in strong uptrends
and oversold in strong downtrends.  Use it alongside trend indicators, not in isolation.
""")

    with st.expander("MACD (12/26/9) — Moving Average Convergence Divergence"):
        st.markdown("""
📎 *Source: [MACD — Investopedia](https://www.investopedia.com/terms/m/macd.asp)*

**What it is**
MACD is a momentum indicator that shows the relationship between two exponential
moving averages (EMAs) of price.

**How it is calculated**
| Component | Formula |
|---|---|
| MACD line | EMA(12) − EMA(26) |
| Signal line | EMA(9) of the MACD line |
| Histogram | MACD line − Signal line |

**How to read it**
| Condition | Signal used in this app |
|---|---|
| MACD line > Signal line | **Bullish** — upward momentum building |
| MACD line < Signal line | **Bearish** — downward momentum building |
| Histogram > 0 (and growing) | Bullish momentum accelerating |
| Histogram < 0 (and shrinking) | Bearish momentum easing |
| MACD crosses above Signal | Bullish crossover (buy signal) |
| MACD crosses below Signal | Bearish crossover (sell signal) |

**Limitations**  MACD is also a lagging indicator.  In low-volatility, range-bound
markets frequent crossovers can generate noisy, unreliable signals.
""")

    with st.expander("BB — Bollinger Bands (20, ±2σ)"):
        st.markdown("""
📎 *Source: [Bollinger Bands — Investopedia](https://www.investopedia.com/terms/b/bollingerbands.asp)*

**What it is**
Bollinger Bands place an envelope around price based on a rolling standard deviation,
giving a dynamic measure of volatility and relative price level.

**How it is calculated**
| Band | Formula |
|---|---|
| Middle (BB Mid) | SMA(20) of closing price |
| Upper (BB Upper) | SMA(20) + 2 × σ(20) |
| Lower (BB Lower) | SMA(20) − 2 × σ(20) |

σ(20) is the rolling standard deviation of closing prices over the last 20 days.

**BB %ile — Bollinger Band Percentile**
The position of the current price within the band, expressed as a fraction:

$$BB\\% = \\frac{P - BB_{lower}}{BB_{upper} - BB_{lower}}$$

| BB %ile | Signal used in this app |
|---|---|
| < 20 % | Near lower band — potential oversold bounce; **bullish** |
| 20 % – 80 % | Mid-band — neutral; no directional signal |
| > 80 % | Near upper band — potential overbought reversal; **bearish** |

**Limitations**  Price can "walk the bands" during strong trends — touching the upper
band repeatedly in a rally is normal and not necessarily a reversal signal.
""")

    with st.expander("Volume signal"):
        st.markdown("""
📎 *Source: [Volume Analysis — Investopedia](https://www.investopedia.com/terms/v/volumeanalysis.asp)*

**What it is**
Compares today's trading volume to its 20-day average to flag unusually active sessions.

**Rule used in this app**
If the latest day's volume is **≥ 1.5× the 20-day average**, a *High volume* neutral
signal is raised.

High volume alone is not directional, but it amplifies the reliability of whatever
price signal coincides with it — a breakout on high volume is generally more
significant than the same move on thin volume.
""")

    with st.expander("Signal Bias — how the overall rating is calculated"):
        st.markdown("""
Each ticker receives a composite bias label based on the ratio of bullish to total
directional signals (bullish + bearish):

| Bull ratio | Label |
|---|---|
| ≥ 75 % | **Bullish** |
| 60 % – 74 % | **Leaning Bullish** |
| 41 % – 59 % | **Mixed** |
| 26 % – 40 % | **Leaning Bearish** |
| ≤ 25 % | **Bearish** |

Neutral signals (e.g. mid-Bollinger, high volume) are counted separately and do not
affect the bull ratio.
""")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    st.markdown("### Monte Carlo Price Forecast")

    with st.expander("Method overview — bootstrapped return simulation"):
        st.markdown("""
📎 *Sources: [Monte Carlo Simulation — Investopedia](https://www.investopedia.com/terms/m/montecarlosimulation.asp) · [Bootstrap in Statistics — Investopedia](https://www.investopedia.com/terms/b/bootstrap.asp)*

The Monte Carlo module projects a range of possible future prices by repeatedly
sampling from the stock's own historical daily returns — rather than assuming a
fixed normal distribution.  This makes the forecast inherit the actual skewness,
fat tails, and clustering present in the data.

**Step-by-step**

1. **Compute daily log returns** from the full price history available for the chosen period.
2. **Weight the returns** using an exponential decay so that recent days matter more
   than distant ones (controlled by the *halflife* parameter — see below).
3. **Bootstrap paths**: for each of the *N* simulations, draw *H* daily returns at
   random (with replacement) from the weighted distribution, then chain them into a
   price path:
   $$P_t = P_0 \\prod_{i=1}^{t}(1 + r_i)$$
4. **Compute percentile bands** across all paths at each future date to produce the
   P10, P25, P50, P75, and P90 fan chart.
""")

    with st.expander("Horizon — trading days"):
        st.markdown("""
The number of **business days** (Mon–Fri, excluding weekends) to project forward.
Future dates are generated with `pandas.bdate_range` so weekends are automatically
skipped.

| Slider value | Approximate calendar equivalent |
|---|---|
| 5 | 1 week |
| 21 | 1 month |
| 63 | 3 months |
""")

    with st.expander("Simulations (N)"):
        st.markdown("""
The number of independent price paths generated.  More paths reduce sampling noise
and make the percentile bands smoother and more stable, at the cost of compute time.

| N | Typical use |
|---|---|
| 500 | Quick exploratory run |
| 1 000 | Default — good balance of speed and stability |
| 5 000 | High-confidence bands; noticeably slower |
""")

    with st.expander("Return halflife — exponential decay weighting"):
        st.markdown("""
📎 *Source: [Exponential Moving Average — Investopedia](https://www.investopedia.com/terms/e/ema.asp)*

Controls how quickly the influence of past returns fades.  A day *k* periods ago
receives a weight proportional to:

$$w_k = \\lambda^k, \\quad \\lambda = 0.5^{\\,1/\\text{halflife}}$$

At exactly *halflife* days ago, the weight is half that of today.

| Halflife | Effect |
|---|---|
| Short (5–10 d) | Only very recent returns influence the simulation; captures current regime but more random noise |
| Medium (21 d) | Default — balances recent behaviour with a reasonable history |
| Long (42–63 d) | Draws from a broader history; smoother but slower to react to regime changes |
""")

    with st.expander("P10, P25, P50, P75, P90 — percentile bands"):
        st.markdown("""
📎 *Source: [Percentile — Investopedia](https://www.investopedia.com/terms/p/percentile.asp)*

At each future date, all *N* simulated prices are sorted and the following quantiles
are read off:

| Label | Meaning |
|---|---|
| **P10** | 10 % of paths ended *below* this price — pessimistic tail |
| **P25** | 25 % of paths ended below — lower quartile |
| **P50** | Median simulated price — equal number of paths above and below |
| **P75** | 75 % of paths ended below — upper quartile |
| **P90** | 90 % of paths ended below — optimistic tail |

The **P10–P90 band** (lighter shading) captures 80 % of all simulated outcomes.
The **P25–P75 band** (darker shading) captures the central 50 %.

**Important caveats**
- The simulation is based purely on past returns and contains no fundamental or macro
  information.
- It assumes the return distribution is stationary (same regime going forward as in
  the history window), which is rarely true in practice.
- Percentile bands should be read as *conditional on the model*, not as probability
  guarantees about the real future price.
""")

    # ── General abbreviations ─────────────────────────────────────────────────
    st.markdown("### Quick-reference abbreviations")
    st.markdown("""
| Abbreviation | Full name |
|---|---|
| BB | Bollinger Bands |
| BB %ile | Bollinger Band percentile (price position within the band) |
| EMA | Exponential Moving Average |
| HL | Halflife (Monte Carlo decay parameter) |
| MA50 | 50-day Simple Moving Average |
| MA200 | 200-day Simple Moving Average |
| MACD | Moving Average Convergence Divergence |
| P10 … P90 | 10th … 90th percentile of simulated price paths |
| RSI | Relative Strength Index |
| SMA | Simple Moving Average |
| σ | Standard deviation |
| TTM | Trailing Twelve Months |
""")

