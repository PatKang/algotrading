"""
modules/technical.py
Module 2 — Technical Analysis
  • 4-panel interactive Plotly chart per ticker  (Price+BB+MAs, Volume, RSI, MACD)
  • Signal summary table comparing all tickers
  • Normalised price comparison (rebased to 100)
  • Monte Carlo price forecast fan chart
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import get_history, get_histories


# ── indicator helpers ─────────────────────────────────────────────────────────
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line   = (
        series.ewm(span=fast, adjust=False).mean()
        - series.ewm(span=slow, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── 4-panel chart ─────────────────────────────────────────────────────────────
def _make_tech_chart(ticker: str, hist: pd.DataFrame) -> go.Figure:
    close  = hist["Close"]
    volume = hist["Volume"]

    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    bb_m  = close.rolling(20).mean()
    bb_s  = close.rolling(20).std()
    bb_u  = bb_m + 2 * bb_s
    bb_l  = bb_m - 2 * bb_s

    rsi_s                       = _rsi(close)
    macd_line, sig_line, hist_m = _macd(close)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.16, 0.17, 0.17],
        vertical_spacing=0.03,
        subplot_titles=[f"{ticker} — Price", "Volume", "RSI (14)", "MACD (12/26/9)"],
    )

    # ── panel 1: price ────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=close, name="Price",
            increasing_line_color="#4caf50", decreasing_line_color="#f44336",
        ), row=1, col=1,
    )
    for y, name, color, dash in [
        (ma50,  "MA 50",  "#ffc107", "solid"),
        (ma200, "MA 200", "#e91e63", "dash"),
        (bb_u,  "BB Upper", "rgba(100,149,237,0.7)", "dot"),
        (bb_l,  "BB Lower", "rgba(100,149,237,0.7)", "dot"),
        (bb_m,  "BB Mid",   "rgba(100,149,237,0.4)", "dot"),
    ]:
        fig.add_trace(
            go.Scatter(x=hist.index, y=y, name=name,
                       line=dict(color=color, dash=dash, width=1.2),
                       showlegend=True),
            row=1, col=1,
        )

    # Bollinger band fill
    fig.add_trace(
        go.Scatter(
            x=pd.concat([pd.Series(hist.index), pd.Series(hist.index[::-1])]),
            y=pd.concat([bb_u, bb_l[::-1]]),
            fill="toself",
            fillcolor="rgba(100,149,237,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="BB Band", showlegend=False,
        ), row=1, col=1,
    )

    # ── panel 2: volume ───────────────────────────────────────────────────────
    colors_vol = [
        "#4caf50" if c >= o else "#f44336"
        for c, o in zip(hist["Close"], hist["Open"])
    ]
    fig.add_trace(
        go.Bar(x=hist.index, y=volume, name="Volume",
               marker_color=colors_vol, showlegend=False),
        row=2, col=1,
    )

    # ── panel 3: RSI ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=hist.index, y=rsi_s, name="RSI",
                   line=dict(color="#ab47bc", width=1.5), showlegend=False),
        row=3, col=1,
    )
    for level, color in [(70, "rgba(244,67,54,0.35)"), (30, "rgba(76,175,80,0.35)")]:
        fig.add_hline(y=level, line_dash="dash",
                      line_color=color, row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)",
                  line_width=0, row=3, col=1)

    # ── panel 4: MACD ─────────────────────────────────────────────────────────
    bar_colors = [
        "#4caf50" if v >= 0 else "#f44336" for v in hist_m.fillna(0)
    ]
    fig.add_trace(
        go.Bar(x=hist.index, y=hist_m, name="Histogram",
               marker_color=bar_colors, showlegend=False),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(x=hist.index, y=macd_line, name="MACD",
                   line=dict(color="#29b6f6", width=1.4), showlegend=False),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(x=hist.index, y=sig_line, name="Signal",
                   line=dict(color="#ff7043", width=1.4, dash="dot"),
                   showlegend=False),
        row=4, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey",
                  line_width=0.8, row=4, col=1)

    fig.update_layout(
        height=720,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(t=60, b=20, l=50, r=20),
    )
    fig.update_yaxes(row=3, col=1, range=[0, 100])
    return fig


# ── signal summary ────────────────────────────────────────────────────────────
def _compute_signals(ticker: str, hist: pd.DataFrame) -> dict:
    """Compute a rich signal dict for one ticker."""
    close  = hist["Close"].dropna()
    volume = hist["Volume"].reindex(close.index)
    last   = float(close.iloc[-1])

    rsi_s                       = _rsi(close)
    macd_line, sig_line, hist_m = _macd(close)
    ma50   = close.rolling(50).mean()
    ma200  = close.rolling(200).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_u   = float((bb_mid + 2 * bb_std).iloc[-1])
    bb_l   = float((bb_mid - 2 * bb_std).iloc[-1])
    bb_pct = (last - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) else 0.5

    rsi         = float(rsi_s.iloc[-1]) if not rsi_s.isna().all() else None
    macd_val    = float(macd_line.iloc[-1])
    sig_val     = float(sig_line.iloc[-1])
    hist_val    = float(hist_m.iloc[-1])
    ma50_val    = float(ma50.iloc[-1]) if not ma50.isna().all() else None
    ma200_val   = float(ma200.iloc[-1]) if not ma200.isna().all() else None
    vol_avg     = float(volume.rolling(20).mean().iloc[-1]) if len(volume) > 20 else None
    vol_last    = float(volume.iloc[-1])

    bullish, bearish, neutral = [], [], []

    # ── MA signals ─────────────────────────────────────────────────────
    if ma50_val and ma200_val:
        if last > ma50_val:  bullish.append("Price above MA50")
        else:                bearish.append("Price below MA50")
        if last > ma200_val: bullish.append("Price above MA200")
        else:                bearish.append("Price below MA200")
        if ma50_val > ma200_val: bullish.append("Golden Cross (MA50 > MA200)")
        else:                    bearish.append("Death Cross (MA50 < MA200)")

    # ── RSI signals ────────────────────────────────────────────────────
    if rsi is not None:
        if rsi < 30:       bullish.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:     bearish.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 50:     bullish.append(f"RSI above 50 ({rsi:.1f})")
        else:              bearish.append(f"RSI below 50 ({rsi:.1f})")

    # ── MACD signals ───────────────────────────────────────────────────
    if macd_val > sig_val:   bullish.append("MACD above signal line")
    else:                    bearish.append("MACD below signal line")
    if hist_val > 0:         bullish.append("MACD histogram positive")
    else:                    bearish.append("MACD histogram negative")

    # ── Bollinger Band signals ─────────────────────────────────────────
    if bb_pct < 0.2:         bullish.append(f"Near BB lower band ({bb_pct:.0%})")
    elif bb_pct > 0.8:       bearish.append(f"Near BB upper band ({bb_pct:.0%})")
    else:                    neutral.append(f"Mid Bollinger Band ({bb_pct:.0%})")

    # ── Volume signal ──────────────────────────────────────────────────
    if vol_avg and vol_last > vol_avg * 1.5:
        neutral.append(f"High volume ({vol_last/vol_avg:.1f}x avg)")

    n_bull, n_bear = len(bullish), len(bearish)
    total = n_bull + n_bear
    ratio = n_bull / total if total else 0.5

    if   ratio >= 0.75: bias = "Bullish"
    elif ratio >= 0.60: bias = "Leaning Bullish"
    elif ratio <= 0.25: bias = "Bearish"
    elif ratio <= 0.40: bias = "Leaning Bearish"
    else:               bias = "Mixed"

    return {
        "ticker":   ticker,
        "bias":     bias,
        "n_bull":   n_bull,
        "n_bear":   n_bear,
        "bullish":  bullish,
        "bearish":  bearish,
        "neutral":  neutral,
        "rsi":      rsi,
        "ma50":     ma50_val,
        "ma200":    ma200_val,
        "last":     last,
        "bb_pct":   bb_pct,
    }


# ── signal comparison table ───────────────────────────────────────────────────
def _signal_table(signals: list[dict]) -> pd.DataFrame:
    rows = []
    for s in signals:
        rows.append({
            "Ticker":     s["ticker"],
            "Signal":     s["bias"],
            "Bull":       s["n_bull"],
            "Bear":       s["n_bear"],
            "RSI":        f"{s['rsi']:.1f}" if s["rsi"] else "—",
            "Price":      f"${s['last']:.2f}",
            "MA50":       f"${s['ma50']:.2f}" if s["ma50"] else "—",
            "MA200":      f"${s['ma200']:.2f}" if s["ma200"] else "—",
            "BB %ile":    f"{s['bb_pct']:.0%}" if s["bb_pct"] is not None else "—",
        })
    return pd.DataFrame(rows).set_index("Ticker")


# ── normalised price comparison ───────────────────────────────────────────────
def _price_comparison_chart(
    tickers: list[str],
    histories: dict[str, pd.DataFrame],
    period: str,
) -> go.Figure:
    fig = go.Figure()
    for tkr in tickers:
        hist = histories.get(tkr)
        if hist is None or hist.empty:
            continue
        close = hist["Close"].dropna()
        rebased = close / close.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased,
            name=tkr, mode="lines",
            hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:.1f}}<extra>{tkr}</extra>",
        ))
    fig.update_layout(
        title=f"Normalised Price Performance ({period}, rebased to 100)",
        yaxis_title="Index (base = 100)",
        height=400,
        template="plotly_dark",
        margin=dict(t=50, b=40, l=50, r=20),
        legend_title="Ticker",
    )
    fig.add_hline(y=100, line_dash="dash", line_color="grey", line_width=1)
    return fig


# ── Monte Carlo price forecast ─────────────────────────────────────────────────
def _monte_carlo_chart(
    ticker: str,
    hist: pd.DataFrame,
    horizon: int = 21,
    n_sims: int = 1000,
    halflife: int = 21,
    seed: int = 42,
) -> go.Figure:
    close    = hist["Close"].dropna()
    rets     = close.pct_change().dropna().values
    last     = float(close.iloc[-1])
    last_dt  = close.index[-1]

    # exponential decay weights
    lam     = 0.5 ** (1.0 / halflife)
    raw_w   = lam ** np.arange(len(rets) - 1, -1, -1)
    weights = raw_w / raw_w.sum()

    rng      = np.random.default_rng(seed)
    idx      = rng.choice(len(rets), size=(horizon, n_sims), replace=True, p=weights)
    sampled  = rets[idx]
    paths    = last * np.cumprod(1 + sampled, axis=0)

    # percentile bands
    p10, p25, p50, p75, p90 = [
        np.percentile(paths, q, axis=1) for q in [10, 25, 50, 75, 90]
    ]
    future_dates = pd.bdate_range(start=last_dt, periods=horizon + 1)[1:]

    fig = go.Figure()
    # Historical (last 60 days)
    hist_plot = close.iloc[-60:]
    fig.add_trace(go.Scatter(
        x=hist_plot.index, y=hist_plot,
        name="Historical", line=dict(color="#90caf9", width=2),
    ))
    # Forecast bands
    x_fwd = list(future_dates)
    for low, high, color, name in [
        (p10, p90, "rgba(76,175,80,0.12)",  "P10–P90"),
        (p25, p75, "rgba(76,175,80,0.25)",  "P25–P75"),
    ]:
        fig.add_trace(go.Scatter(
            x=x_fwd + x_fwd[::-1],
            y=list(high) + list(low[::-1]),
            fill="toself", fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"),
            name=name, showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=x_fwd, y=p50,
        name="P50 (median)",
        line=dict(color="#4caf50", width=2, dash="dash"),
    ))
    fig.add_vline(x=str(last_dt.date()), line_dash="dot",
                  line_color="grey", line_width=1)

    fig.update_layout(
        title=f"{ticker} — Monte Carlo Forecast ({horizon}d, {n_sims:,} paths, HL={halflife}d)",
        yaxis_title="Price (USD)",
        height=420,
        template="plotly_dark",
        margin=dict(t=50, b=40, l=50, r=20),
        legend_title="",
    )
    return fig, {"P10": p10[-1], "P25": p25[-1], "P50": p50[-1],
                 "P75": p75[-1], "P90": p90[-1]}


# ── public render function ────────────────────────────────────────────────────
def render_technical(tickers: list[str]) -> None:
    """Entry point called from app.py for the Technical tab."""
    if not tickers:
        st.info("Add tickers in the sidebar to get started.")
        return

    # ── controls ──────────────────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        period = st.selectbox(
            "Price history period:",
            ["6mo", "1y", "2y", "5y"],
            index=2,
            key="tech_period",
        )
    with col_r:
        selected = st.selectbox(
            "Detailed chart for:",
            tickers,
            key="tech_selected",
        )

    # ── fetch data (cached) ───────────────────────────────────────────────────
    with st.spinner("Fetching price data…"):
        histories = get_histories(tickers, period)

    # ── signal table (all tickers) ────────────────────────────────────────────
    st.subheader("Signal Summary — All Tickers")
    signals = []
    for tkr in tickers:
        h = histories.get(tkr)
        if h is not None and not h.empty:
            signals.append(_compute_signals(tkr, h))

    if signals:
        sig_df = _signal_table(signals)

        # colour the Signal column
        def _color_signal(val):
            colours = {
                "Bullish": "color:#4caf50;font-weight:700",
                "Leaning Bullish": "color:#8bc34a;font-weight:700",
                "Mixed": "color:#ffc107;font-weight:700",
                "Leaning Bearish": "color:#ff9800;font-weight:700",
                "Bearish": "color:#f44336;font-weight:700",
            }
            return colours.get(val, "")

        styled = sig_df.style.applymap(_color_signal, subset=["Signal"])
        st.dataframe(styled, use_container_width=True)

    st.divider()

    # ── normalised comparison chart ───────────────────────────────────────────
    st.subheader("Price Comparison (normalised)")
    comp_fig = _price_comparison_chart(tickers, histories, period)
    st.plotly_chart(comp_fig, use_container_width=True)

    st.divider()

    # ── detailed chart for selected ticker ────────────────────────────────────
    st.subheader(f"Detailed Chart — {selected}")
    hist = histories.get(selected)
    if hist is None or hist.empty:
        st.warning(f"No price data found for {selected}.")
    else:
        tech_fig = _make_tech_chart(selected, hist)
        st.plotly_chart(tech_fig, use_container_width=True)

        # signal detail for selected ticker
        sig = next((s for s in signals if s["ticker"] == selected), None)
        if sig:
            bc, nc, mc = st.columns(3)
            with bc:
                st.markdown("**Bullish signals**")
                for item in sig["bullish"]:
                    st.markdown(f"  🟢 {item}")
            with nc:
                st.markdown("**Neutral signals**")
                for item in sig["neutral"]:
                    st.markdown(f"  🟡 {item}")
            with mc:
                st.markdown("**Bearish signals**")
                for item in sig["bearish"]:
                    st.markdown(f"  🔴 {item}")

    st.divider()

    # ── Monte Carlo forecast ──────────────────────────────────────────────────
    st.subheader(f"Monte Carlo Forecast — {selected}")
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        mc_horizon  = st.slider("Horizon (trading days):", 5, 63, 21, key="mc_h")
    with mc_col2:
        mc_n_sims   = st.select_slider("Simulations:", [500, 1000, 5000], value=1000, key="mc_n")
    with mc_col3:
        mc_halflife = st.slider("Return halflife (days):", 5, 63, 21, key="mc_hl")

    if hist is not None and not hist.empty:
        mc_fig, mc_stats = _monte_carlo_chart(
            selected, hist,
            horizon=mc_horizon,
            n_sims=mc_n_sims,
            halflife=mc_halflife,
        )
        st.plotly_chart(mc_fig, use_container_width=True)

        # summary stats
        last_price = float(hist["Close"].iloc[-1])
        stat_cols  = st.columns(5)
        labels     = ["P10", "P25", "P50 (median)", "P75", "P90"]
        keys       = ["P10", "P25", "P50", "P75", "P90"]
        for col, lbl, key in zip(stat_cols, labels, keys):
            price = mc_stats[key]
            ret   = (price / last_price - 1) * 100
            sign  = "+" if ret >= 0 else ""
            col.metric(lbl, f"${price:.2f}", f"{sign}{ret:.1f}%")
