"""
modules/fundamentals.py
Module 1 — Fundamentals Comparison
  • Snapshot metrics table (all selected tickers side-by-side)
  • Year-over-year growth charts for key income-statement items
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get_info, get_financials, get_infos
from utils.formatting import fmt_val, fmt_pct_chg, fmt_large


# ── metric definitions ────────────────────────────────────────────────────────
SNAPSHOT_METRICS: list[tuple[str, str]] = [
    ("Market Cap",       "marketCap"),
    ("Enterprise Value", "enterpriseValue"),
    ("Trailing P/E",     "trailingPE"),
    ("Forward P/E",      "forwardPE"),
    ("PEG Ratio",        "pegRatio"),
    ("Price/Sales",      "priceToSalesTrailing12Months"),
    ("Price/Book",       "priceToBook"),
    ("EV/EBITDA",        "enterpriseToEbitda"),
    ("Profit Margin",    "profitMargins"),
    ("Operating Margin", "operatingMargins"),
    ("Gross Margin",     "grossMargins"),
    ("ROE",              "returnOnEquity"),
    ("ROA",              "returnOnAssets"),
    ("Revenue (TTM)",    "totalRevenue"),
    ("EBITDA (TTM)",     "ebitda"),
    ("Total Cash",       "totalCash"),
    ("Total Debt",       "totalDebt"),
    ("Debt/Equity",      "debtToEquity"),
    ("Current Ratio",    "currentRatio"),
    ("Dividend Yield",   "dividendYield"),
    ("Beta",             "beta"),
    ("52W High",         "fiftyTwoWeekHigh"),
    ("52W Low",          "fiftyTwoWeekLow"),
]

YOY_INCOME_METRICS: list[tuple[str, list[str]]] = [
    ("Revenue",          ["Total Revenue", "Operating Revenue"]),
    ("Gross Profit",     ["Gross Profit"]),
    ("Operating Income", ["Operating Income", "EBIT"]),
    ("Net Income",       ["Net Income", "Net Income Common Stockholders"]),
    ("EBITDA",           ["EBITDA", "Normalized EBITDA"]),
    ("EPS (Diluted)",    ["Diluted EPS", "Basic EPS"]),
]

YOY_BALANCE_METRICS: list[tuple[str, list[str]]] = [
    ("Total Cash", ["Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents"]),
    ("Total Debt",  ["Total Debt", "Long Term Debt And Capital Lease Obligation"]),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _extract_row(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name].dropna().sort_index()
    return None


def _yoy_pct(series: pd.Series) -> pd.Series:
    """Year-over-year % change, newest first (income_stmt cols are newest→oldest)."""
    s = series.sort_index()
    return s.pct_change().dropna()


def _snapshot_df(tickers: list[str], infos: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for label, key in SNAPSHOT_METRICS:
        row = {"Metric": label}
        for tkr in tickers:
            row[tkr] = fmt_val(key, infos.get(tkr, {}).get(key))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Metric")


# ── YoY chart ─────────────────────────────────────────────────────────────────
def _yoy_bar_chart(
    metric_label: str,
    candidates: list[str],
    tickers: list[str],
    fin_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> go.Figure | None:
    fig = go.Figure()
    has_data = False

    for tkr in tickers:
        income, balance = fin_data.get(tkr, (pd.DataFrame(), pd.DataFrame()))
        series = _extract_row(income, candidates)
        if series is None:
            series = _extract_row(balance, candidates)
        if series is None or len(series) < 2:
            continue

        # keep last 4 fiscal years, compute YoY %
        series = series.sort_index().iloc[-4:]
        chg    = series.pct_change().dropna()

        # x-axis: fiscal year labels
        years = [str(d.year) if hasattr(d, "year") else str(d) for d in chg.index]

        fig.add_trace(go.Bar(
            name=tkr,
            x=years,
            y=(chg.values * 100).round(1),
            text=[f"{v:.1f}%" for v in chg.values * 100],
            textposition="outside",
        ))
        has_data = True

    if not has_data:
        return None

    fig.update_layout(
        title=f"{metric_label} — Year-over-Year Growth (%)",
        barmode="group",
        yaxis_title="YoY %",
        xaxis_title="Fiscal Year",
        legend_title="Ticker",
        height=380,
        margin=dict(t=50, b=40, l=40, r=20),
        template="plotly_dark",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    return fig


# ── absolute level chart ──────────────────────────────────────────────────────
def _abs_line_chart(
    metric_label: str,
    candidates: list[str],
    tickers: list[str],
    fin_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> go.Figure | None:
    fig = go.Figure()
    has_data = False

    for tkr in tickers:
        income, balance = fin_data.get(tkr, (pd.DataFrame(), pd.DataFrame()))
        series = _extract_row(income, candidates)
        if series is None:
            series = _extract_row(balance, candidates)
        if series is None or series.empty:
            continue

        series = series.sort_index().iloc[-5:]
        years  = [str(d.year) if hasattr(d, "year") else str(d) for d in series.index]

        fig.add_trace(go.Scatter(
            name=tkr,
            x=years,
            y=series.values,
            mode="lines+markers",
            text=[fmt_large(v) for v in series.values],
            hovertemplate="%{x}: %{text}<extra>%{fullData.name}</extra>",
        ))
        has_data = True

    if not has_data:
        return None

    fig.update_layout(
        title=f"{metric_label} — Absolute Level",
        yaxis_title="Value (USD)",
        xaxis_title="Fiscal Year",
        legend_title="Ticker",
        height=350,
        margin=dict(t=50, b=40, l=40, r=20),
        template="plotly_dark",
    )
    return fig


# ── public render function ────────────────────────────────────────────────────
def render_fundamentals(tickers: list[str]) -> None:
    """Entry point called from app.py for the Fundamentals tab."""
    if not tickers:
        st.info("Add tickers in the sidebar to get started.")
        return

    # ── fetch data (cached) ───────────────────────────────────────────────────
    with st.spinner("Fetching fundamentals…"):
        infos    = get_infos(tickers)
        fin_data = {t: get_financials(t) for t in tickers}

    # ── section: snapshot table ───────────────────────────────────────────────
    st.subheader("Snapshot Metrics")
    snap_df = _snapshot_df(tickers, infos)
    st.dataframe(snap_df, use_container_width=True)

    st.divider()

    # ── section: YoY growth charts ────────────────────────────────────────────
    st.subheader("Year-over-Year Growth")

    all_yoy = YOY_INCOME_METRICS + YOY_BALANCE_METRICS
    metric_options = [label for label, _ in all_yoy]
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        options=metric_options,
        default=metric_options[:4],
    )

    view = st.radio(
        "Chart type:",
        ["YoY % Change", "Absolute Level"],
        horizontal=True,
    )

    for label, candidates in all_yoy:
        if label not in selected_metrics:
            continue

        if view == "YoY % Change":
            fig = _yoy_bar_chart(label, candidates, tickers, fin_data)
        else:
            fig = _abs_line_chart(label, candidates, tickers, fin_data)

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"No financial data available for **{label}**.")
