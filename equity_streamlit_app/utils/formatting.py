"""
utils/formatting.py
Shared number-formatting helpers used by all three modules.
"""
from __future__ import annotations

import numpy as np


# ── large numbers ─────────────────────────────────────────────────────────────
def fmt_large(val) -> str:
    """Format absolute numbers with T/B/M/K suffixes."""
    if val is None:
        return "—"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(val):
        return "—"
    sign = "-" if val < 0 else ""
    av   = abs(val)
    if av >= 1e12:  return f"{sign}{av/1e12:.2f}T"
    if av >= 1e9:   return f"{sign}{av/1e9:.2f}B"
    if av >= 1e6:   return f"{sign}{av/1e6:.2f}M"
    if av >= 1e3:   return f"{sign}{av/1e3:.1f}K"
    return f"{sign}{av:.2f}"


# ── percentages ───────────────────────────────────────────────────────────────
def fmt_pct(val, decimals: int = 1) -> str:
    """Format a 0-to-1 fraction as a percentage string."""
    if val is None:
        return "—"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(val):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def fmt_pct_chg(val, decimals: int = 1) -> str:
    """Format a year-over-year decimal change with a leading + or -."""
    if val is None:
        return "—"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(val):
        return "—"
    return f"{val * 100:+.{decimals}f}%"


# ── ratios / plain floats ─────────────────────────────────────────────────────
def fmt_ratio(val, decimals: int = 2, suffix: str = "x") -> str:
    if val is None:
        return "—"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(val):
        return "—"
    return f"{val:.{decimals}f}{suffix}"


# ── dispatch ──────────────────────────────────────────────────────────────────
_LARGE_KEYS = {
    "marketCap", "enterpriseValue", "totalRevenue", "ebitda",
    "totalCash", "totalDebt", "freeCashflow", "operatingCashflow",
}
_PCT_KEYS = {
    "profitMargins", "operatingMargins", "returnOnEquity",
    "returnOnAssets", "dividendYield", "grossMargins", "ebitdaMargins",
}
_RATIO_KEYS = {
    "trailingPE", "forwardPE", "pegRatio", "priceToSalesTrailing12Months",
    "priceToBook", "enterpriseToEbitda", "debtToEquity", "currentRatio", "beta",
}


def fmt_val(key: str, val) -> str:
    """Format a yfinance info value based on its key name."""
    if key in _LARGE_KEYS:   return fmt_large(val)
    if key in _PCT_KEYS:     return fmt_pct(val)
    if key in _RATIO_KEYS:   return fmt_ratio(val, suffix="")
    # fallback
    if val is None:
        return "—"
    try:
        v = float(val)
        return "—" if np.isnan(v) else f"{v:.2f}"
    except (TypeError, ValueError):
        return str(val)


# ── colour helpers (for Streamlit markdown) ───────────────────────────────────
def sentiment_color(score: float) -> str:
    """Map a sentiment score (-1..1) to a hex colour."""
    if score >= 0.05:   return "#4caf50"   # green
    if score <= -0.05:  return "#f44336"   # red
    return "#ffc107"                       # amber


def signal_badge(bias: str) -> str:
    """Return an HTML badge coloured by technical bias."""
    colours = {
        "Bullish":         ("#4caf50", "#fff"),
        "Leaning Bullish": ("#8bc34a", "#fff"),
        "Mixed":           ("#ffc107", "#000"),
        "Leaning Bearish": ("#ff9800", "#fff"),
        "Bearish":         ("#f44336", "#fff"),
        "Neutral":         ("#9e9e9e", "#fff"),
    }
    bg, fg = colours.get(bias, ("#9e9e9e", "#fff"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{bias}</span>'
    )
