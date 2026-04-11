"""
utils/data.py
Cached data-fetching layer.  All functions use st.cache_data so that
repeated renders within the same session (and across users on Streamlit
Community Cloud) do not hammer the upstream APIs.
"""
from __future__ import annotations

import streamlit as st
import yfinance as yf
import pandas as pd


# ── generic TTL constants ─────────────────────────────────────────────────────
_PRICE_TTL  = 900   # 15 min — price / history
_INFO_TTL   = 3600  # 1 h  — fundamentals / metadata
_NEWS_TTL   = 1800  # 30 min — news / sentiment


# ── ticker info ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_info(ticker: str) -> dict:
    """Return yfinance .info dict; empty dict on error."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


# ── price history ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=_PRICE_TTL, show_spinner=False)
def get_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Return OHLCV history; empty DataFrame on error."""
    try:
        df = yf.Ticker(ticker).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── financials ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_financials(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (income_stmt, balance_sheet); empty DataFrames on error."""
    t = yf.Ticker(ticker)
    try:
        income = t.income_stmt
        if income is None or income.empty:
            income = getattr(t, "financials", pd.DataFrame())
    except Exception:
        income = pd.DataFrame()

    try:
        balance = t.balance_sheet
        if balance is None or balance.empty:
            balance = pd.DataFrame()
    except Exception:
        balance = pd.DataFrame()

    return income, balance


# ── news ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=_NEWS_TTL, show_spinner=False)
def get_news(ticker: str, max_items: int = 20) -> list[dict]:
    """Return list of news dicts from yfinance; empty list on error."""
    try:
        raw = yf.Ticker(ticker).news or []
        return raw[:max_items]
    except Exception:
        return []


# ── bulk helpers ──────────────────────────────────────────────────────────────
def get_infos(tickers: list[str]) -> dict[str, dict]:
    return {t: get_info(t) for t in tickers}


def get_histories(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    return {t: get_history(t, period) for t in tickers}
