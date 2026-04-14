"""
utils/data.py
Cached data-fetching layer.  All functions use st.cache_data so that
repeated renders within the same session (and across users on Streamlit
Community Cloud) do not hammer the upstream APIs.
"""
from __future__ import annotations

import requests
import streamlit as st
import yfinance as yf
import pandas as pd


# ── generic TTL constants ─────────────────────────────────────────────────────
_PRICE_TTL  = 900   # 15 min — price / history
_INFO_TTL   = 3600  # 1 h  — fundamentals / metadata
_NEWS_TTL   = 1800  # 30 min — news / sentiment

# Browser-like User-Agent helps avoid Yahoo Finance IP blocks on cloud hosts
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _session() -> requests.Session:
    """Return a requests.Session with a browser User-Agent."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _ticker(symbol: str) -> yf.Ticker:
    """Return a yf.Ticker that uses a browser-spoofed session."""
    return yf.Ticker(symbol, session=_session())


# ── ticker info ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_info(ticker: str) -> dict:
    """Return yfinance .info dict; falls back to fast_info keys on error."""
    try:
        info = _ticker(ticker).info or {}
        if info:
            return info
    except Exception:
        pass

    # fast_info is a lighter endpoint — try as fallback
    try:
        fi = _ticker(ticker).fast_info
        return {k: getattr(fi, k, None) for k in fi.__dict__ if not k.startswith("_")}
    except Exception:
        return {}


# ── price history ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=_PRICE_TTL, show_spinner=False)
def get_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Return OHLCV history; empty DataFrame on error."""
    try:
        df = _ticker(ticker).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── financials ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_financials(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (income_stmt, balance_sheet); empty DataFrames on error."""
    t = _ticker(ticker)
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
        raw = _ticker(ticker).news or []
        return raw[:max_items]
    except Exception:
        return []


# ── bulk helpers ──────────────────────────────────────────────────────────────
def get_infos(tickers: list[str]) -> dict[str, dict]:
    return {t: get_info(t) for t in tickers}


def get_histories(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    return {t: get_history(t, period) for t in tickers}


# ── ownership & insider data ───────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_major_holders(ticker: str) -> pd.DataFrame:
    """Return yfinance .major_holders (insider %, institution %, # institutions)."""
    try:
        df = _ticker(ticker).major_holders
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_institutional_holders(ticker: str) -> pd.DataFrame:
    """Return yfinance .institutional_holders; empty DataFrame on error."""
    try:
        df = _ticker(ticker).institutional_holders
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_insider_transactions(ticker: str) -> pd.DataFrame:
    """Return yfinance .insider_transactions (Form 4 filings); empty DF on error."""
    try:
        df = _ticker(ticker).insider_transactions
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
