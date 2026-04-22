"""
utils/data.py
Cached data-fetching layer.  All functions use st.cache_data so that
repeated renders within the same session (and across users on Streamlit
Community Cloud) do not hammer the upstream APIs.

Primary source for fundamentals: Financial Modeling Prep (FMP).
Fallback: yfinance.
"""
from __future__ import annotations

import os

import requests
import streamlit as st
import yfinance as yf
import pandas as pd


# ── generic TTL constants ─────────────────────────────────────────────────────
_PRICE_TTL  = 900   # 15 min — price / history
_INFO_TTL   = 3600  # 1 h  — fundamentals / metadata
_NEWS_TTL   = 1800  # 30 min — news / sentiment

_FMP_BASE = "https://financialmodelingprep.com/api/v3"


# ── secret helpers ────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str | None:
    """Check st.secrets first, then os.getenv. Returns None if not found."""
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        pass
    return os.getenv(key)


def _get_fmp_key() -> str | None:
    return _get_secret("FMP_KEY")


# ── FMP: info ─────────────────────────────────────────────────────────────────
def _fmp_get_info(ticker: str) -> dict | None:
    """Fetch profile + key-metrics-ttm from FMP and map to yfinance field names.

    Returns a dict on (partial or full) success, None if the API key is
    missing or the profile call itself fails.
    """
    api_key = _get_fmp_key()
    if not api_key:
        return None

    result: dict = {}

    # ── profile ───────────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{_FMP_BASE}/profile/{ticker}",
            params={"apikey": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        profiles = resp.json()
        if not profiles or not isinstance(profiles, list):
            return None
        p = profiles[0]

        # Parse 52-week range string "low-high" (both sides are positive floats)
        raw_range = str(p.get("range", "") or "")
        if "-" in raw_range:
            parts = raw_range.split("-", 1)
            try:
                week52_low  = float(parts[0].strip())
                week52_high = float(parts[1].strip())
            except (ValueError, IndexError):
                week52_low = week52_high = None
        else:
            week52_low = week52_high = None

        result.update({
            "marketCap":             p.get("mktCap"),
            "beta":                  p.get("beta"),
            "sector":                p.get("sector"),
            "industry":              p.get("industry"),
            "longName":              p.get("companyName"),
            "website":               p.get("website"),
            "longBusinessSummary":   p.get("description"),
            "currentPrice":          p.get("price"),
            "currency":              p.get("currency"),
            "exchange":              p.get("exchangeShortName"),
            "fiftyTwoWeekHigh":             week52_high,
            "fiftyTwoWeekLow":              week52_low,
            "regularMarketPreviousClose":   p.get("previousClose"),
        })
    except Exception:
        return None

    # ── key-metrics-ttm ───────────────────────────────────────────────────────
    # A failure here still returns the profile data gathered above.
    try:
        resp = requests.get(
            f"{_FMP_BASE}/key-metrics-ttm/{ticker}",
            params={"apikey": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        metrics = resp.json()
        if metrics and isinstance(metrics, list):
            m = metrics[0]

            # FMP debtToEquityTTM is a ratio (e.g. 1.5); yfinance reports it
            # as a percentage (e.g. 150.0), so multiply by 100 for consistency.
            dte = m.get("debtToEquityTTM")
            if dte is not None:
                try:
                    dte = float(dte) * 100
                except (TypeError, ValueError):
                    dte = None

            result.update({
                "enterpriseValue":              m.get("enterpriseValueTTM"),
                "trailingPE":                   m.get("peRatioTTM"),
                "forwardPE":                    m.get("forwardPeTTM"),
                "pegRatio":                     m.get("pegRatioTTM"),
                "priceToSalesTrailing12Months":  m.get("priceToSalesRatioTTM"),
                "priceToBook":                  m.get("pbRatioTTM"),
                "enterpriseToEbitda":           m.get("enterpriseValueOverEBITDATTM"),
                "profitMargins":                m.get("netProfitMarginTTM"),
                "operatingMargins":             m.get("operatingProfitMarginTTM"),
                "grossMargins":                 m.get("grossProfitMarginTTM"),
                "returnOnEquity":               m.get("roeTTM"),
                "returnOnAssets":               m.get("roaTTM"),
                "debtToEquity":                 dte,
                "currentRatio":                 m.get("currentRatioTTM"),
                # FMP dividendYieldTTM is in percentage (e.g. 2.55); fmt_pct()
                # expects a 0-to-1 fraction, so divide by 100.
                "dividendYield": (
                    m["dividendYieldTTM"] / 100
                    if m.get("dividendYieldTTM") is not None else None
                ),
                "ebitda":                       m.get("ebitdaTTM"),
            })
    except Exception:
        pass  # return partial profile data; caller will use it

    # ── income-statement (latest record) — totalRevenue ───────────────────────
    try:
        resp = requests.get(
            f"{_FMP_BASE}/income-statement/{ticker}",
            params={"apikey": api_key, "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        inc = resp.json()
        if inc and isinstance(inc, list):
            result["totalRevenue"] = inc[0].get("revenue")
    except Exception:
        pass

    # ── balance-sheet (latest record) — totalCash, totalDebt ─────────────────
    try:
        resp = requests.get(
            f"{_FMP_BASE}/balance-sheet-statement/{ticker}",
            params={"apikey": api_key, "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        bs = resp.json()
        if bs and isinstance(bs, list):
            result["totalCash"] = bs[0].get("cashAndShortTermInvestments")
            result["totalDebt"] = bs[0].get("totalDebt")
    except Exception:
        pass

    return result if result else None


# ── FMP: financials ───────────────────────────────────────────────────────────
def _fmp_get_financials(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch income-statement and balance-sheet from FMP.

    Returns DataFrames whose index/column structure mirrors yfinance output
    (metric-name index, Timestamp columns), or None on failure.
    """
    api_key = _get_fmp_key()
    if not api_key:
        return None

    # ── income statement ──────────────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{_FMP_BASE}/income-statement/{ticker}",
            params={"apikey": api_key, "limit": 5},
            timeout=10,
        )
        resp.raise_for_status()
        income_json = resp.json()
        if not income_json or not isinstance(income_json, list):
            return None
    except Exception:
        return None

    # Mapping: yfinance row-name → FMP field key
    _INCOME_MAP: dict[str, str] = {
        "Total Revenue":    "revenue",
        "Gross Profit":     "grossProfit",
        "Operating Income": "operatingIncome",
        "Net Income":       "netIncome",
        "EBITDA":           "ebitda",
        "Diluted EPS":      "epsdiluted",
        "Basic EPS":        "eps",
    }

    income_data: dict[str, dict] = {label: {} for label in _INCOME_MAP}
    for record in income_json:
        try:
            date = pd.Timestamp(record["date"])
        except Exception:
            continue
        for label, fmp_key in _INCOME_MAP.items():
            income_data[label][date] = record.get(fmp_key)

    income_df = pd.DataFrame(income_data).T
    if not income_df.empty:
        income_df.columns = pd.to_datetime(income_df.columns)

    # ── balance sheet ─────────────────────────────────────────────────────────
    balance_df = pd.DataFrame()
    try:
        resp = requests.get(
            f"{_FMP_BASE}/balance-sheet-statement/{ticker}",
            params={"apikey": api_key, "limit": 5},
            timeout=10,
        )
        resp.raise_for_status()
        balance_json = resp.json()
        if balance_json and isinstance(balance_json, list):
            _BALANCE_MAP: dict[str, str] = {
                "Cash Cash Equivalents And Short Term Investments": "cashAndShortTermInvestments",
                "Cash And Cash Equivalents":                        "cashAndCashEquivalents",
                "Total Debt":                                       "totalDebt",
                "Long Term Debt And Capital Lease Obligation":      "longTermDebt",
            }

            balance_data: dict[str, dict] = {label: {} for label in _BALANCE_MAP}
            for record in balance_json:
                try:
                    date = pd.Timestamp(record["date"])
                except Exception:
                    continue
                for label, fmp_key in _BALANCE_MAP.items():
                    balance_data[label][date] = record.get(fmp_key)

            balance_df = pd.DataFrame(balance_data).T
            if not balance_df.empty:
                balance_df.columns = pd.to_datetime(balance_df.columns)
    except Exception:
        pass  # income data is sufficient; balance is best-effort

    return income_df, balance_df


# ── yfinance ticker helper ────────────────────────────────────────────────────
def _ticker(symbol: str) -> yf.Ticker:
    """Return a yf.Ticker; let yfinance manage its own session/cookies."""
    return yf.Ticker(symbol)


# ── ticker info ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_info(ticker: str) -> dict:
    """Return fundamentals dict; tries FMP first, falls back to yfinance."""
    # Try FMP first
    try:
        fmp_info = _fmp_get_info(ticker)
        if fmp_info:
            return fmp_info
    except Exception:
        pass

    # Fall back to yfinance
    try:
        info = _ticker(ticker).info or {}
        if info:
            return info
    except Exception as e:
        st.warning(
            f"⚠️ Could not fetch info for **{ticker}** "
            f"({type(e).__name__}: {e}). Yahoo Finance may be blocking this environment."
        )

    # fast_info is a lighter endpoint — try as last resort
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
    """Return (income_stmt, balance_sheet); tries FMP first, falls back to yfinance."""
    # Try FMP first
    try:
        fmp_result = _fmp_get_financials(ticker)
        if fmp_result is not None:
            income, balance = fmp_result
            if not income.empty:
                return income, balance
    except Exception:
        pass

    # Fall back to yfinance
    t = _ticker(ticker)
    try:
        income = t.income_stmt
        if income is None or income.empty:
            income = getattr(t, "financials", pd.DataFrame())
    except Exception as e:
        st.warning(f"⚠️ Could not fetch income statement for **{ticker}** ({type(e).__name__}: {e}).")
        income = pd.DataFrame()

    try:
        balance = t.balance_sheet
        if balance is None or balance.empty:
            balance = pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Could not fetch balance sheet for **{ticker}** ({type(e).__name__}: {e}).")
        balance = pd.DataFrame()

    return income, balance


# ── news ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=_NEWS_TTL, show_spinner=False)
def get_news(ticker: str, max_items: int = 20) -> list[dict]:
    """Return list of news dicts from yfinance; empty list on error."""
    try:
        raw = _ticker(ticker).news or []
        return raw[:max_items]
    except Exception as e:
        st.warning(f"⚠️ Could not fetch news for **{ticker}** ({type(e).__name__}: {e}).")
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
