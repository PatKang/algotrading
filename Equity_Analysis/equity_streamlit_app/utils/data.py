"""
utils/data.py
Cached data-fetching layer.  All functions use st.cache_data so that
repeated renders within the same session (and across users on Streamlit
Community Cloud) do not hammer the upstream APIs.

Primary source for all data: Financial Modeling Prep (FMP) stable API.
Fallback: yfinance.
"""
from __future__ import annotations

import os
from datetime import date, timedelta, datetime, timezone

import requests
import streamlit as st
import yfinance as yf
import pandas as pd


# ── generic TTL constants ─────────────────────────────────────────────────────
_PRICE_TTL  = 900   # 15 min — price / history
_INFO_TTL   = 3600  # 1 h  — fundamentals / metadata
_NEWS_TTL   = 1800  # 30 min — news / sentiment

_FMP_STABLE = "https://financialmodelingprep.com/stable"


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


def _fmp_get(path: str, params: dict, timeout: int = 10) -> list | dict | None:
    """GET a FMP stable endpoint; returns parsed JSON or None on any error."""
    api_key = _get_fmp_key()
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"{_FMP_STABLE}/{path}",
            params={**params, "apikey": api_key},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ── FMP: info ─────────────────────────────────────────────────────────────────
def _fmp_get_info(ticker: str) -> dict | None:
    """Fetch fundamentals from FMP stable API and map to yfinance field names.

    Combines profile + ratios-ttm + key-metrics-ttm + income/balance statements.
    Returns a dict on (partial or full) success, None if profile call fails.
    """
    result: dict = {}

    # ── profile ───────────────────────────────────────────────────────────────
    profiles = _fmp_get("profile", {"symbol": ticker})
    if not profiles or not isinstance(profiles, list):
        return None
    p = profiles[0]

    # Parse 52-week range string "low-high"
    raw_range = str(p.get("range", "") or "")
    week52_low = week52_high = None
    if "-" in raw_range:
        parts = raw_range.replace(" ", "").split("-")
        try:
            week52_low  = float(parts[0])
            week52_high = float(parts[1])
        except (ValueError, IndexError):
            pass

    price  = p.get("price")
    change = p.get("change")
    prev_close = (price - change) if price is not None and change is not None else None

    result.update({
        "marketCap":                  p.get("marketCap"),
        "beta":                       p.get("beta"),
        "sector":                     p.get("sector"),
        "industry":                   p.get("industry"),
        "longName":                   p.get("companyName"),
        "website":                    p.get("website"),
        "longBusinessSummary":        p.get("description"),
        "currentPrice":               price,
        "currency":                   p.get("currency"),
        "exchange":                   p.get("exchange"),
        "fiftyTwoWeekHigh":           week52_high,
        "fiftyTwoWeekLow":            week52_low,
        "regularMarketPreviousClose": prev_close,
    })

    # ── ratios-ttm ────────────────────────────────────────────────────────────
    ratios = _fmp_get("ratios-ttm", {"symbol": ticker})
    if ratios and isinstance(ratios, list):
        r = ratios[0]

        # stable debtToEquityRatioTTM is a ratio (e.g. 0.80); yfinance uses
        # percentage (e.g. 80.0), so multiply by 100 for display consistency.
        dte = r.get("debtToEquityRatioTTM")
        if dte is not None:
            try:
                dte = float(dte) * 100
            except (TypeError, ValueError):
                dte = None

        result.update({
            "trailingPE":                   r.get("priceToEarningsRatioTTM"),
            "pegRatio":                     r.get("priceToEarningsGrowthRatioTTM"),
            "priceToSalesTrailing12Months": r.get("priceToSalesRatioTTM"),
            "priceToBook":                  r.get("priceToBookRatioTTM"),
            "enterpriseValue":              r.get("enterpriseValueTTM"),
            "enterpriseToEbitda":           r.get("enterpriseValueMultipleTTM"),
            "grossMargins":                 r.get("grossProfitMarginTTM"),
            "operatingMargins":             r.get("operatingProfitMarginTTM"),
            "profitMargins":                r.get("netProfitMarginTTM"),
            "currentRatio":                 r.get("currentRatioTTM"),
            # stable dividendYieldTTM is already a 0-to-1 decimal (e.g. 0.0037
            # for a 0.37% yield) — pass directly to fmt_pct, no division needed.
            "dividendYield":                r.get("dividendYieldTTM"),
            "debtToEquity":                 dte,
        })

    # ── key-metrics-ttm — ROE, ROA ────────────────────────────────────────────
    km = _fmp_get("key-metrics-ttm", {"symbol": ticker})
    if km and isinstance(km, list):
        m = km[0]
        result.update({
            "returnOnEquity": m.get("returnOnEquityTTM"),
            "returnOnAssets": m.get("returnOnAssetsTTM"),
        })

    # ── income-statement — totalRevenue, ebitda ───────────────────────────────
    inc = _fmp_get("income-statement", {"symbol": ticker, "limit": 1})
    if inc and isinstance(inc, list):
        result["totalRevenue"] = inc[0].get("revenue")
        result["ebitda"]       = inc[0].get("ebitda")

    # ── balance-sheet — totalCash, totalDebt ─────────────────────────────────
    bs = _fmp_get("balance-sheet-statement", {"symbol": ticker, "limit": 1})
    if bs and isinstance(bs, list):
        result["totalCash"] = bs[0].get("cashAndShortTermInvestments")
        result["totalDebt"] = bs[0].get("totalDebt")

    return result if result else None


# ── FMP: financials ───────────────────────────────────────────────────────────
def _fmp_get_financials(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch income-statement and balance-sheet from FMP stable API.

    Returns DataFrames whose index/column structure mirrors yfinance output
    (metric-name index, Timestamp columns), or None on failure.
    """
    # ── income statement ──────────────────────────────────────────────────────
    income_json = _fmp_get("income-statement", {"symbol": ticker, "limit": 5})
    if not income_json or not isinstance(income_json, list):
        return None

    _INCOME_MAP: dict[str, str] = {
        "Total Revenue":    "revenue",
        "Gross Profit":     "grossProfit",
        "Operating Income": "operatingIncome",
        "Net Income":       "netIncome",
        "EBITDA":           "ebitda",
        "Diluted EPS":      "epsDiluted",
        "Basic EPS":        "eps",
    }

    income_data: dict[str, dict] = {label: {} for label in _INCOME_MAP}
    for record in income_json:
        try:
            dt = pd.Timestamp(record["date"])
        except Exception:
            continue
        for label, fmp_key in _INCOME_MAP.items():
            income_data[label][dt] = record.get(fmp_key)

    income_df = pd.DataFrame(income_data).T
    if not income_df.empty:
        income_df.columns = pd.to_datetime(income_df.columns)

    # ── balance sheet ─────────────────────────────────────────────────────────
    balance_df = pd.DataFrame()
    balance_json = _fmp_get("balance-sheet-statement", {"symbol": ticker, "limit": 5})
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
                dt = pd.Timestamp(record["date"])
            except Exception:
                continue
            for label, fmp_key in _BALANCE_MAP.items():
                balance_data[label][dt] = record.get(fmp_key)

        balance_df = pd.DataFrame(balance_data).T
        if not balance_df.empty:
            balance_df.columns = pd.to_datetime(balance_df.columns)

    return income_df, balance_df


# ── FMP: price history ────────────────────────────────────────────────────────
def _period_to_from_date(period: str) -> str | None:
    """Map a yfinance-style period string to an ISO date for FMP's 'from' param."""
    mapping = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
    }
    if period == "max":
        return None
    if period == "ytd":
        return date(date.today().year, 1, 1).strftime("%Y-%m-%d")
    days = mapping.get(period, 730)
    return (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def _fmp_get_history(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Fetch daily OHLCV from FMP stable historical-price-eod/full endpoint.

    Returns a DataFrame with the same column structure as yfinance
    (Open, High, Low, Close, Volume; DatetimeIndex), or None on failure.
    """
    params: dict = {"symbol": ticker}
    from_date = _period_to_from_date(period)
    if from_date:
        params["from"] = from_date
        params["to"]   = date.today().strftime("%Y-%m-%d")

    data = _fmp_get("historical-price-eod/full", params, timeout=15)
    # stable endpoint returns a plain list (not wrapped in {"historical": [...]})
    if not data or not isinstance(data, list):
        return None

    df = pd.DataFrame(data)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    })
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[cols] if cols else None


# ── FMP: news ─────────────────────────────────────────────────────────────────
def _fmp_get_news(ticker: str, max_items: int = 20) -> list[dict] | None:
    """Fetch news from FMP stable news/stock endpoint.

    Returns a list of dicts compatible with yfinance news format, or None.
    """
    articles = _fmp_get("news/stock", {"symbol": ticker, "limit": max_items})
    if not articles or not isinstance(articles, list):
        return None

    result = []
    for art in articles:
        pub_ts = None
        pub_str = art.get("publishedDate", "")
        if pub_str:
            try:
                dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                pub_ts = int(dt.timestamp())
            except Exception:
                pass

        url = art.get("url") or art.get("link") or ""
        result.append({
            "title":               art.get("title", ""),
            "link":                url,
            "url":                 url,
            "publisher":           art.get("site") or art.get("publisher") or "",
            "providerPublishTime": pub_ts,
        })

    return result or None


# ── FMP: institutional holders ────────────────────────────────────────────────
def _fmp_get_institutional_holders(ticker: str) -> pd.DataFrame | None:
    """Fetch top institutional holders from FMP stable API."""
    data = _fmp_get(
        "institutional-ownership/institutional-holders/symbol-ownership",
        {"symbol": ticker},
    )
    if not data or not isinstance(data, list):
        return None

    rows = [
        {
            "Holder":        h.get("holder", ""),
            "Shares":        h.get("shares"),
            "Date Reported": h.get("dateReported", ""),
            "% Out":         h.get("weightPercent"),
        }
        for h in data
    ]
    df = pd.DataFrame(rows)
    return df if not df.empty else None


# ── FMP: insider transactions ─────────────────────────────────────────────────
def _fmp_get_insider_transactions(ticker: str, limit: int = 50) -> pd.DataFrame | None:
    """Fetch insider trading records from FMP stable API."""
    data = _fmp_get("insider-trading", {"symbol": ticker, "limit": limit})
    if not data or not isinstance(data, list):
        return None

    rows = [
        {
            "Insider":     rec.get("reportingName", ""),
            "Position":    rec.get("typeOfOwner", ""),
            "Transaction": rec.get("transactionType", ""),
            "Shares":      rec.get("securitiesTransacted"),
            "Value":       rec.get("price"),
            "Date":        rec.get("transactionDate", ""),
            "URL":         rec.get("link", ""),
        }
        for rec in data
    ]
    df = pd.DataFrame(rows)
    return df if not df.empty else None


# ── yfinance ticker helper ────────────────────────────────────────────────────
def _ticker(symbol: str) -> yf.Ticker:
    """Return a yf.Ticker; let yfinance manage its own session/cookies."""
    return yf.Ticker(symbol)


# ── ticker info ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_info(ticker: str) -> dict:
    """Return fundamentals dict; tries FMP first, falls back to yfinance."""
    try:
        fmp_info = _fmp_get_info(ticker)
        if fmp_info:
            return fmp_info
    except Exception:
        pass

    try:
        info = _ticker(ticker).info or {}
        if info:
            return info
    except Exception as e:
        st.warning(
            f"⚠️ Could not fetch info for **{ticker}** "
            f"({type(e).__name__}: {e}). Yahoo Finance may be blocking this environment."
        )

    try:
        fi = _ticker(ticker).fast_info
        return {k: getattr(fi, k, None) for k in fi.__dict__ if not k.startswith("_")}
    except Exception:
        return {}


# ── price history ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=_PRICE_TTL, show_spinner=False)
def get_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Return OHLCV history; tries FMP first, falls back to yfinance."""
    try:
        df = _fmp_get_history(ticker, period)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        df = _ticker(ticker).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── financials ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_financials(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (income_stmt, balance_sheet); tries FMP first, falls back to yfinance."""
    try:
        fmp_result = _fmp_get_financials(ticker)
        if fmp_result is not None:
            income, balance = fmp_result
            if not income.empty:
                return income, balance
    except Exception:
        pass

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
    """Return list of news dicts; tries FMP first, falls back to yfinance."""
    try:
        fmp_news = _fmp_get_news(ticker, max_items)
        if fmp_news:
            return fmp_news
    except Exception:
        pass

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
    """Return institutional holders; tries FMP first, falls back to yfinance."""
    try:
        df = _fmp_get_institutional_holders(ticker)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        df = _ticker(ticker).institutional_holders
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_INFO_TTL, show_spinner=False)
def get_insider_transactions(ticker: str) -> pd.DataFrame:
    """Return insider transactions (Form 4); tries FMP first, falls back to yfinance."""
    try:
        df = _fmp_get_insider_transactions(ticker)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        df = _ticker(ticker).insider_transactions
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
