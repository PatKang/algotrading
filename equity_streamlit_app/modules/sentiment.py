"""
modules/sentiment.py
Module 3 — Headline Sentiment
  • Fetches recent news from yfinance
  • Scores every headline with VADER
  • Shows per-ticker headline table + aggregate comparison chart
  • Optional: Alpha Vantage enriched sentiment (if ALPHA_VANTAGE_KEY is set)
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get_news
from utils.formatting import sentiment_color

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_OK = True
except ImportError:
    _VADER_OK = False

# ── VADER singleton ───────────────────────────────────────────────────────────
@st.cache_resource
def _get_vader():
    if not _VADER_OK:
        return None
    return SentimentIntensityAnalyzer()


# ── Alpha Vantage (optional) ──────────────────────────────────────────────────
def _get_av_key() -> str | None:
    """Read AV key from Streamlit secrets, then from env, then return None."""
    try:
        key = st.secrets.get("ALPHA_VANTAGE_KEY")
        if key:
            return key
    except Exception:
        pass
    import os
    return os.getenv("ALPHA_VANTAGE_KEY")


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_av_news(tickers_tuple: tuple[str, ...], limit_per_ticker: int = 30) -> list[dict]:
    """Fetch AV news sentiment (1 req/ticker, rate-limited to 1/s)."""
    import requests

    key = _get_av_key()
    if not key:
        return []

    seen_urls: set[str] = set()
    combined: list[dict] = []

    for i, tkr in enumerate(tickers_tuple):
        if i > 0:
            time.sleep(1.2)
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers":  tkr,
            "limit":    limit_per_ticker,
            "apikey":   key,
        }
        try:
            resp = requests.get("https://www.alphavantage.co/query",
                                params=params, timeout=15).json()
        except Exception:
            continue

        if "feed" not in resp:
            continue

        for article in resp["feed"]:
            url_key = article.get("url", article.get("title", ""))
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                combined.append(article)

    return combined


# ── headline scoring ──────────────────────────────────────────────────────────
def _score_headlines(tickers: list[str]) -> pd.DataFrame:
    """Fetch headlines from yfinance and score with VADER."""
    vader = _get_vader()
    rows: list[dict] = []

    for tkr in tickers:
        articles = get_news(tkr, max_items=20)
        for art in articles:
            title = art.get("title", "")
            if not title:
                continue

            ts = art.get("providerPublishTime")
            pub_str = (
                datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                if ts else "—"
            )

            if vader:
                sc = vader.polarity_scores(title)
                compound = sc["compound"]
                pos, neg, neu = sc["pos"], sc["neg"], sc["neu"]
            else:
                compound = pos = neg = neu = None

            sentiment_label = (
                "Positive" if compound is not None and compound >= 0.05
                else "Negative" if compound is not None and compound <= -0.05
                else "Neutral"
            )

            rows.append({
                "Ticker":    tkr,
                "Published": pub_str,
                "Headline":  title,
                "Source":    art.get("publisher", ""),
                "Compound":  round(compound, 3) if compound is not None else None,
                "Positive":  round(pos, 3)      if pos      is not None else None,
                "Negative":  round(neg, 3)      if neg      is not None else None,
                "Neutral":   round(neu, 3)       if neu      is not None else None,
                "Sentiment": sentiment_label,
                "URL":       art.get("link", ""),
            })

    return pd.DataFrame(rows)


# ── aggregate chart ───────────────────────────────────────────────────────────
def _sentiment_bar_chart(df: pd.DataFrame) -> go.Figure:
    agg = (
        df.groupby("Ticker")
        .agg(
            Count    =("Compound", "count"),
            Mean     =("Compound", "mean"),
            Positive =("Sentiment", lambda s: (s == "Positive").sum()),
            Negative =("Sentiment", lambda s: (s == "Negative").sum()),
            Neutral  =("Sentiment", lambda s: (s == "Neutral").sum()),
        )
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive", x=agg["Ticker"], y=agg["Positive"],
        marker_color="#4caf50",
    ))
    fig.add_trace(go.Bar(
        name="Neutral", x=agg["Ticker"], y=agg["Neutral"],
        marker_color="#ffc107",
    ))
    fig.add_trace(go.Bar(
        name="Negative", x=agg["Ticker"], y=agg["Negative"],
        marker_color="#f44336",
    ))
    fig.update_layout(
        barmode="stack",
        title="Headline Sentiment Distribution by Ticker",
        yaxis_title="Number of Headlines",
        height=380,
        template="plotly_dark",
        margin=dict(t=50, b=40, l=50, r=20),
        legend_title="Sentiment",
    )
    return fig


def _mean_score_chart(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("Ticker")["Compound"].mean().reset_index()
    colors = [
        "#4caf50" if v >= 0.05 else "#f44336" if v <= -0.05 else "#ffc107"
        for v in agg["Compound"]
    ]
    fig = go.Figure(go.Bar(
        x=agg["Ticker"], y=agg["Compound"],
        marker_color=colors,
        text=[f"{v:+.3f}" for v in agg["Compound"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Mean VADER Compound Score by Ticker",
        yaxis_title="Mean Compound Score",
        height=360,
        template="plotly_dark",
        margin=dict(t=50, b=40, l=50, r=20),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_yaxes(range=[-1, 1])
    return fig


# ── AV summary section ────────────────────────────────────────────────────────
def _render_av_section(tickers: list[str]) -> None:
    st.subheader("Alpha Vantage — Enriched Sentiment")

    av_key = _get_av_key()
    if not av_key:
        st.info(
            "No Alpha Vantage key found. Add `ALPHA_VANTAGE_KEY` to your "
            "[Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/"
            "deploy-your-app/secrets-management) or a local `.env` file to enable "
            "this enriched sentiment section."
        )
        return

    with st.spinner("Fetching Alpha Vantage news (rate-limited — ~1 s/ticker)…"):
        feed = _fetch_av_news(tuple(tickers))

    if not feed:
        st.warning("No articles returned from Alpha Vantage.")
        return

    LABEL_ORDER = ["Bullish", "Somewhat-Bullish", "Neutral", "Somewhat-Bearish", "Bearish"]

    rows = []
    for tkr in tickers:
        relevant = []
        for article in feed:
            for ts in article.get("ticker_sentiment", []):
                if ts["ticker"] != tkr:
                    continue
                if float(ts.get("relevance_score", 0)) <= 0.3:
                    continue
                try:
                    pub = datetime.strptime(
                        article["time_published"], "%Y%m%dT%H%M%S"
                    )
                except Exception:
                    pub = None
                relevant.append({
                    "label":     ts["ticker_sentiment_label"],
                    "score":     float(ts["ticker_sentiment_score"]),
                    "relevance": float(ts["relevance_score"]),
                    "published": pub,
                })

        if not relevant:
            rows.append({
                "Ticker": tkr, "Verdict": "No data",
                "Bull": 0, "Bear": 0, "Neutral_AV": 0,
                "Avg Score": None, "Articles": 0,
            })
            continue

        lc = {l: 0 for l in LABEL_ORDER}
        for a in relevant:
            if a["label"] in lc:
                lc[a["label"]] += 1

        bull = lc["Bullish"] + lc["Somewhat-Bullish"]
        bear = lc["Bearish"] + lc["Somewhat-Bearish"]
        avg  = np.mean([a["score"] for a in relevant])

        rows.append({
            "Ticker":    tkr,
            "Verdict":   "Bullish" if bull > bear else "Bearish" if bear > bull else "Neutral",
            "Bull":      bull,
            "Bear":      bear,
            "Neutral_AV": lc["Neutral"],
            "Avg Score": round(avg, 3),
            "Articles":  len(relevant),
        })

    av_df = pd.DataFrame(rows).set_index("Ticker")
    av_df.rename(columns={"Neutral_AV": "Neutral"}, inplace=True)
    st.dataframe(av_df, use_container_width=True)

    st.caption(
        f"Source: Alpha Vantage NEWS_SENTIMENT · {len(feed)} unique articles fetched · "
        "relevance threshold > 0.30"
    )


# ── public render function ────────────────────────────────────────────────────
def render_sentiment(tickers: list[str]) -> None:
    """Entry point called from app.py for the Sentiment tab."""
    if not tickers:
        st.info("Add tickers in the sidebar to get started.")
        return

    if not _VADER_OK:
        st.error(
            "`vaderSentiment` is not installed. "
            "Run `pip install vaderSentiment` and restart the app."
        )
        return

    # ── fetch & score ─────────────────────────────────────────────────────────
    with st.spinner("Fetching and scoring headlines…"):
        df = _score_headlines(tickers)

    if df.empty:
        st.warning("No headlines could be retrieved. Try again in a moment.")
        return

    # ── aggregate charts ──────────────────────────────────────────────────────
    st.subheader("Sentiment Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_sentiment_bar_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(_mean_score_chart(df), use_container_width=True)

    # ── aggregate table ───────────────────────────────────────────────────────
    agg = (
        df.groupby("Ticker")
        .agg(
            Headlines    =("Compound", "count"),
            Mean_Score   =("Compound", "mean"),
            Median_Score =("Compound", "median"),
            Positive_Pct =("Sentiment", lambda s: f"{(s=='Positive').mean():.0%}"),
            Negative_Pct =("Sentiment", lambda s: f"{(s=='Negative').mean():.0%}"),
        )
        .round(3)
    )
    st.dataframe(agg, use_container_width=True)

    st.divider()

    # ── per-ticker headline table ─────────────────────────────────────────────
    st.subheader("Headlines by Ticker")

    filter_tkr = st.selectbox(
        "Filter by ticker (or All):",
        ["All"] + tickers,
        key="sent_filter",
    )
    filter_sent = st.multiselect(
        "Filter by sentiment:",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
        key="sent_sentiment_filter",
    )

    view_df = df.copy()
    if filter_tkr != "All":
        view_df = view_df[view_df["Ticker"] == filter_tkr]
    if filter_sent:
        view_df = view_df[view_df["Sentiment"].isin(filter_sent)]

    # make headlines clickable
    def _make_link(row):
        url = row.get("URL", "")
        headline = row.get("Headline", "")
        if url:
            return f'<a href="{url}" target="_blank">{headline}</a>'
        return headline

    display_df = view_df[
        ["Ticker", "Published", "Source", "Headline", "Compound", "Sentiment"]
    ].copy()
    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Compound": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.caption(
        f"Showing {len(display_df)} of {len(df)} headlines · "
        "Scored with VADER (compound: +1 = very positive, −1 = very negative)"
    )

    st.divider()

    # ── Alpha Vantage (optional) ───────────────────────────────────────────────
    _render_av_section(tickers)
