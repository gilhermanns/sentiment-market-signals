"""No-lookahead alignment between a day-t sentiment feature and forward returns/volatility.

The core design choice that makes lookahead structurally hard: callers only ever pass in
*raw*, same-day-dated series (sentiment observed on day t, close price observed on day t).
The forward shifting that pairs day-t information with day-(t+h) outcomes happens entirely
inside these functions using an explicit positive horizon -- there is no code path through
which a caller can hand in an already-forward-shifted series and have it accepted as if it
were day-t information.
"""
from __future__ import annotations

import pandas as pd


def build_lagged_dataset(sentiment: pd.Series, close: pd.Series, horizon: int) -> pd.DataFrame:
    """Pair day-t sentiment with the forward return from day t to day t+horizon.

    fwd_ret[t] = close[t+horizon] / close[t] - 1, computed by trading-day position (not
    calendar days), so it always looks strictly forward relative to sentiment[t].

    Raises ValueError if horizon is not a positive integer, which is what would be required
    to accidentally construct a same-day or backward-looking pairing.
    """
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted ascending by date")

    aligned_sentiment = sentiment.reindex(close.index)
    fwd_ret = close.shift(-horizon) / close - 1.0

    out = pd.DataFrame({"sentiment": aligned_sentiment, "fwd_ret": fwd_ret}, index=close.index)
    return out.dropna(subset=["sentiment", "fwd_ret"])


def build_lagged_datasets(sentiment: pd.Series, close: pd.Series, horizons: list[int]) -> dict[int, pd.DataFrame]:
    return {h: build_lagged_dataset(sentiment, close, h) for h in horizons}


def build_volatility_dataset(
    abs_sentiment: pd.Series, sentiment_dispersion: pd.Series, realized_vol: pd.Series
) -> pd.DataFrame:
    """Pair day-t |sentiment| and dispersion with day-(t+1) realized volatility.

    realized_vol is expected to already be the trailing/backward rolling-std series computed
    by market_data.compute_returns (same-day-available), and this function shifts it back by
    one so that the target is the *next* day's value relative to day t's sentiment features.
    """
    next_day_vol = realized_vol.shift(-1)
    idx = abs_sentiment.index
    out = pd.DataFrame(
        {
            "abs_sentiment": abs_sentiment.reindex(idx),
            "dispersion": sentiment_dispersion.reindex(idx),
            "fwd_vol": next_day_vol.reindex(idx),
        },
        index=idx,
    )
    return out.dropna(subset=["abs_sentiment", "fwd_vol"])


def daily_sentiment_aggregates(headlines: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Collapse a headlines-per-row DataFrame (columns: date, ticker, <score_col>) to one row
    per (date, ticker) with mean sentiment, |mean| sentiment, dispersion (std, 0 if n=1), and
    headline count."""
    grouped = headlines.groupby(["date", "ticker"])[score_col]
    agg = grouped.agg(mean_sentiment="mean", dispersion="std", n_headlines="count")
    agg["dispersion"] = agg["dispersion"].fillna(0.0)
    agg["abs_sentiment"] = agg["mean_sentiment"].abs()
    return agg.reset_index()
