"""Event study: abnormal return around strong-sentiment headline events, vs a market-model
expected return."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class MarketModel:
    alpha: float
    beta: float


def fit_market_model(ticker_ret: pd.Series, market_ret: pd.Series) -> MarketModel:
    data = pd.DataFrame({"y": ticker_ret, "x": market_ret}).dropna()
    X = sm.add_constant(data["x"])
    model = sm.OLS(data["y"], X).fit()
    return MarketModel(alpha=float(model.params["const"]), beta=float(model.params["x"]))


def abnormal_returns(ticker_ret: pd.Series, market_ret: pd.Series, model: MarketModel) -> pd.Series:
    """AR_t = actual_t - (alpha + beta * market_t), the market-model expected return."""
    expected = model.alpha + model.beta * market_ret
    return (ticker_ret - expected).reindex(ticker_ret.index)


def find_strong_sentiment_events(
    daily_sentiment: pd.Series, threshold: float
) -> pd.DatetimeIndex:
    """Dates where |mean daily sentiment| exceeds the given threshold."""
    return daily_sentiment.index[daily_sentiment.abs() >= threshold]


def run_event_study(
    abnormal_ret: pd.Series, event_dates: pd.DatetimeIndex, window: tuple[int, int] = (-5, 5)
) -> pd.DataFrame:
    """Average abnormal return (AAR) and cumulative AAR (CAR) at each relative trading day
    in [window[0], window[1]] around event day 0, averaged across all events.

    Relative days are trading-day offsets within abnormal_ret's index (positions), not
    calendar days, so weekends/holidays don't distort the window.
    """
    lo, hi = window
    idx = abnormal_ret.index
    pos_of = {date: i for i, date in enumerate(idx)}
    rel_days = list(range(lo, hi + 1))
    collected: dict[int, list[float]] = {r: [] for r in rel_days}

    for event_date in event_dates:
        if event_date not in pos_of:
            continue
        center = pos_of[event_date]
        for r in rel_days:
            pos = center + r
            if 0 <= pos < len(idx):
                val = abnormal_ret.iloc[pos]
                if pd.notna(val):
                    collected[r].append(val)

    rows = []
    car = 0.0
    for r in rel_days:
        vals = collected[r]
        aar = float(np.mean(vals)) if vals else float("nan")
        if not np.isnan(aar):
            car += aar
        rows.append({"rel_day": r, "aar": aar, "car": car, "n_events": len(vals)})
    return pd.DataFrame(rows)
