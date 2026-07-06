"""Lag-structure regressions: does day-t sentiment predict the day-(t..t+h) forward return?"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from sentiment_signals.alignment import build_lagged_dataset


@dataclass
class LagRegressionResult:
    horizon: int
    n_obs: int
    coef: float
    tstat: float
    pvalue: float
    r_squared: float


def run_lag_regression(sentiment: pd.Series, close: pd.Series, horizon: int) -> LagRegressionResult:
    """OLS of forward return on day-t sentiment, with Newey-West (HAC) standard errors to
    account for the serial correlation induced by overlapping forward-return windows when
    horizon > 1."""
    data = build_lagged_dataset(sentiment, close, horizon)
    if len(data) < 5:
        return LagRegressionResult(horizon=horizon, n_obs=len(data), coef=float("nan"),
                                    tstat=float("nan"), pvalue=float("nan"), r_squared=float("nan"))
    X = sm.add_constant(data["sentiment"])
    y = data["fwd_ret"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon, 1)})
    return LagRegressionResult(
        horizon=horizon,
        n_obs=int(model.nobs),
        coef=float(model.params["sentiment"]),
        tstat=float(model.tvalues["sentiment"]),
        pvalue=float(model.pvalues["sentiment"]),
        r_squared=float(model.rsquared),
    )


def run_lag_regression_grid(sentiment: pd.Series, close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    rows = [run_lag_regression(sentiment, close, h).__dict__ for h in horizons]
    return pd.DataFrame(rows)
