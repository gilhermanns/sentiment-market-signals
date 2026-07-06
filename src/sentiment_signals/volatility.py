"""Does |sentiment| or same-day sentiment dispersion predict next-day realized volatility?"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from sentiment_signals.alignment import build_volatility_dataset


@dataclass
class VolRegressionResult:
    n_obs: int
    coef_abs_sentiment: float
    tstat_abs_sentiment: float
    pvalue_abs_sentiment: float
    coef_dispersion: float
    tstat_dispersion: float
    pvalue_dispersion: float
    r_squared: float


def run_volatility_regression(
    abs_sentiment: pd.Series, dispersion: pd.Series, realized_vol: pd.Series
) -> VolRegressionResult:
    data = build_volatility_dataset(abs_sentiment, dispersion, realized_vol)
    if len(data) < 5:
        nan = float("nan")
        return VolRegressionResult(len(data), nan, nan, nan, nan, nan, nan, nan)
    X = sm.add_constant(data[["abs_sentiment", "dispersion"]])
    y = data["fwd_vol"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return VolRegressionResult(
        n_obs=int(model.nobs),
        coef_abs_sentiment=float(model.params["abs_sentiment"]),
        tstat_abs_sentiment=float(model.tvalues["abs_sentiment"]),
        pvalue_abs_sentiment=float(model.pvalues["abs_sentiment"]),
        coef_dispersion=float(model.params["dispersion"]),
        tstat_dispersion=float(model.tvalues["dispersion"]),
        pvalue_dispersion=float(model.pvalues["dispersion"]),
        r_squared=float(model.rsquared),
    )
