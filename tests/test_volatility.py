import numpy as np
import pandas as pd
import pytest

from sentiment_signals.volatility import run_volatility_regression


def test_volatility_regression_recovers_known_relationship():
    rng = np.random.RandomState(0)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    abs_sentiment = pd.Series(rng.uniform(0, 1, n), index=dates)
    dispersion = pd.Series(rng.uniform(0, 0.5, n), index=dates)
    realized_vol_next = 0.01 + 0.02 * abs_sentiment + 0.01 * dispersion + rng.normal(0, 0.001, n)
    # build_volatility_dataset shifts realized_vol back by one day to form the target, so
    # realized_vol here must be realized_vol_next shifted *forward* by one day.
    realized_vol = realized_vol_next.shift(1).fillna(0.02)

    result = run_volatility_regression(abs_sentiment, dispersion, realized_vol)
    assert result.coef_abs_sentiment == pytest.approx(0.02, rel=0.3)
    assert result.tstat_abs_sentiment > 3


def test_volatility_regression_on_unrelated_series_shows_weak_signal():
    rng = np.random.RandomState(5)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    abs_sentiment = pd.Series(rng.uniform(0, 1, n), index=dates)
    dispersion = pd.Series(rng.uniform(0, 0.5, n), index=dates)
    realized_vol = pd.Series(rng.uniform(0.005, 0.03, n), index=dates)
    result = run_volatility_regression(abs_sentiment, dispersion, realized_vol)
    assert abs(result.tstat_abs_sentiment) < 4
