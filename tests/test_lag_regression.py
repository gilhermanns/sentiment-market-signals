import numpy as np
import pandas as pd
import pytest

from sentiment_signals.lag_regression import run_lag_regression, run_lag_regression_grid


def test_regression_recovers_known_relationship_with_significant_tstat():
    rng = np.random.RandomState(0)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    sentiment = pd.Series(rng.normal(0, 1, n), index=dates)
    # construct close prices whose day-to-day return is directly driven by sentiment
    daily_ret = 0.01 * sentiment.shift(1).fillna(0) + rng.normal(0, 0.0005, n)
    close = pd.Series(100 * (1 + daily_ret).cumprod().values, index=dates)

    result = run_lag_regression(sentiment, close, horizon=1)
    assert result.coef == pytest.approx(0.01, rel=0.3)
    assert abs(result.tstat) > 3  # should be clearly statistically significant


def test_regression_on_pure_noise_gives_no_reliable_signal():
    rng = np.random.RandomState(1)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    sentiment = pd.Series(rng.normal(0, 1, n), index=dates)
    close = pd.Series(100 * (1 + rng.normal(0, 0.01, n)).cumprod(), index=dates)
    result = run_lag_regression(sentiment, close, horizon=1)
    assert abs(result.tstat) < 3  # not guaranteed <2, but should not be wildly significant


def test_grid_returns_one_row_per_horizon():
    rng = np.random.RandomState(2)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    sentiment = pd.Series(rng.normal(0, 1, n), index=dates)
    close = pd.Series(100 * (1 + rng.normal(0, 0.01, n)).cumprod(), index=dates)
    grid = run_lag_regression_grid(sentiment, close, horizons=[1, 2, 3])
    assert grid["horizon"].tolist() == [1, 2, 3]
    assert len(grid) == 3
