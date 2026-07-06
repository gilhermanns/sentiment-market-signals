import pandas as pd
import pytest

from sentiment_signals.backtest import generate_positions, run_backtest, trailing_sentiment


def test_trailing_sentiment_fills_gaps_with_zero_and_smooths():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    sparse = pd.Series([1.0], index=[dates[0]])
    out = trailing_sentiment(sparse, dates, window=2)
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(0.5)  # rolling mean of [1.0, 0.0]
    assert out.iloc[2] == pytest.approx(0.0)


def test_generate_positions_uses_prior_day_signal_not_same_day():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    trailing = pd.Series([1.0, 1.0, -1.0, -1.0], index=dates)
    pos = generate_positions(trailing, long_threshold=0.5, short_threshold=-0.5)
    # position on day0 must be 0 (no prior-day info yet, shift creates NaN -> not >= threshold)
    assert pos.iloc[0] == 0.0
    # position on day1 reflects day0's trailing sentiment (1.0 -> long)
    assert pos.iloc[1] == 1.0
    # position on day2 still reflects day1's trailing sentiment (1.0 -> long), not day2's own -1.0
    assert pos.iloc[2] == 1.0
    # position on day3 reflects day2's trailing sentiment (-1.0 -> short)
    assert pos.iloc[3] == -1.0


def test_backtest_charges_commission_on_position_changes():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    returns = pd.Series([0.01, 0.01, 0.01], index=dates)
    positions = pd.Series([1.0, 1.0, 0.0], index=dates)  # one position change at the end
    detail, strat_stats, bh_stats = run_backtest(returns, positions, commission_bps=100.0)
    # day0: turnover = |1-0| treated as fresh entry -> cost = 100bps = 0.01
    assert detail["cost"].iloc[0] == pytest.approx(0.01)
    # day1: no change -> no cost
    assert detail["cost"].iloc[1] == pytest.approx(0.0)
    # day2: position goes from 1 to 0 -> turnover 1 -> cost 0.01
    assert detail["cost"].iloc[2] == pytest.approx(0.01)


def test_backtest_zero_commission_matches_pure_position_times_return():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    returns = pd.Series([0.02, -0.01, 0.03], index=dates)
    positions = pd.Series([1.0, 1.0, 1.0], index=dates)
    detail, strat_stats, bh_stats = run_backtest(returns, positions, commission_bps=0.0)
    assert detail["strategy_ret"].tolist() == pytest.approx(returns.tolist())
    assert strat_stats.total_return == pytest.approx(bh_stats.total_return)
