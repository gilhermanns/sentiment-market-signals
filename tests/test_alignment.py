import pandas as pd
import pytest

from sentiment_signals.alignment import (
    build_lagged_dataset,
    build_volatility_dataset,
    daily_sentiment_aggregates,
)


def _toy_close():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.Series([100, 102, 101, 105, 110, 108], index=dates)


def test_horizon_must_be_positive_integer():
    close = _toy_close()
    sentiment = pd.Series([0.5] * len(close), index=close.index)
    with pytest.raises(ValueError):
        build_lagged_dataset(sentiment, close, horizon=0)
    with pytest.raises(ValueError):
        build_lagged_dataset(sentiment, close, horizon=-1)
    with pytest.raises(ValueError):
        build_lagged_dataset(sentiment, close, horizon=2.5)


def test_forward_return_matches_hand_computation():
    close = _toy_close()
    sentiment = pd.Series([1.0] * len(close), index=close.index)
    out = build_lagged_dataset(sentiment, close, horizon=1)
    # day0->day1: 102/100 - 1; day1->day2: 101/102 -1; etc. Last day has no t+1, dropped.
    expected = [102 / 100 - 1, 101 / 102 - 1, 105 / 101 - 1, 110 / 105 - 1, 108 / 110 - 1]
    assert out["fwd_ret"].tolist() == pytest.approx(expected)
    assert len(out) == len(close) - 1  # last row has no forward horizon available


def test_horizon_two_skips_two_days_forward_not_one():
    close = _toy_close()
    sentiment = pd.Series([1.0] * len(close), index=close.index)
    out = build_lagged_dataset(sentiment, close, horizon=2)
    expected_first = 101 / 100 - 1  # day0 -> day2, NOT day0->day1
    assert out["fwd_ret"].iloc[0] == pytest.approx(expected_first)
    assert len(out) == len(close) - 2


def test_sentiment_is_paired_with_its_own_day_not_shifted():
    """The function must pick up sentiment[t] and pair it with the return starting at t,
    never with a sentiment value from a different day (which would be a silent lookahead
    or lookbehind bug)."""
    close = _toy_close()
    # distinct sentiment per day so any misalignment is detectable
    sentiment = pd.Series(range(len(close)), index=close.index, dtype=float)
    out = build_lagged_dataset(sentiment, close, horizon=1)
    assert out["sentiment"].tolist() == list(range(len(close) - 1))


def test_a_naively_pre_shifted_future_feature_produces_detectably_different_output():
    """Simulates the classic lookahead bug: someone builds a 'sentiment' series that has
    already been shifted to align with the future outcome before calling the aligner. The
    aligner has no way to know this and will pair it at face value with day t's row -- so
    the *correct* usage is to always pass the raw, same-day-dated sentiment series. This
    test demonstrates that a pre-shifted series yields a materially different (and wrong)
    pairing than the correct raw series, so such a bug is observable via a mismatch against
    a known-good reference alignment rather than silently identical results."""
    close = _toy_close()
    raw_sentiment = pd.Series(range(len(close)), index=close.index, dtype=float)
    buggy_pre_shifted = raw_sentiment.shift(-1)  # accidentally using tomorrow's value today

    correct = build_lagged_dataset(raw_sentiment, close, horizon=1)
    buggy = build_lagged_dataset(buggy_pre_shifted, close, horizon=1)

    assert correct["sentiment"].tolist() != buggy["sentiment"].tolist()
    assert correct["sentiment"].tolist() == [0, 1, 2, 3, 4]
    # last row is dropped either way (fwd_ret is NaN there since there's no day 6 close);
    # but the buggy series' remaining values are shifted by one day relative to "correct"
    assert buggy["sentiment"].tolist() == [1, 2, 3, 4, 5]


def test_unsorted_close_index_is_rejected():
    close = _toy_close().iloc[::-1]
    sentiment = pd.Series([1.0] * len(close), index=close.index)
    with pytest.raises(ValueError):
        build_lagged_dataset(sentiment, close, horizon=1)


def test_volatility_dataset_shifts_target_forward_by_one_day():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    abs_sent = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=dates)
    dispersion = pd.Series([0.0] * 5, index=dates)
    realized_vol = pd.Series([1, 2, 3, 4, 5], index=dates, dtype=float)
    out = build_volatility_dataset(abs_sent, dispersion, realized_vol)
    assert out["fwd_vol"].tolist() == [2, 3, 4, 5]
    assert out["abs_sentiment"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_daily_sentiment_aggregates_dispersion_and_mean():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "ticker": ["AAA", "AAA", "AAA"],
            "score": [1.0, -1.0, 0.5],
        }
    )
    out = daily_sentiment_aggregates(df, "score")
    day1 = out[out["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert day1["mean_sentiment"] == pytest.approx(0.0)
    assert day1["n_headlines"] == 2
    assert day1["dispersion"] > 0
    day2 = out[out["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert day2["dispersion"] == 0.0  # single headline -> no dispersion
