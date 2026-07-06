import numpy as np
import pandas as pd
import pytest

from sentiment_signals.event_study import (
    abnormal_returns,
    fit_market_model,
    find_strong_sentiment_events,
    run_event_study,
)


def _toy_market_and_ticker(n=40, alpha=0.001, beta=2.0, injected_ar=0.05, event_pos=20):
    rng = np.random.RandomState(0)
    market_ret = rng.normal(0, 0.01, size=n)
    ticker_ret = alpha + beta * market_ret
    ticker_ret[event_pos] += injected_ar  # a known abnormal shock on one day, exactly
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(market_ret, index=dates), pd.Series(ticker_ret, index=dates), dates[event_pos]


def test_market_model_recovers_known_alpha_beta():
    market_ret, ticker_ret, _ = _toy_market_and_ticker(alpha=0.002, beta=1.5, injected_ar=0.0)
    model = fit_market_model(ticker_ret, market_ret)
    assert model.alpha == pytest.approx(0.002, abs=1e-9)
    assert model.beta == pytest.approx(1.5, abs=1e-9)


def test_abnormal_return_matches_hand_computed_injected_shock():
    market_ret, ticker_ret, event_date = _toy_market_and_ticker(alpha=0.001, beta=2.0, injected_ar=0.05)
    # standard event-study practice: estimate the market model on an estimation window that
    # excludes the event itself, so the shock doesn't contaminate alpha/beta.
    clean_market, clean_ticker = market_ret.drop(index=event_date), ticker_ret.drop(index=event_date)
    model = fit_market_model(clean_ticker, clean_market)
    ar = abnormal_returns(ticker_ret, market_ret, model)
    # by construction every day's AR should be ~0 except the injected event day, which
    # should show ~0.05 abnormal return (hand-computed: actual - (alpha + beta*market))
    assert ar[event_date] == pytest.approx(0.05, abs=1e-9)
    other_days_ar = ar.drop(index=event_date)
    assert other_days_ar.abs().max() < 1e-9


def test_event_study_car_at_day_zero_matches_injected_shock():
    market_ret, ticker_ret, event_date = _toy_market_and_ticker(alpha=0.001, beta=2.0, injected_ar=0.05)
    clean_market, clean_ticker = market_ret.drop(index=event_date), ticker_ret.drop(index=event_date)
    model = fit_market_model(clean_ticker, clean_market)
    ar = abnormal_returns(ticker_ret, market_ret, model)
    result = run_event_study(ar, pd.DatetimeIndex([event_date]), window=(-2, 2))
    day0 = result[result["rel_day"] == 0].iloc[0]
    assert day0["aar"] == pytest.approx(0.05, abs=1e-9)
    assert day0["n_events"] == 1
    # CAR before day 0 should be ~0 (no abnormal return before the shock)
    day_minus1 = result[result["rel_day"] == -1].iloc[0]
    assert day_minus1["car"] == pytest.approx(0.0, abs=1e-9)
    # CAR should jump up by the shock size at day 0 and hold afterward
    day1 = result[result["rel_day"] == 1].iloc[0]
    assert day1["car"] == pytest.approx(0.05, abs=1e-9)


def test_find_strong_sentiment_events_thresholds_correctly():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    sentiment = pd.Series([0.9, 0.1, -0.95, 0.0, 0.5], index=dates)
    events = find_strong_sentiment_events(sentiment, threshold=0.8)
    assert list(events) == [dates[0], dates[2]]


def test_event_study_averages_across_multiple_events():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    ar = pd.Series(0.0, index=dates)
    ar.iloc[5] = 0.02
    ar.iloc[10] = 0.04
    events = pd.DatetimeIndex([dates[5], dates[10]])
    result = run_event_study(ar, events, window=(0, 0))
    assert result.iloc[0]["aar"] == pytest.approx(0.03)  # mean of 0.02 and 0.04
    assert result.iloc[0]["n_events"] == 2
