import json
import urllib.error
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sentiment_signals.market_data import compute_returns, fetch_ohlc


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_ohlc_parses_history_payload():
    payload = {
        "data": [
            {"t": "2024-01-03", "o": 101, "h": 102, "l": 100, "c": 101.5, "v": 1000},
            {"t": "2024-01-02", "o": 99, "h": 100, "l": 98, "c": 99.5, "v": 900},
        ]
    }
    with patch("urllib.request.urlopen", return_value=_FakeResponse(json.dumps(payload).encode())):
        df = fetch_ohlc("AAPL")
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]  # sorted ascending
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.loc[pd.Timestamp("2024-01-02"), "Close"] == 99.5


def test_fetch_ohlc_raises_on_empty_data():
    payload = {"data": []}
    with patch("urllib.request.urlopen", return_value=_FakeResponse(json.dumps(payload).encode())):
        with pytest.raises(ValueError):
            fetch_ohlc("NOTAREALTICKER")


def test_fetch_ohlc_raises_connection_error_on_network_failure_not_a_raw_traceback():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network is unreachable")):
        with pytest.raises(ConnectionError, match="AAPL"):
            fetch_ohlc("AAPL")


def test_fetch_ohlc_raises_value_error_on_unparseable_response():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(b"<html>rate limited</html>")):
        with pytest.raises(ValueError, match="AAPL"):
            fetch_ohlc("AAPL")


def test_compute_returns_realized_vol_is_backward_looking_only():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    close = pd.Series(np.linspace(100, 110, 10), index=dates)
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1})
    result = compute_returns(df, vol_window=3)
    # ret.iloc[0] is always NaN (pct_change has no prior day), so the first fully-populated
    # 3-day rolling window is at position 3 (ret indices 1,2,3) -- realized_vol must use only
    # those, never index 4 or later.
    manual = result.ret.iloc[1:4].std()
    assert result.realized_vol.iloc[3] == pytest.approx(manual)
    assert pd.isna(result.realized_vol.iloc[2])
