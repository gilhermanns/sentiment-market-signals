"""Real historical OHLC data via the stockanalysis.com public history API (no auth/key required).

stooq.com's CSV endpoint is behind a JavaScript proof-of-work bot check as of this writing and
is not usable from a plain HTTP client, so this module uses stockanalysis.com's JSON history
endpoint instead, which serves the same kind of real, unauthenticated daily OHLCV data.
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

import pandas as pd

HISTORY_URL = "https://stockanalysis.com/api/symbol/s/{symbol}/history?range={range_}&period=Daily"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) sentiment-market-signals/0.1"


def fetch_ohlc(ticker: str, range_: str = "5Y", timeout: int = 15) -> pd.DataFrame:
    """Download daily OHLC history for a US-listed ticker.

    Returns a DataFrame indexed by trading date (ascending) with columns
    Open, High, Low, Close, Volume. Raises ValueError if the response has no rows
    (e.g. unknown ticker) or if the endpoint is unreachable (network error, timeout, or
    an unparseable response), with a message that says which ticker and endpoint failed
    rather than surfacing a raw urllib traceback.
    """
    url = HISTORY_URL.format(symbol=ticker.upper(), range_=range_)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise ConnectionError(
            f"Could not reach {url} for ticker {ticker!r} ({exc}). "
            "This pipeline needs live network access to fetch real price history; "
            "pytest and the rest of the package work fully offline without it."
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Response from {url} for ticker {ticker!r} was not valid JSON "
            f"(got {len(raw)} bytes) -- the endpoint may be down or rate-limiting."
        ) from exc
    rows = payload.get("data") or []
    if not rows:
        raise ValueError(f"No history data returned for ticker {ticker!r}")
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["t"])
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_many(tickers: list[str], range_: str = "5Y") -> dict[str, pd.DataFrame]:
    return {t: fetch_ohlc(t, range_=range_) for t in tickers}


@dataclass
class PriceReturns:
    close: pd.Series
    ret: pd.Series
    realized_vol: pd.Series


def compute_returns(df: pd.DataFrame, vol_window: int = 5) -> PriceReturns:
    """Simple daily returns and a trailing realized-volatility series.

    realized_vol[t] uses returns up to and including day t (a backward-looking
    rolling std), so it is a same-day-available feature, not a lookahead one.
    """
    close = df["Close"]
    ret = close.pct_change()
    realized_vol = ret.rolling(vol_window).std()
    return PriceReturns(close=close, ret=ret, realized_vol=realized_vol)
