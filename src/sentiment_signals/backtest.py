"""Sentiment-filtered long/flat/short strategy vs buy-and-hold, with commission costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def trailing_sentiment(daily_mean_sentiment: pd.Series, trading_index: pd.DatetimeIndex, window: int = 3) -> pd.Series:
    """Reindex sparse per-day mean sentiment onto the full trading calendar (0 on days with
    no headlines) and take a trailing rolling mean."""
    full = daily_mean_sentiment.reindex(trading_index).fillna(0.0)
    return full.rolling(window, min_periods=1).mean()


def generate_positions(trailing: pd.Series, long_threshold: float, short_threshold: float) -> pd.Series:
    """Position held *during* day t, decided from information available at the close of day
    t-1 (trailing sentiment shifted forward by one day) -- so day t's return is never used to
    pick day t's position."""
    signal = trailing.shift(1)
    pos = pd.Series(0.0, index=trailing.index)
    pos[signal >= long_threshold] = 1.0
    pos[signal <= short_threshold] = -1.0
    return pos


@dataclass
class BacktestStats:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float


def _stats(cum_ret: pd.Series, daily_ret: pd.Series) -> BacktestStats:
    n = len(daily_ret)
    total_return = float(cum_ret.iloc[-1] - 1) if len(cum_ret) else float("nan")
    ann_return = float((cum_ret.iloc[-1]) ** (TRADING_DAYS_PER_YEAR / n) - 1) if n > 0 and cum_ret.iloc[-1] > 0 else float("nan")
    ann_vol = float(daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(ann_return / ann_vol) if ann_vol and not np.isnan(ann_vol) and ann_vol != 0 else float("nan")
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1.0
    max_dd = float(drawdown.min())
    return BacktestStats(total_return, ann_return, ann_vol, sharpe, max_dd)


def run_backtest(
    returns: pd.Series, positions: pd.Series, commission_bps: float = 5.0
) -> tuple[pd.DataFrame, BacktestStats, BacktestStats]:
    """Returns (per-day detail frame, strategy stats, buy-and-hold stats).

    Commission is charged on |change in position| * commission_bps/10000, applied on the day
    the position changes.
    """
    df = pd.DataFrame({"ret": returns, "pos": positions}).dropna()
    turnover = df["pos"].diff().abs().fillna(df["pos"].abs())
    cost = turnover * (commission_bps / 10_000.0)
    strat_ret = df["pos"] * df["ret"] - cost
    bh_ret = df["ret"]

    detail = pd.DataFrame(
        {
            "ret": df["ret"],
            "position": df["pos"],
            "cost": cost,
            "strategy_ret": strat_ret,
            "strategy_cum": (1 + strat_ret).cumprod(),
            "buy_hold_cum": (1 + bh_ret).cumprod(),
        }
    )
    strat_stats = _stats(detail["strategy_cum"], strat_ret)
    bh_stats = _stats(detail["buy_hold_cum"], bh_ret)
    return detail, strat_stats, bh_stats
