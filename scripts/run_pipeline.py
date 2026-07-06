"""End-to-end pipeline run: fetch real prices, generate synthetic headlines, score sentiment,
run predictive-value tests, backtest, and render the README charts + dashboard.

Usage: python scripts/run_pipeline.py
Writes CSV summaries to reports/ and SVG charts to docs/img/, and prints a text summary
(the same numbers used in the README) to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_signals import classifier, headlines, lexicon, market_data
from sentiment_signals.alignment import daily_sentiment_aggregates
from sentiment_signals.backtest import generate_positions, run_backtest, trailing_sentiment
from sentiment_signals.event_study import (
    abnormal_returns,
    fit_market_model,
    find_strong_sentiment_events,
    run_event_study,
)
from sentiment_signals.lag_regression import run_lag_regression_grid
from sentiment_signals.volatility import run_volatility_regression
from sentiment_signals.dashboard import build_dashboard_html

TICKERS = ["AAPL", "MSFT", "TSLA", "JPM", "XOM"]
HORIZONS = [1, 2, 3, 4, 5]
REPORTS = ROOT / "reports"
IMG = ROOT / "docs" / "img"
REPORTS.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

# -- palette (dataviz skill reference palette, light mode) --
C_BLUE = "#2a78d6"
C_RED = "#e34948"
C_GREEN = "#1baf7a"
C_VIOLET = "#4a3aa7"
C_ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def log(msg):
    print(msg, flush=True)


def main():
    summary = {}

    # ---------------------------------------------------------------- market data
    log("Fetching real OHLC price history from stockanalysis.com ...")
    prices = {}
    for t in TICKERS:
        df = market_data.fetch_ohlc(t, range_="5Y")
        prices[t] = df
        log(f"  {t}: {len(df)} trading days, {df.index.min().date()} to {df.index.max().date()}")

    returns = {t: market_data.compute_returns(prices[t]) for t in TICKERS}
    summary["tickers"] = TICKERS
    summary["price_history"] = {
        t: {"n_days": len(prices[t]), "start": str(prices[t].index.min().date()),
            "end": str(prices[t].index.max().date())}
        for t in TICKERS
    }

    # ---------------------------------------------------------------- classifier training
    log("Training TF-IDF + Logistic Regression sentiment classifier on synthetic corpus ...")
    corpus = headlines.generate_labeled_corpus(n_per_class=500, seed=42)
    clf, report = classifier.train_and_evaluate(corpus, test_size=0.25, random_state=0)
    log(f"  accuracy={report.accuracy:.3f} macro_f1={report.macro_f1:.3f} "
        f"majority_baseline={report.majority_baseline_accuracy:.3f} "
        f"n_train={report.n_train} n_test={report.n_test}")
    summary["classifier"] = {
        "accuracy": report.accuracy, "macro_f1": report.macro_f1,
        "majority_baseline_accuracy": report.majority_baseline_accuracy,
        "n_train": report.n_train, "n_test": report.n_test,
        "confusion_labels": report.labels, "confusion_matrix": report.confusion.tolist(),
    }
    plot_confusion_matrix(report.confusion, report.labels, IMG / "confusion_matrix.svg")

    # ---------------------------------------------------------------- synthetic market headlines
    log("Generating synthetic market headlines tied to real tickers/trading dates ...")
    all_dates = sorted(set().union(*[set(prices[t].index) for t in TICKERS]))
    all_dates = pd.DatetimeIndex(all_dates)
    mh = headlines.generate_market_headlines(TICKERS, all_dates, max_per_day=2, headline_prob=0.35, seed=7)
    log(f"  generated {len(mh)} headlines across {len(TICKERS)} tickers / {len(all_dates)} trading dates")
    mh["lexicon_score"] = lexicon.score_many(mh["text"].tolist())
    mh["classifier_score"] = clf.score(mh["text"].tolist())
    mh.to_csv(REPORTS / "synthetic_headlines.csv", index=False)
    summary["n_headlines"] = len(mh)

    lex_daily = daily_sentiment_aggregates(mh, "lexicon_score")
    clf_daily = daily_sentiment_aggregates(mh, "classifier_score")

    # ---------------------------------------------------------------- lag regressions
    log("Running lag-structure regressions (sentiment[t] -> forward return[t..t+h]) ...")
    lag_rows = []
    for ticker in TICKERS:
        close = prices[ticker]["Close"]
        for scorer_name, daily in [("lexicon", lex_daily), ("classifier", clf_daily)]:
            sub = daily[daily["ticker"] == ticker].set_index("date")["mean_sentiment"]
            grid = run_lag_regression_grid(sub, close, HORIZONS)
            grid["ticker"] = ticker
            grid["scorer"] = scorer_name
            lag_rows.append(grid)
    lag_df = pd.concat(lag_rows, ignore_index=True)
    lag_df.to_csv(REPORTS / "lag_regressions.csv", index=False)

    pooled_lag = (
        lag_df.groupby(["scorer", "horizon"])
        .agg(mean_coef=("coef", "mean"), mean_tstat=("tstat", "mean"), n_tickers=("ticker", "count"))
        .reset_index()
    )
    pooled_lag.to_csv(REPORTS / "lag_regressions_pooled.csv", index=False)
    log(pooled_lag.to_string(index=False))
    summary["lag_regression_pooled"] = pooled_lag.to_dict(orient="records")
    plot_lag_tstats(pooled_lag, IMG / "lag_regression_tstats.svg")

    # ---------------------------------------------------------------- volatility regression
    log("Running |sentiment| / dispersion -> next-day realized volatility regressions ...")
    vol_rows = []
    for ticker in TICKERS:
        rv = returns[ticker].realized_vol
        for scorer_name, daily in [("lexicon", lex_daily), ("classifier", clf_daily)]:
            sub = daily[daily["ticker"] == ticker].set_index("date")
            abs_sent = sub["abs_sentiment"].reindex(rv.index)
            disp = sub["dispersion"].reindex(rv.index)
            res = run_volatility_regression(abs_sent, disp, rv)
            vol_rows.append({"ticker": ticker, "scorer": scorer_name, **res.__dict__})
    vol_df = pd.DataFrame(vol_rows)
    vol_df.to_csv(REPORTS / "volatility_regressions.csv", index=False)
    log(vol_df.to_string(index=False))
    summary["volatility_regression"] = vol_df.to_dict(orient="records")

    # ---------------------------------------------------------------- event study
    log("Running event study: abnormal return around strong-sentiment headline events ...")
    window = (-5, 5)
    per_ticker_results = []
    for ticker in TICKERS:
        others = [x for x in TICKERS if x != ticker]
        own_ret = returns[ticker].ret
        mkt_ret = pd.concat([returns[o].ret for o in others], axis=1).mean(axis=1)
        model = fit_market_model(own_ret, mkt_ret)
        ar = abnormal_returns(own_ret, mkt_ret, model)

        sub = lex_daily[lex_daily["ticker"] == ticker].set_index("date")["mean_sentiment"]
        threshold = sub.abs().quantile(0.85)
        events = find_strong_sentiment_events(sub, threshold=max(threshold, 1e-9))
        result = run_event_study(ar, events, window=window)
        result["ticker"] = ticker
        result["n_events_ticker"] = len(events)
        per_ticker_results.append(result)

    event_df = pd.concat(per_ticker_results, ignore_index=True)
    event_df.to_csv(REPORTS / "event_study_per_ticker.csv", index=False)

    pooled_event = (
        event_df.assign(weighted_aar=lambda d: d["aar"] * d["n_events"])
        .groupby("rel_day")
        .apply(lambda d: pd.Series({
            "aar": d["weighted_aar"].sum() / d["n_events"].sum() if d["n_events"].sum() else float("nan"),
            "n_events": d["n_events"].sum(),
        }), include_groups=False)
        .reset_index()
        .sort_values("rel_day")
    )
    pooled_event["car"] = pooled_event["aar"].cumsum()
    pooled_event.to_csv(REPORTS / "event_study_pooled.csv", index=False)
    log(pooled_event.to_string(index=False))
    summary["event_study_pooled"] = pooled_event.to_dict(orient="records")
    plot_event_study(pooled_event, IMG / "event_study_car.svg")

    # ---------------------------------------------------------------- backtest
    log("Backtesting a trailing-sentiment-filtered long/flat/short strategy vs buy-and-hold ...")
    bt_rows = []
    detail_for_plot = None
    for ticker in TICKERS:
        close = prices[ticker]["Close"]
        ret = returns[ticker].ret
        sub = lex_daily[lex_daily["ticker"] == ticker].set_index("date")["mean_sentiment"]
        trailing = trailing_sentiment(sub, close.index, window=3)
        long_thr = trailing[trailing != 0].quantile(0.7) if (trailing != 0).any() else 0.34
        short_thr = trailing[trailing != 0].quantile(0.3) if (trailing != 0).any() else -0.34
        positions = generate_positions(trailing, long_threshold=long_thr, short_threshold=short_thr)
        detail, strat_stats, bh_stats = run_backtest(ret, positions, commission_bps=5.0)
        bt_rows.append({
            "ticker": ticker,
            "strategy_total_return": strat_stats.total_return,
            "strategy_sharpe": strat_stats.sharpe,
            "strategy_max_dd": strat_stats.max_drawdown,
            "buy_hold_total_return": bh_stats.total_return,
            "buy_hold_sharpe": bh_stats.sharpe,
            "buy_hold_max_dd": bh_stats.max_drawdown,
        })
        if ticker == "AAPL":
            detail_for_plot = detail
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(REPORTS / "backtest_results.csv", index=False)
    log(bt_df.to_string(index=False))
    summary["backtest"] = bt_df.to_dict(orient="records")
    plot_backtest(detail_for_plot, "AAPL", IMG / "backtest_equity_curve.svg")

    # ---------------------------------------------------------------- dashboard
    log("Building headline/price dashboard for AAPL ...")
    dash_start = str(prices["AAPL"].index[-80].date())
    dash_end = str(prices["AAPL"].index[-1].date())
    html_out = build_dashboard_html("AAPL", prices["AAPL"], mh, dash_start, dash_end, score_col="lexicon_score")
    (REPORTS / "dashboard_AAPL.html").write_text(html_out)
    log(f"  wrote {REPORTS / 'dashboard_AAPL.html'} covering {dash_start} to {dash_end}")

    with open(REPORTS / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log("Done. Summary written to reports/run_summary.json")


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Classifier confusion matrix (row-normalized)", fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.5 else INK
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center",
                    fontsize=8.5, color=color)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_lag_tstats(pooled_lag: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.35
    horizons = sorted(pooled_lag["horizon"].unique())
    x = np.arange(len(horizons))
    for i, (scorer, color) in enumerate([("lexicon", C_BLUE), ("classifier", C_ORANGE)]):
        sub = pooled_lag[pooled_lag["scorer"] == scorer].sort_values("horizon")
        ax.bar(x + (i - 0.5) * width, sub["mean_tstat"], width=width, label=scorer, color=color)
    ax.axhline(1.96, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.axhline(-1.96, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(len(horizons) - 0.5, 1.96, " 95% sig.", fontsize=7.5, color=INK_MUTED, va="bottom")
    ax.set_xticks(x, [f"t+{h}" for h in horizons])
    ax.set_xlabel("Forward return horizon")
    ax.set_ylabel("Mean t-statistic across tickers")
    ax.set_title("Sentiment -> forward return: lag regression t-stats", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_event_study(pooled_event: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(pooled_event["rel_day"], pooled_event["car"] * 100, color=C_BLUE, linewidth=2,
            marker="o", markersize=4)
    ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xlabel("Trading days relative to strong-sentiment headline (day 0)")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.set_title("Event study: CAR around strong-sentiment events, pooled across tickers", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_backtest(detail: pd.DataFrame, ticker: str, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(detail.index, detail["strategy_cum"], color=C_VIOLET, linewidth=1.8, label="Sentiment-filtered strategy")
    ax.plot(detail.index, detail["buy_hold_cum"], color=INK_MUTED, linewidth=1.8,
            label="Buy and hold", linestyle="--")
    ax.set_ylabel("Growth of $1")
    ax.set_title(f"{ticker}: sentiment-filtered strategy vs buy-and-hold (5bps commission)", fontsize=10)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
