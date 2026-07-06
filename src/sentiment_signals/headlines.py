"""Synthetic financial headline generation.

No live news archive is reachable from this environment, so headline text here is templated
and randomly assembled rather than scraped. It is intentionally generated independent of the
realized market data (see market_data.py) so that any predictive relationship found later is
not an artifact of construction -- a deliberate, documented placebo design. Swap
`generate_market_headlines` for a real ingestion function (e.g. a news API client or a load of
the Financial PhraseBank dataset) to point the rest of the pipeline at real headlines.
"""
from __future__ import annotations

import random

import pandas as pd

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "XOM": "Exxon Mobil",
}

_POS_VERBS = ["surged", "soared", "rallied", "jumped", "climbed", "advanced"]
_NEG_VERBS = ["plunged", "tumbled", "sank", "slumped", "fell", "dropped"]
_POS_ADJ = ["record", "strong", "robust", "better-than-expected", "resilient"]
_NEG_ADJ = ["disappointing", "weak", "soft", "worse-than-expected", "troubling"]
_METRICS = ["quarterly earnings", "revenue", "sales", "profit margins", "guidance", "cash flow"]
_EVENTS_POS = [
    "announced a share buyback program",
    "beat analyst earnings estimates",
    "raised its full-year guidance",
    "unveiled a new product line to strong demand",
    "received an analyst upgrade",
    "reported record quarterly profit",
]
_EVENTS_NEG = [
    "missed analyst earnings estimates",
    "cut its full-year guidance",
    "faces a regulatory investigation",
    "announced layoffs amid cost cuts",
    "received an analyst downgrade",
    "warned of a demand slowdown",
]
_EVENTS_NEU = [
    "will report quarterly results next week",
    "held its annual shareholder meeting",
    "announced a change to its board of directors",
    "presented at an industry conference",
    "filed its routine quarterly disclosure",
    "maintained its previous guidance",
]

POSITIVE_TEMPLATES = [
    "{company} shares {pos_verb} after {pos_event}",
    "{company} stock {pos_verb} on {pos_adj} {metric}",
    "{company} {pos_event}, shares climb",
    "Analysts turn bullish on {company} after {pos_adj} {metric}",
    "{company} rallies as it {pos_event}",
]
NEGATIVE_TEMPLATES = [
    "{company} shares {neg_verb} after {neg_event}",
    "{company} stock {neg_verb} on {neg_adj} {metric}",
    "{company} {neg_event}, shares slide",
    "Analysts turn cautious on {company} after {neg_adj} {metric}",
    "{company} drops as it {neg_event}",
]
NEUTRAL_TEMPLATES = [
    "{company} {neu_event}",
    "{company} shares trade flat as it {neu_event}",
    "{company} {metric} in line with expectations",
    "{company} to hold investor call on {metric}",
    "{company} {neu_event}, no major stock reaction",
]


def _fill(template: str, company: str, rng: random.Random) -> str:
    return template.format(
        company=company,
        pos_verb=rng.choice(_POS_VERBS),
        neg_verb=rng.choice(_NEG_VERBS),
        pos_adj=rng.choice(_POS_ADJ),
        neg_adj=rng.choice(_NEG_ADJ),
        metric=rng.choice(_METRICS),
        pos_event=rng.choice(_EVENTS_POS),
        neg_event=rng.choice(_EVENTS_NEG),
        neu_event=rng.choice(_EVENTS_NEU),
    )


def generate_headline(label: str, company: str, rng: random.Random) -> str:
    templates = {"positive": POSITIVE_TEMPLATES, "negative": NEGATIVE_TEMPLATES, "neutral": NEUTRAL_TEMPLATES}[label]
    return _fill(rng.choice(templates), company, rng)


def generate_labeled_corpus(n_per_class: int = 400, seed: int = 42) -> pd.DataFrame:
    """A larger synthetic labeled corpus for training/evaluating the classifier.

    One row per headline with columns text, label. Company names are drawn from the
    full ticker universe so the classifier learns sentiment vocabulary, not company identity.
    """
    rng = random.Random(seed)
    companies = list(COMPANY_NAMES.values())
    rows = []
    for label in ("positive", "negative", "neutral"):
        for _ in range(n_per_class):
            company = rng.choice(companies)
            rows.append({"text": generate_headline(label, company, rng), "label": label})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_market_headlines(
    tickers: list[str],
    trading_dates: pd.DatetimeIndex,
    max_per_day: int = 2,
    headline_prob: float = 0.35,
    label_weights: dict[str, float] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Templated headlines attached to real tickers and real trading dates.

    Each (ticker, date) independently gets 0..max_per_day headlines. Label assignment is
    i.i.d. random per headline_prob/label_weights and does not depend on realized returns,
    by construction -- this keeps the downstream predictive-value tests honest placebo tests
    rather than a self-fulfilling correlation.
    """
    weights = label_weights or {"positive": 0.4, "negative": 0.4, "neutral": 0.2}
    labels = list(weights.keys())
    probs = list(weights.values())
    rng = random.Random(seed)
    rows = []
    for ticker in tickers:
        company = COMPANY_NAMES.get(ticker, ticker)
        for date in trading_dates:
            n_headlines = 0
            for _ in range(max_per_day):
                if rng.random() < headline_prob:
                    n_headlines += 1
            for _ in range(n_headlines):
                label = rng.choices(labels, weights=probs, k=1)[0]
                text = generate_headline(label, company, rng)
                rows.append({"date": date, "ticker": ticker, "text": text, "true_label": label})
    return pd.DataFrame(rows)
