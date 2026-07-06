"""Hand-built finance sentiment lexicon: a naive keyword-count scorer with no external downloads."""
from __future__ import annotations

import re

POSITIVE_TERMS = {
    "surge", "surged", "surges", "soar", "soared", "soars", "rally", "rallied", "rallies",
    "jump", "jumped", "jumps", "gain", "gained", "gains", "beat", "beats", "beating",
    "outperform", "outperforms", "outperformed", "upgrade", "upgraded", "upgrades",
    "record", "strong", "stronger", "strength", "growth", "grew", "grows", "profit",
    "profits", "profitable", "bullish", "boom", "booming", "expand", "expands",
    "expanded", "expansion", "raise", "raised", "raises", "raising", "optimistic",
    "optimism", "exceed", "exceeds", "exceeded", "exceeding", "buyback", "dividend",
    "breakthrough", "milestone", "robust", "accelerate", "accelerated", "accelerating",
    "win", "wins", "won", "winning", "recovery", "recovered", "recovers", "upside",
    "outlook", "positive", "boost", "boosted", "boosts", "climb", "climbed", "climbs",
    "advance", "advanced", "advances", "top", "topped", "tops", "resilient",
}

NEGATIVE_TERMS = {
    "plunge", "plunged", "plunges", "tumble", "tumbled", "tumbles", "slump", "slumped",
    "slumps", "sink", "sank", "sinks", "fall", "fell", "falls", "falling", "drop",
    "dropped", "drops", "dropping", "miss", "missed", "misses", "missing",
    "underperform", "underperforms", "underperformed", "downgrade", "downgraded",
    "downgrades", "weak", "weaker", "weakness", "loss", "losses", "losing", "bearish",
    "bust", "shrink", "shrinks", "shrank", "shrinking", "contraction", "cut", "cuts",
    "cutting", "layoff", "layoffs", "lawsuit", "investigation", "probe", "fraud",
    "scandal", "recall", "warns", "warned", "warning", "pessimistic", "pessimism",
    "shortfall", "decline", "declined", "declines", "declining", "slowdown",
    "downturn", "crash", "crashed", "crashes", "selloff", "risk", "risks",
    "concern", "concerns", "concerning", "default", "bankruptcy", "bankrupt", "delay",
    "delayed", "delays", "volatile", "volatility", "sued", "penalty", "fine", "fined",
}

_NEGATIONS = {"not", "no", "never", "without", "isn't", "wasn't", "aren't", "won't", "cannot"}
_WORD_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def score(text: str) -> float:
    """Naive lexicon sentiment score in [-1, 1]: (pos - neg) / (pos + neg), 0 if no hits.

    A simple bigram negation flip is applied: a positive/negative term immediately
    preceded by a negation word ("not", "no", ...) contributes to the opposite polarity.
    """
    tokens = _tokenize(text)
    pos_hits = 0
    neg_hits = 0
    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in _NEGATIONS
        if tok in POSITIVE_TERMS:
            if negated:
                neg_hits += 1
            else:
                pos_hits += 1
        elif tok in NEGATIVE_TERMS:
            if negated:
                pos_hits += 1
            else:
                neg_hits += 1
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    return (pos_hits - neg_hits) / total


def score_many(texts: list[str]) -> list[float]:
    return [score(t) for t in texts]
