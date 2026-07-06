import pandas as pd

from sentiment_signals import headlines


def test_generate_labeled_corpus_has_balanced_classes_and_expected_size():
    corpus = headlines.generate_labeled_corpus(n_per_class=50, seed=1)
    assert len(corpus) == 150
    counts = corpus["label"].value_counts()
    assert set(counts.index) == {"positive", "negative", "neutral"}
    assert (counts == 50).all()


def test_generate_labeled_corpus_is_deterministic_given_seed():
    a = headlines.generate_labeled_corpus(n_per_class=20, seed=99)
    b = headlines.generate_labeled_corpus(n_per_class=20, seed=99)
    assert a["text"].tolist() == b["text"].tolist()
    assert a["label"].tolist() == b["label"].tolist()


def test_generate_labeled_corpus_has_vocabulary_variety():
    corpus = headlines.generate_labeled_corpus(n_per_class=100, seed=3)
    positive_texts = corpus.loc[corpus["label"] == "positive", "text"]
    assert positive_texts.nunique() > 20  # templated but not all identical


def test_generate_market_headlines_only_uses_requested_tickers_and_dates():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    out = headlines.generate_market_headlines(["AAPL", "MSFT"], dates, seed=5)
    assert set(out["ticker"]).issubset({"AAPL", "MSFT"})
    assert set(out["date"]).issubset(set(dates))
    assert set(out["true_label"]).issubset({"positive", "negative", "neutral"})


def test_generate_market_headlines_respects_max_per_day():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    out = headlines.generate_market_headlines(["AAPL"], dates, max_per_day=2, headline_prob=1.0, seed=1)
    counts = out.groupby("date").size()
    assert counts.max() <= 2
