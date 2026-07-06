from sentiment_signals import lexicon


def test_clearly_positive_sentence_scores_positive():
    text = "Company shares surged after record profit and a strong outlook, analysts turn bullish"
    assert lexicon.score(text) > 0


def test_clearly_negative_sentence_scores_negative():
    text = "Company shares plunged after a disappointing miss, layoffs, and a bearish downgrade"
    assert lexicon.score(text) < 0


def test_neutral_sentence_with_no_terms_scores_zero():
    text = "The company will report quarterly results next week"
    assert lexicon.score(text) == 0.0


def test_balanced_positive_and_negative_terms_cancel_out():
    text = "Shares surged then plunged, gains offset by losses"
    assert lexicon.score(text) == 0.0


def test_negation_flips_polarity():
    positive_text = "Sales showed strong growth"
    negated_text = "Sales did not show strong growth, no growth here"
    assert lexicon.score(positive_text) > 0
    assert lexicon.score(negated_text) < lexicon.score(positive_text)


def test_score_many_matches_individual_scores():
    texts = ["Shares surged on strong growth", "Shares plunged on weak earnings"]
    assert lexicon.score_many(texts) == [lexicon.score(t) for t in texts]


def test_score_bounded_between_minus_one_and_one():
    for text in [
        "surge surge surge plunge",
        "plunge plunge plunge surge surge surge surge",
        "no sentiment words here at all",
    ]:
        s = lexicon.score(text)
        assert -1.0 <= s <= 1.0


def test_every_lexicon_term_is_reachable_by_the_tokenizer():
    """Every entry in POSITIVE_TERMS/NEGATIVE_TERMS must round-trip through _tokenize as a
    single token, or it can never actually match anything -- e.g. a hyphenated entry like
    "sell-off" is silently dead code, since _WORD_RE splits on hyphens and only ever
    produces "sell" and "off" as separate tokens."""
    for term in lexicon.POSITIVE_TERMS | lexicon.NEGATIVE_TERMS:
        assert lexicon._tokenize(term) == [term], (
            f"lexicon term {term!r} does not round-trip through the tokenizer and can never match"
        )
