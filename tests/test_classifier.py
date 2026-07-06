from sentiment_signals import headlines
from sentiment_signals.classifier import train_and_evaluate


def test_classifier_beats_majority_class_baseline_on_held_out_data():
    corpus = headlines.generate_labeled_corpus(n_per_class=300, seed=42)
    clf, report = train_and_evaluate(corpus, test_size=0.25, random_state=0)
    assert report.accuracy > report.majority_baseline_accuracy
    assert report.accuracy > 0.7


def test_confusion_matrix_shape_and_diagonal_dominance():
    corpus = headlines.generate_labeled_corpus(n_per_class=300, seed=42)
    clf, report = train_and_evaluate(corpus, test_size=0.25, random_state=0)
    assert report.confusion.shape == (3, 3)
    assert report.confusion.trace() > report.confusion.sum() / 3


def test_score_is_bounded_and_signed_sensibly():
    corpus = headlines.generate_labeled_corpus(n_per_class=200, seed=1)
    clf, _ = train_and_evaluate(corpus, random_state=0)
    scores = clf.score(["Shares surged after record profit and a strong outlook",
                        "Shares plunged after a disappointing miss and layoffs"])
    assert -1.0 <= scores[0] <= 1.0
    assert -1.0 <= scores[1] <= 1.0
    assert scores[0] > scores[1]


def test_predict_label_returns_known_labels():
    corpus = headlines.generate_labeled_corpus(n_per_class=100, seed=7)
    clf, _ = train_and_evaluate(corpus, random_state=0)
    preds = clf.predict_label(["Apple shares surged after record quarterly profit"])
    assert preds[0] in {"positive", "negative", "neutral"}
