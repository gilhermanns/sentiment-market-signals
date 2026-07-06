"""TF-IDF + Logistic Regression sentiment classifier trained on the synthetic labeled corpus.

A FinBERT (transformers, ProsusAI/finbert) model is a drop-in replacement for this class:
implement the same `.predict_label` / `.score` interface backed by the transformers pipeline
and the rest of the codebase (alignment, regressions, event study, backtest) is unaffected.
That swap needs a model download that this sandbox cannot reliably perform, so it is
documented here rather than implemented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]
_LABEL_SIGN = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


@dataclass
class EvalReport:
    accuracy: float
    macro_f1: float
    majority_baseline_accuracy: float
    confusion: np.ndarray
    labels: list[str]
    n_train: int
    n_test: int


class HeadlineSentimentClassifier:
    def __init__(self, max_features: int = 3000, C: float = 2.0, random_state: int = 0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=1, stop_words="english"
        )
        self.model = LogisticRegression(max_iter=2000, C=C, random_state=random_state)
        self._fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "HeadlineSentimentClassifier":
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def predict_label(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: list[str]) -> pd.DataFrame:
        X = self.vectorizer.transform(texts)
        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, columns=self.model.classes_)

    def score(self, texts: list[str]) -> np.ndarray:
        """Continuous sentiment score in [-1, 1]: P(positive) - P(negative)."""
        proba = self.predict_proba(texts)
        pos = proba["positive"] if "positive" in proba else 0.0
        neg = proba["negative"] if "negative" in proba else 0.0
        return (pos - neg).to_numpy()


def train_and_evaluate(
    corpus: pd.DataFrame, test_size: float = 0.25, random_state: int = 0
) -> tuple[HeadlineSentimentClassifier, EvalReport]:
    x_train, x_test, y_train, y_test = train_test_split(
        corpus["text"].tolist(),
        corpus["label"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=corpus["label"],
    )
    clf = HeadlineSentimentClassifier(random_state=random_state).fit(x_train, y_train)
    preds = clf.predict_label(x_test)

    majority_label = pd.Series(y_train).value_counts().idxmax()
    majority_preds = [majority_label] * len(y_test)

    report = EvalReport(
        accuracy=accuracy_score(y_test, preds),
        macro_f1=f1_score(y_test, preds, average="macro"),
        majority_baseline_accuracy=accuracy_score(y_test, majority_preds),
        confusion=confusion_matrix(y_test, preds, labels=LABELS),
        labels=LABELS,
        n_train=len(x_train),
        n_test=len(x_test),
    )
    return clf, report
