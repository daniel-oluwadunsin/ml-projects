"""Data, model, and metric helpers for language identification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data.download_data import download_data


PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
LANGUAGE_NAMES = {
    "eng": "English",
    "fra": "French",
    "spa": "Spanish",
    "deu": "German",
    "por": "Portuguese",
}


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def load_official_splits(download: bool = True) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load the five selected language classes from WiLI's original splits."""
    paths = download_data() if download else {
        name: PROJECT_DIR / "data" / "raw" / name
        for name in ("x_train.txt", "y_train.txt", "x_test.txt", "y_test.txt")
    }
    train_texts, train_labels = _read_lines(paths["x_train.txt"]), _read_lines(paths["y_train.txt"])
    test_texts, test_labels = _read_lines(paths["x_test.txt"]), _read_lines(paths["y_test.txt"])
    if len(train_texts) != len(train_labels) or len(test_texts) != len(test_labels):
        raise ValueError("WiLI text and label files have different record counts")

    def select(texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
        selected = [(text, code) for text, code in zip(texts, labels) if code in LANGUAGE_NAMES]
        if not selected:
            raise ValueError("No selected WiLI language records were found")
        return [row[0] for row in selected], [row[1] for row in selected]

    selected_train_texts, selected_train_labels = select(train_texts, train_labels)
    selected_test_texts, selected_test_labels = select(test_texts, test_labels)
    return selected_train_texts, selected_train_labels, selected_test_texts, selected_test_labels


def split_training_data(
    texts: list[str], labels: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    return train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def candidate_models() -> dict[str, Any]:
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=4.0, max_iter=1000, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE),
    }


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("tfidf", make_vectorizer()), ("classifier", estimator)])


def classification_metrics(actual: list[str], predicted: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, average="macro", zero_division=0)),
    }
