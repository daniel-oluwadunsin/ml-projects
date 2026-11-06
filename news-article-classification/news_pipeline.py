"""Data, feature, model, and metric helpers for AG News classification."""

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
CLASS_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}
CLASS_LABELS = list(CLASS_NAMES.values())
RAW_COLUMNS = ["class_index", "title", "description"]


def _read_split(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, header=None, names=RAW_COLUMNS, dtype={"class_index": "int8"})
    if set(data["class_index"].unique()) != set(CLASS_NAMES):
        raise ValueError(f"{path.name} must contain all four expected classes")
    if data[["title", "description"]].isna().any().any():
        raise ValueError(f"{path.name} contains missing text")
    return data


def prepare_text(data: pd.DataFrame) -> list[str]:
    """Combine headline and description, retaining original words and punctuation."""
    return (
        data["title"].fillna("").astype(str).str.strip()
        + " . "
        + data["description"].fillna("").astype(str).str.strip()
    ).tolist()


def load_training_set(download: bool = True) -> tuple[list[str], list[str]]:
    paths = download_data() if download else {
        "train.csv": PROJECT_DIR / "data" / "raw" / "train.csv"
    }
    data = _read_split(paths["train.csv"])
    return prepare_text(data), [CLASS_NAMES[int(value)] for value in data["class_index"]]


def load_official_test_set(download: bool = True) -> tuple[list[str], list[str]]:
    """Read the official test set; call only for final evaluation or app checks."""
    paths = download_data() if download else {
        "test.csv": PROJECT_DIR / "data" / "raw" / "test.csv"
    }
    data = _read_split(paths["test.csv"])
    return prepare_text(data), [CLASS_NAMES[int(value)] for value in data["class_index"]]


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
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=250_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def candidate_models() -> dict[str, Any]:
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(
            C=4.0, max_iter=300, tol=1e-3, random_state=RANDOM_STATE
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
