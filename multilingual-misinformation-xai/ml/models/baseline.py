"""TF-IDF and Logistic Regression baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline(
    *,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 150_000,
    min_df: int | float = 2,
    c: float = 2.0,
    class_weight: str | None = "balanced",
    max_iter: int = 1_500,
    seed: int = 42,
) -> Pipeline:
    """Build the complete preprocessing/model pipeline for a claim-only task."""
    if analyzer not in {"word", "char", "char_wb"}:
        raise ValueError("analyzer must be word, char, or char_wb")
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=min_df,
                    lowercase=True,
                    strip_accents=None,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def predict_with_probabilities(model: Pipeline, texts: Sequence[str]):
    """Return labels and class probabilities from the fitted pipeline."""
    if len(texts) == 0 or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Provide one or more non-empty claim strings")
    return model.predict(texts), model.predict_proba(texts)
