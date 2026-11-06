from __future__ import annotations

import numpy as np
from joblib import load

from news_pipeline import (
    CLASS_LABELS,
    PROJECT_DIR,
    classification_metrics,
    load_official_test_set,
    load_training_set,
    split_training_data,
)


def test_official_training_data_is_balanced_and_validation_is_stratified() -> None:
    texts, labels = load_training_set(download=False)
    assert len(texts) == len(labels) == 120_000
    assert set(labels) == set(CLASS_LABELS)
    for category in CLASS_LABELS:
        assert labels.count(category) == 30_000

    x_train, x_validation, y_train, y_validation = split_training_data(texts, labels)
    assert len(x_train) == len(y_train) == 96_000
    assert len(x_validation) == len(y_validation) == 24_000
    for category in CLASS_LABELS:
        assert y_train.count(category) == 24_000
        assert y_validation.count(category) == 6_000


def test_official_test_split_is_balanced() -> None:
    texts, labels = load_official_test_set(download=False)
    assert len(texts) == len(labels) == 7_600
    for category in CLASS_LABELS:
        assert labels.count(category) == 1_900


def test_saved_calibrated_model_returns_probabilities_for_real_articles() -> None:
    model = load(PROJECT_DIR / "models" / "best_pipeline.joblib")
    texts, labels = load_official_test_set(download=False)
    samples = [next(text for text, label in zip(texts, labels) if label == category) for category in CLASS_LABELS]
    probabilities = model.predict_proba(samples)

    assert set(model.predict(samples)).issubset(set(CLASS_LABELS))
    assert probabilities.shape == (4, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(probabilities).all()


def test_classification_metrics_are_finite_and_bounded() -> None:
    metrics = classification_metrics(
        ["World", "Sports", "Business", "Sci/Tech"],
        ["World", "Sports", "World", "Sci/Tech"],
    )
    assert metrics["accuracy"] == 0.75
    for metric in ("macro_f1", "macro_precision", "macro_recall"):
        assert 0.0 <= metrics[metric] <= 1.0
