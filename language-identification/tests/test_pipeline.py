from __future__ import annotations

import numpy as np
from joblib import load

from language_pipeline import (
    LANGUAGE_NAMES,
    PROJECT_DIR,
    classification_metrics,
    load_official_splits,
    split_training_data,
)


def test_selected_official_splits_are_balanced_and_separate() -> None:
    train_texts, train_labels, test_texts, test_labels = load_official_splits(download=False)
    assert len(train_texts) == len(train_labels) == 2500
    assert len(test_texts) == len(test_labels) == 2500
    assert set(train_labels) == set(LANGUAGE_NAMES)
    assert set(test_labels) == set(LANGUAGE_NAMES)
    for code in LANGUAGE_NAMES:
        assert train_labels.count(code) == 500
        assert test_labels.count(code) == 500


def test_validation_split_is_stratified() -> None:
    train_texts, train_labels, _, _ = load_official_splits(download=False)
    x_train, x_validation, y_train, y_validation = split_training_data(train_texts, train_labels)
    assert len(x_train) == len(y_train) == 2000
    assert len(x_validation) == len(y_validation) == 500
    for code in LANGUAGE_NAMES:
        assert y_train.count(code) == 400
        assert y_validation.count(code) == 100


def test_saved_calibrated_model_predicts_real_text_with_probabilities() -> None:
    model = load(PROJECT_DIR / "models" / "best_pipeline.joblib")
    _, _, test_texts, test_labels = load_official_splits(download=False)
    sample_texts = [next(text for text, label in zip(test_texts, test_labels) if label == code) for code in LANGUAGE_NAMES]
    probabilities = model.predict_proba(sample_texts)

    assert set(model.predict(sample_texts)).issubset(set(LANGUAGE_NAMES))
    assert probabilities.shape == (len(LANGUAGE_NAMES), len(LANGUAGE_NAMES))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(probabilities).all()


def test_classification_metrics_are_macro_averaged() -> None:
    metrics = classification_metrics(["eng", "fra", "fra"], ["eng", "fra", "eng"])
    assert metrics["accuracy"] == 2 / 3
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["macro_precision"] <= 1
    assert 0 <= metrics["macro_recall"] <= 1
