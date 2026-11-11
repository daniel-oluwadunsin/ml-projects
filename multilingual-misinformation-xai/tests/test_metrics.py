from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.metrics import (
    classification_metrics,
    per_language_metrics,
    probability_metrics,
    probabilities_to_entropy,
)


def test_metrics_match_manually_checkable_multiclass_example():
    result = classification_metrics(["a", "b", "c"], ["a", "a", "c"], ["a", "b", "c"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["per_class"]["b"]["recall"] == 0
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert result["n_examples"] == 3


def test_missing_class_is_reported_with_zero_support():
    result = classification_metrics(["a", "a"], ["a", "a"], ["a", "b"])
    assert result["per_class"]["b"]["support"] == 0
    assert result["per_class"]["b"]["f1"] == 0


def test_per_language_counts_do_not_merge_languages():
    results = per_language_metrics(
        ["yes", "no", "yes"], ["yes", "yes", "yes"], ["en", "yo", "en"], ["no", "yes"]
    )
    assert results["en"]["n_examples"] == 2
    assert results["yo"]["n_examples"] == 1


def test_entropy_requires_probability_mass():
    assert probabilities_to_entropy(np.array([[1.0, 0.0]])).tolist() == [0.0]
    with pytest.raises(ValueError, match="positive mass"):
        probabilities_to_entropy(np.array([[0.0, 0.0]]))


def test_probability_diagnostics_reward_correct_confident_predictions():
    targets = ["a", "b"]
    correct = probability_metrics(targets, np.array([[0.9, 0.1], [0.05, 0.95]]), ["a", "b"])
    uncertain = probability_metrics(targets, np.array([[0.5, 0.5], [0.5, 0.5]]), ["a", "b"])
    assert correct["multiclass_brier_score"] < uncertain["multiclass_brier_score"]
    assert correct["top_label_ece_10_bins"] > 0
