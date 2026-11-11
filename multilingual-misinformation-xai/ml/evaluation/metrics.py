"""Dataset-level and subgroup classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    labels = list(labels or sorted(set(y_true).union(y_pred)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_examples": len(y_true),
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def per_language_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    languages: Sequence[str | None],
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not (len(y_true) == len(y_pred) == len(languages)):
        raise ValueError("Targets, predictions, and languages must have equal lengths")
    results: dict[str, dict[str, Any]] = {}
    for language in sorted({value or "unknown" for value in languages}):
        indices = [i for i, value in enumerate(languages) if (value or "unknown") == language]
        results[language] = classification_metrics(
            [y_true[i] for i in indices], [y_pred[i] for i in indices], labels
        )
    return results


def probabilities_to_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Calculate per-row Shannon entropy; this is not a calibration measure."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or np.any(probabilities < 0):
        raise ValueError("probabilities must be a non-negative 2D array")
    row_sums = probabilities.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("each probability row must have positive mass")
    normalized = probabilities / row_sums[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(normalized > 0, normalized * np.log(normalized), 0.0)
    return -terms.sum(axis=1)


def probability_metrics(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str], *, bins: int = 10
) -> dict[str, float]:
    """Return multiclass Brier score and top-label expected calibration error.

    These are diagnostics for the supplied probabilities; they do not certify
    calibration. Lower scores are better for both returned metrics.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = list(labels)
    if probabilities.shape != (len(y_true), len(labels)):
        raise ValueError("probability matrix shape must match targets and labels")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must be non-negative and sum to one")
    label_to_index = {label: index for index, label in enumerate(labels)}
    try:
        target_indices = np.asarray([label_to_index[label] for label in y_true])
    except KeyError as exc:
        raise ValueError(f"Target label {exc.args[0]!r} is not in labels") from exc
    one_hot = np.eye(len(labels), dtype=float)[target_indices]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == target_indices
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
    return {"multiclass_brier_score": brier, "top_label_ece_10_bins": float(ece)}
