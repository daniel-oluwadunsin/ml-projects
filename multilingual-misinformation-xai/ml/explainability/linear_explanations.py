"""Global coefficients and local contributions for Logistic Regression."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


def _class_coefficients(model: Pipeline, class_label: str) -> tuple[np.ndarray, float]:
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if class_label not in classes:
        raise ValueError(f"Unknown class {class_label!r}; available: {classes}")
    index = classes.index(class_label)
    if classifier.coef_.shape[0] == 1:
        # Binary sklearn LogisticRegression stores only the positive-class logit.
        row = classifier.coef_[0] if index == 1 else -classifier.coef_[0]
        intercept = classifier.intercept_[0] if index == 1 else -classifier.intercept_[0]
    else:
        row = classifier.coef_[index]
        intercept = classifier.intercept_[index]
    return row, float(intercept)


def global_feature_importance(
    model: Pipeline, class_label: str, *, top_n: int = 20
) -> dict[str, list[dict[str, float | str]]]:
    """Return highest positive and negative class-specific coefficient weights."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    vectorizer = model.named_steps["tfidf"]
    coefficients, _ = _class_coefficients(model, class_label)
    features = vectorizer.get_feature_names_out()
    order = np.argsort(coefficients)
    negative = [
        {"feature": str(features[index]), "coefficient": float(coefficients[index])}
        for index in order[:top_n]
    ]
    positive = [
        {"feature": str(features[index]), "coefficient": float(coefficients[index])}
        for index in order[-top_n:][::-1]
    ]
    return {"positive": positive, "negative": negative}


def local_feature_contributions(
    model: Pipeline, text: str, class_label: str, *, top_n: int = 20
) -> dict[str, Any]:
    """Explain one class logit as active TF-IDF values times fitted coefficients."""
    if not text.strip():
        raise ValueError("text must not be empty")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    vectorizer = model.named_steps["tfidf"]
    matrix = vectorizer.transform([text])
    coefficients, intercept = _class_coefficients(model, class_label)
    feature_indices = matrix.indices
    contributions = matrix.data * coefficients[feature_indices]
    terms = vectorizer.get_feature_names_out()[feature_indices]
    ordered = sorted(
        ({"feature": str(term), "contribution": float(value)} for term, value in zip(terms, contributions)),
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )[:top_n]
    return {
        "class_label": class_label,
        "method": "active TF-IDF value × class-specific Logistic Regression coefficient",
        "intercept": intercept,
        "feature_contributions": ordered,
        "sum_of_all_feature_contributions": float(contributions.sum()),
        "linear_score": float(contributions.sum() + intercept),
        "warning": "This explains a fitted class score, not factual correctness or causal reasoning.",
    }
