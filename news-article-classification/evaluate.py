"""Recompute metrics on AG News's untouched official test set."""

from __future__ import annotations

import json

import numpy as np
from joblib import load

from news_pipeline import (
    CLASS_LABELS,
    PROJECT_DIR,
    classification_metrics,
    load_official_test_set,
)


def main() -> None:
    model_path = PROJECT_DIR / "models" / "best_pipeline.joblib"
    if not model_path.exists():
        raise SystemExit("Trained model not found. Run python train.py first.")
    texts, labels = load_official_test_set()
    model = load(model_path)
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    metrics = classification_metrics(labels, predictions)
    class_indexes = {label: index for index, label in enumerate(model.classes_)}
    ordered_probabilities = probabilities[:, [class_indexes[label] for label in CLASS_LABELS]]
    one_hot = np.column_stack([np.asarray(labels) == label for label in CLASS_LABELS]).astype(float)
    metrics["multiclass_brier"] = float(
        np.mean(np.sum((ordered_probabilities - one_hot) ** 2, axis=1))
    )
    output_path = PROJECT_DIR / "results" / "test_metrics.json"
    stored = json.loads(output_path.read_text()) if output_path.exists() else {}
    stored.update(metrics)
    output_path.write_text(json.dumps(stored, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
