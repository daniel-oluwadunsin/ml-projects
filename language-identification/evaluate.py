"""Recompute metrics on WiLI's untouched official test split."""

from __future__ import annotations

import json

import numpy as np
from joblib import load

from language_pipeline import (
    LANGUAGE_NAMES,
    PROJECT_DIR,
    classification_metrics,
    load_official_splits,
)


def main() -> None:
    model_path = PROJECT_DIR / "models" / "best_pipeline.joblib"
    if not model_path.exists():
        raise SystemExit("Trained model not found. Run python train.py first.")
    _, _, test_texts, test_labels = load_official_splits()
    model = load(model_path)
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    metrics = classification_metrics(test_labels, predictions)
    labels = list(LANGUAGE_NAMES)
    indexes = {label: index for index, label in enumerate(model.classes_)}
    ordered_probabilities = probabilities[:, [indexes[label] for label in labels]]
    one_hot = np.column_stack([np.asarray(test_labels) == label for label in labels]).astype(float)
    metrics["multiclass_brier"] = float(np.mean(np.sum((ordered_probabilities - one_hot) ** 2, axis=1)))
    output_path = PROJECT_DIR / "results" / "test_metrics.json"
    stored = json.loads(output_path.read_text()) if output_path.exists() else {}
    stored.update(metrics)
    output_path.write_text(json.dumps(stored, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
