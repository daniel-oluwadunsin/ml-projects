"""Recompute final regression metrics on the deterministic held-out test split."""

from __future__ import annotations

import json

from joblib import load
from student_pipeline import (
    PROJECT_DIR,
    load_dataset,
    regression_metrics,
    split_dataset,
)


def main() -> None:
    model_path = PROJECT_DIR / "models" / "best_pipeline.joblib"

    if not model_path.exists():
        raise SystemExit("Trained model not found. Run python train.py first.")

    _, _, test_df = split_dataset(load_dataset())

    model = load(model_path)
    actual = test_df["G3"]
    prediction = model.predict(test_df.drop(columns="G3"))
    metrics = regression_metrics(actual, prediction)
    output_path = PROJECT_DIR / "results" / "test_metrics.json"
    stored = json.loads(output_path.read_text()) if output_path.exists() else {}
    stored.update(metrics)
    output_path.write_text(json.dumps(stored, indent=2) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
