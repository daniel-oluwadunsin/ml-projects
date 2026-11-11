"""Recompute aggregate metrics from a local run's label-only predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.evaluation.metrics import classification_metrics, probability_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/fever/latest"))
    args = parser.parse_args()
    predictions = [
        json.loads(line)
        for line in (args.run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = sorted({row["true_label"] for row in predictions})
    metrics = classification_metrics(
        [row["true_label"] for row in predictions],
        [row["predicted_label"] for row in predictions],
        labels,
    )
    probabilities = np.asarray(
        [[row["probabilities"][label] for label in labels] for row in predictions], dtype=float
    )
    metrics["probability_diagnostics"] = probability_metrics(
        [row["true_label"] for row in predictions], probabilities, labels
    )
    print(json.dumps(metrics, indent=2))
    stored = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))["heldout"]
    if metrics != stored:
        raise SystemExit("Recomputed metrics did not match metrics.json")
    print("Stored metrics verified against local prediction rows.")


if __name__ == "__main__":
    main()
