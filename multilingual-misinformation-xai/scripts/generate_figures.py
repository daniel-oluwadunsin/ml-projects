"""Generate static figures from measured aggregate-only FEVER artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/fever/latest"))
    parser.add_argument("--summary", type=Path, default=Path("results/fever/data_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/fever"))
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = metrics["heldout"]["labels"]
    readable_labels = [label.replace("_", " ") for label in labels]

    matrix = np.asarray(metrics["heldout"]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), readable_labels, rotation=20, ha="right")
    axis.set_yticks(range(len(labels)), readable_labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("FEVER labelled development confusion matrix")
    figure.colorbar(image, ax=axis, label="Number of claims")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion-matrix.png", dpi=160)
    plt.close(figure)

    per_class = metrics["heldout"]["per_class"]
    figure, axis = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(labels))
    width = 0.24
    for offset, key, display in [
        (-width, "precision", "Precision"),
        (0, "recall", "Recall"),
        (width, "f1", "F1"),
    ]:
        axis.bar(
            x_positions + offset, [per_class[label][key] for label in labels], width, label=display
        )
    axis.set_xticks(x_positions, readable_labels, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Per-class held-out metrics")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "per-class-metrics.png", dpi=160)
    plt.close(figure)

    train_counts = summary["train"]["label_counts"]
    dev_counts = summary["official_labelled_dev"]["label_counts"]
    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.36
    axis.bar(
        x_positions - width / 2,
        [train_counts[label] for label in labels],
        width,
        label="Train after cross-partition deduplication",
    )
    axis.bar(
        x_positions + width / 2,
        [dev_counts[label] for label in labels],
        width,
        label="Labelled development",
    )
    axis.set_xticks(x_positions, readable_labels, rotation=20, ha="right")
    axis.set_ylabel("Examples")
    axis.set_title("FEVER label distribution")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "label-distribution.png", dpi=160)
    plt.close(figure)
    print(f"Generated figures from measured results under {args.output_dir}")


if __name__ == "__main__":
    main()
