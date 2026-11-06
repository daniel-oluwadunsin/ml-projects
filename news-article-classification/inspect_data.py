"""Inspect the official AG News training distribution and text lengths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.download_data import download_data

PROJECT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_training_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=["class_index", "title", "description"],
        dtype={"class_index": "int8"},
    )
    if set(frame["class_index"].unique()) != set(CLASS_NAMES):
        raise ValueError("AG News training data must contain the four expected class indexes")
    return frame


def main() -> None:
    paths = download_data()
    train = load_training_data(paths["train.csv"])
    counts = train["class_index"].value_counts().sort_index()
    summary = {
        "dataset": "AG News",
        "split_used_for_inspection": "official training data only",
        "train_records": int(len(train)),
        "training_missing_cells": int(train.isna().sum().sum()),
        "category_counts": {
            CLASS_NAMES[int(index)]: int(count) for index, count in counts.items()
        },
        "mean_title_characters": round(float(train["title"].str.len().mean()), 1),
        "mean_description_characters": round(float(train["description"].str.len().mean()), 1),
    }
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    distribution = pd.DataFrame(
        {"category": [CLASS_NAMES[int(index)] for index in counts.index], "training_records": counts.values}
    )
    distribution.to_csv(results_dir / "training_distribution.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(distribution["category"], distribution["training_records"], color=["#5271ff", "#28a88a", "#ef9f43", "#8365d4"])
    axis.bar_label(bars, fmt="%d", padding=3)
    axis.set(title="AG News category distribution · official training set", ylabel="Articles")
    fig.tight_layout()
    fig.savefig(results_dir / "training_distribution.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved training-only summary and chart in {results_dir}")


if __name__ == "__main__":
    main()
