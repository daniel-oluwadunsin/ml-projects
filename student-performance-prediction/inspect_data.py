"""Explore the UCI Mathematics data and save compact summary artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
from data.download_data import download_data  # noqa: E402


def main() -> None:
    csv_path = download_data()
    data = pd.read_csv(csv_path, sep=";")
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {
        "dataset": "UCI Student Performance - Mathematics",
        "source_file": "student-mat.csv",
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_cells": int(data.isna().sum().sum()),
        "target": "G3",
        "target_min": int(data["G3"].min()),
        "target_max": int(data["G3"].max()),
        "target_mean": round(float(data["G3"].mean()), 3),
        "target_std": round(float(data["G3"].std()), 3),
        "grade_correlations": {
            col: round(float(data[col].corr(data["G3"])), 3)
            for col in ("G1", "G2")
        },
        "excluded_from_model": ["G1", "G2"],
    }
    (results_dir / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data["G3"], bins=21, color="#5271ff", edgecolor="white")
    axes[0].set(title="Final Mathematics grade (G3)", xlabel="Grade (0–20)", ylabel="Students")
    axes[1].scatter(data["studytime"], data["G3"], alpha=0.55, color="#28a88a")
    axes[1].set(title="Study time and final grade", xlabel="Weekly study time category", ylabel="G3")
    fig.tight_layout()
    fig.savefig(results_dir / "data_exploration.png", dpi=160)
    plt.close(fig)

    print(data.head().to_string(index=False))
    print(f"\nShape: {data.shape}; missing cells: {int(data.isna().sum().sum())}")
    print(f"G3 mean: {data.G3.mean():.2f}; range: {data.G3.min()}–{data.G3.max()}")
    print(f"Saved summary and chart in {results_dir}")


if __name__ == "__main__":
    main()
