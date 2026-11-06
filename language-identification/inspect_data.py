"""Inspect WiLI-2018 and save language counts and length summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.download_data import download_data

PROJECT_DIR = Path(__file__).resolve().parent
LANGUAGE_CODES = {"eng": "English", "fra": "French", "spa": "Spanish", "deu": "German", "por": "Portuguese"}


def read_text(path: Path) -> pd.Series:
    return pd.Series(path.read_text(encoding="utf-8").splitlines(), dtype="string")


def main() -> None:
    files = download_data()
    train_labels = pd.Series(files["y_train.txt"].read_text(encoding="utf-8").splitlines(), dtype="string")
    test_labels = pd.Series(files["y_test.txt"].read_text(encoding="utf-8").splitlines(), dtype="string")
    train_texts = read_text(files["x_train.txt"])
    test_texts = read_text(files["x_test.txt"])

    languages = {}
    for code, name in LANGUAGE_CODES.items():
        train_mask = train_labels == code
        test_mask = test_labels == code
        sample_lengths = train_texts[train_mask].str.len()
        languages[code] = {
            "name": name,
            "train_records": int(train_mask.sum()),
            "test_records": int(test_mask.sum()),
            "mean_train_characters": round(float(sample_lengths.mean()), 1),
            "min_train_characters": int(sample_lengths.min()),
        }

    summary = {
        "dataset": "WiLI-2018",
        "train_records_total": int(len(train_labels)),
        "test_records_total": int(len(test_labels)),
        "languages_selected": languages,
        "notes": "Official split is retained; model selection uses a validation holdout from the official training split.",
    }
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    names = [details["name"] for details in languages.values()]
    train_counts = [details["train_records"] for details in languages.values()]
    test_counts = [details["test_records"] for details in languages.values()]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    position = range(len(names))
    axis.bar([p - 0.18 for p in position], train_counts, width=0.36, label="Official train", color="#5271ff")
    axis.bar([p + 0.18 for p in position], test_counts, width=0.36, label="Official test", color="#28a88a")
    axis.set_xticks(list(position), names)
    axis.set_ylabel("Paragraphs")
    axis.set_title("WiLI records in the selected languages")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "language_distribution.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved data summary and chart in {results_dir}")


if __name__ == "__main__":
    main()
