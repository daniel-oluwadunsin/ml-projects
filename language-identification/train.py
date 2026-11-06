"""Compare text classifiers with character-level TF-IDF on validation data."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from language_pipeline import (
    PROJECT_DIR,
    candidate_models,
    classification_metrics,
    load_official_splits,
    make_pipeline,
    split_training_data,
)


def compare_models() -> tuple[str, pd.DataFrame]:
    train_texts, train_labels, official_test_texts, official_test_labels = load_official_splits()
    x_train, x_validation, y_train, y_validation = split_training_data(train_texts, train_labels)

    rows = []
    for name, estimator in candidate_models().items():
        model = make_pipeline(estimator)
        model.fit(x_train, y_train)
        rows.append({"model": name, **classification_metrics(y_validation, model.predict(x_validation))})
    comparison = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy"], ascending=[False, False]
    )
    selected_name = str(comparison.iloc[0]["model"])
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / "validation_comparison.csv", index=False)
    (results_dir / "validation_selection.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "selection_metric": "highest validation macro F1, then highest accuracy",
                "languages": sorted(set(train_labels)),
                "selected_train_records": len(train_texts),
                "validation_records": len(x_validation),
                "official_test_records": len(official_test_texts),
                "official_test_labels": len(official_test_labels),
            },
            indent=2,
        )
        + "\n"
    )

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(comparison["model"], comparison["macro_f1"], color="#5271ff")
    axis.set(title="Validation macro F1 by model", ylabel="Macro F1", ylim=(0, 1.02))
    axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(results_dir / "validation_comparison.png", dpi=160)
    plt.close(fig)
    return selected_name, comparison


def main() -> None:
    selected, comparison = compare_models()
    print("Validation model comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by validation macro F1: {selected}")


if __name__ == "__main__":
    main()
