"""Compare AG News text classifiers on a validation split of official training data."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from news_pipeline import (
    CLASS_LABELS,
    PROJECT_DIR,
    candidate_models,
    classification_metrics,
    load_training_set,
    make_vectorizer,
    split_training_data,
)


def compare_models() -> tuple[str, pd.DataFrame]:
    texts, labels = load_training_set()
    x_train, x_validation, y_train, y_validation = split_training_data(texts, labels)

    # Fit vocabulary only on the training partition; reuse sparse vectors for fair comparison.
    vectorizer = make_vectorizer()
    x_train_matrix = vectorizer.fit_transform(x_train)
    x_validation_matrix = vectorizer.transform(x_validation)

    rows = []
    for name, estimator in candidate_models().items():
        estimator.fit(x_train_matrix, y_train)
        rows.append(
            {
                "model": name,
                **classification_metrics(y_validation, estimator.predict(x_validation_matrix)),
            }
        )
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
                "validation_rows": len(y_validation),
                "training_rows_used_for_comparison": len(y_train),
                "official_training_rows": len(texts),
                "class_labels": CLASS_LABELS,
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
    print("Validation comparison (official test data not read):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by validation macro F1: {selected}")


if __name__ == "__main__":
    main()
