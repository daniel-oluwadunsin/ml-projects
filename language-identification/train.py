"""Compare text classifiers with character-level TF-IDF on validation data."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import classification_report, confusion_matrix

from language_pipeline import (
    PROJECT_DIR,
    LANGUAGE_NAMES,
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


def finalize_model(selected_name: str) -> dict[str, float]:
    """Fit a calibrated selected model on official train and evaluate official test."""
    train_texts, train_labels, test_texts, test_labels = load_official_splits()
    base_pipeline = make_pipeline(candidate_models()[selected_name])
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5,
        n_jobs=1,
    )
    calibrated_model.fit(train_texts, train_labels)
    predictions = calibrated_model.predict(test_texts)
    probabilities = calibrated_model.predict_proba(test_texts)
    metrics = classification_metrics(test_labels, predictions)
    labels = list(LANGUAGE_NAMES)
    label_indexes = {label: index for index, label in enumerate(calibrated_model.classes_)}
    ordered_probabilities = probabilities[:, [label_indexes[label] for label in labels]]
    one_hot = np.column_stack([np.asarray(test_labels) == label for label in labels]).astype(float)
    metrics["multiclass_brier"] = float(np.mean(np.sum((ordered_probabilities - one_hot) ** 2, axis=1)))

    results_dir = PROJECT_DIR / "results"
    models_dir = PROJECT_DIR / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    dump(calibrated_model, models_dir / "best_pipeline.joblib", compress=3)

    metadata = json.loads((results_dir / "validation_selection.json").read_text())
    metadata.update(
        {
            "selected_model": selected_name,
            "calibration": "5-fold sigmoid calibration on the complete official training subset",
            "calibration_method": "sigmoid",
            **metrics,
        }
    )
    (results_dir / "test_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")

    matrix = confusion_matrix(test_labels, predictions, labels=labels)
    matrix_frame = pd.DataFrame(
        matrix,
        index=[LANGUAGE_NAMES[label] for label in labels],
        columns=[LANGUAGE_NAMES[label] for label in labels],
    )
    matrix_frame.to_csv(results_dir / "confusion_matrix.csv", index_label="actual_language")
    report = classification_report(
        test_labels,
        predictions,
        labels=labels,
        target_names=[LANGUAGE_NAMES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(results_dir / "classification_report.csv")
    pd.DataFrame(
        {
            "actual_code": test_labels,
            "predicted_code": predictions,
            "confidence": probabilities.max(axis=1),
        }
    ).to_csv(results_dir / "test_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    image = axes[0].imshow(matrix, interpolation="nearest", cmap="Blues")
    axes[0].set(
        title="Official test confusion matrix",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=[LANGUAGE_NAMES[label] for label in labels],
        yticklabels=[LANGUAGE_NAMES[label] for label in labels],
        xlabel="Predicted",
        ylabel="Actual",
    )
    axes[0].tick_params(axis="x", rotation=30)
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    for index, label in enumerate(labels):
        actual_binary = (np.asarray(test_labels) == label).astype(int)
        fraction_positive, mean_predicted = calibration_curve(
            actual_binary,
            ordered_probabilities[:, index],
            n_bins=10,
            strategy="quantile",
        )
        axes[1].plot(mean_predicted, fraction_positive, marker="o", label=LANGUAGE_NAMES[label])
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#555", label="Ideal")
    axes[1].set(
        title="One-vs-rest calibration on official test",
        xlabel="Mean predicted probability",
        ylabel="Observed fraction positive",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "test_evaluation.png", dpi=160)
    plt.close(fig)
    return metrics


def main() -> None:
    selected, comparison = compare_models()
    print("Validation model comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by validation macro F1: {selected}")
    metrics = finalize_model(selected)
    print("\nOfficial test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved calibrated pipeline: {PROJECT_DIR / 'models' / 'best_pipeline.joblib'}")


if __name__ == "__main__":
    main()
