"""Compare candidate regressors using a leakage-safe train/validation split."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.inspection import permutation_importance

from student_pipeline import (
    PROJECT_DIR,
    candidate_models,
    load_dataset,
    make_pipeline,
    regression_metrics,
    save_feature_options,
    split_dataset,
)


def compare_models() -> tuple[str, pd.DataFrame]:
    data = load_dataset()
    train_df, validation_df, test_df = split_dataset(data)
    x_train, y_train = train_df.drop(columns="G3"), train_df["G3"]
    x_validation, y_validation = validation_df.drop(columns="G3"), validation_df["G3"]

    comparison_rows = []
    for name, estimator in candidate_models().items():
        model = make_pipeline(estimator)
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        comparison_rows.append({"model": name, **regression_metrics(y_validation, prediction)})

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["mae", "rmse", "r2"], ascending=[True, True, False]
    )
    selected_name = str(comparison.iloc[0]["model"])
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / "validation_comparison.csv", index=False)
    (results_dir / "validation_selection.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "selection_metric": "lowest validation MAE, then lowest validation RMSE, then highest validation R2",
                "split_sizes": {
                    "train": len(train_df),
                    "validation": len(validation_df),
                    "test": len(test_df),
                },
            },
            indent=2,
        )
        + "\n"
    )
    save_feature_options(data)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(comparison["model"], comparison["mae"], color="#5271ff")
    axis.set(title="Validation MAE by model", ylabel="MAE (grade points)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(results_dir / "validation_comparison.png", dpi=160)
    plt.close(fig)
    return selected_name, comparison


def finalize_model(selected_name: str) -> dict[str, float]:
    """Refit the selected estimator and evaluate once on the held-out test data."""
    data = load_dataset()
    train_df, validation_df, test_df = split_dataset(data)
    refit_data = pd.concat([train_df, validation_df], ignore_index=True)
    final_model = make_pipeline(candidate_models()[selected_name])
    final_model.fit(refit_data.drop(columns="G3"), refit_data["G3"])

    test_x = test_df.drop(columns="G3")
    test_y = test_df["G3"]
    test_prediction = final_model.predict(test_x)
    metrics = regression_metrics(test_y, test_prediction)
    results_dir = PROJECT_DIR / "results"
    models_dir = PROJECT_DIR / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    dump(final_model, models_dir / "best_pipeline.joblib", compress=3)

    metadata = json.loads((results_dir / "validation_selection.json").read_text())
    metadata.update({"selected_model": selected_name, **metrics})
    (results_dir / "test_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame({"actual_G3": test_y, "predicted_G3": test_prediction}).to_csv(
        results_dir / "test_predictions.csv", index=False
    )

    importance = permutation_importance(
        final_model,
        test_x,
        test_y,
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance_frame = pd.DataFrame(
        {"feature": test_x.columns, "importance": importance.importances_mean}
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(results_dir / "feature_importance.csv", index=False)

    comparison = pd.read_csv(results_dir / "validation_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(comparison["model"], comparison["mae"], color="#5271ff")
    axes[0].set(title="Validation MAE by model", ylabel="MAE (grade points)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].scatter(test_y, test_prediction, alpha=0.65, color="#28a88a")
    axes[1].plot([0, 20], [0, 20], linestyle="--", color="#555")
    axes[1].set(
        title="Held-out test predictions",
        xlabel="Actual G3",
        ylabel="Predicted G3",
        xlim=(0, 20),
        ylim=(0, 20),
    )
    fig.tight_layout()
    fig.savefig(results_dir / "model_results.png", dpi=160)
    plt.close(fig)

    top_importance = importance_frame.head(10).sort_values("importance")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top_importance["feature"], top_importance["importance"], color="#28a88a")
    axis.set(title="Permutation importance on held-out test data", xlabel="Increase in MAE when shuffled")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=160)
    plt.close(fig)
    return metrics


def main() -> None:
    selected, comparison = compare_models()
    print("Validation model comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected by validation: {selected}")
    metrics = finalize_model(selected)
    print("\nHeld-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {PROJECT_DIR / 'models' / 'best_pipeline.joblib'}")


if __name__ == "__main__":
    main()
