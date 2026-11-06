"""Compare candidate regressors using a leakage-safe train/validation split."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
    train_df, validation_df, _ = split_dataset(data)
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
                    "test": len(split_dataset(data)[2]),
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


def main() -> None:
    selected, comparison = compare_models()
    print("Validation model comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected by validation: {selected}")


if __name__ == "__main__":
    main()
