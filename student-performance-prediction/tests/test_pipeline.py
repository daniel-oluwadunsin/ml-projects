from __future__ import annotations

import numpy as np

from student_pipeline import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    candidate_models,
    load_dataset,
    make_pipeline,
    regression_metrics,
    split_dataset,
)


def test_training_features_exclude_all_grade_leakage() -> None:
    assert "G1" not in FEATURE_COLUMNS
    assert "G2" not in FEATURE_COLUMNS
    assert "G3" not in FEATURE_COLUMNS
    assert set(CATEGORICAL_COLUMNS).isdisjoint(NUMERIC_COLUMNS)


def test_dataset_and_deterministic_split() -> None:
    data = load_dataset(download=False)
    train_df, validation_df, test_df = split_dataset(data)

    assert len(data) == 395
    assert not data.isna().any().any()
    assert (len(train_df), len(validation_df), len(test_df)) == (276, 59, 60)
    assert set(train_df.index).isdisjoint(validation_df.index)
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(validation_df.index).isdisjoint(test_df.index)


def test_pipeline_predicts_real_held_out_rows() -> None:
    train_df, _, test_df = split_dataset(load_dataset(download=False))
    estimator = make_pipeline(candidate_models()["Linear Regression"])
    estimator.fit(train_df.drop(columns="G3"), train_df["G3"])
    predictions = estimator.predict(test_df.drop(columns="G3"))

    assert len(predictions) == len(test_df)
    assert np.isfinite(predictions).all()
    scores = regression_metrics(test_df["G3"], predictions)
    assert set(scores) == {"mae", "rmse", "r2"}


def test_grade_metric_values_are_correct() -> None:
    scores = regression_metrics(np.array([0, 10, 20]), np.array([1, 10, 18]))
    assert scores["mae"] == 1.0
    assert scores["rmse"] > 1.0
    assert -1.0 <= scores["r2"] <= 1.0
