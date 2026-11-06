"""Data preparation and model helpers for the student grade project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from data.download_data import download_data
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TARGET = "G3"
EXCLUDED_FEATURES = {"G1", "G2", TARGET}
CATEGORICAL_COLUMNS = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
]
NUMERIC_COLUMNS = ["age", "absences"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_dataset(download: bool = True) -> pd.DataFrame:
    """Load the UCI Mathematics data, downloading it when requested."""
    path = PROJECT_DIR / "data" / "raw" / "student-mat.csv"
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                "Dataset missing. Run python data/download_data.py first."
            )
        path = download_data()
    data = pd.read_csv(path, sep=";")
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return data[FEATURE_COLUMNS + [TARGET]].copy()


def split_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Create deterministic 70/15/15 train, validation, and test splits."""
    train_df, temporary_df = train_test_split(
        data, test_size=0.30, random_state=RANDOM_STATE
    )
    validation_df, test_df = train_test_split(
        temporary_df, test_size=0.50, random_state=RANDOM_STATE
    )
    return train_df.copy(), validation_df.copy(), test_df.copy()


def make_pipeline(estimator: Any) -> Pipeline:
    """Build preprocessing and regression as a single leakage-safe pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def candidate_models() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def regression_metrics(
    actual: pd.Series | np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def save_feature_options(data: pd.DataFrame) -> Path:
    """Persist UI choices inferred from training data, excluding target leakage."""
    options = {
        column: sorted(data[column].dropna().unique().tolist(), key=str)
        for column in CATEGORICAL_COLUMNS
    }
    output = PROJECT_DIR / "results" / "feature_options.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(options, indent=2) + "\n")
    return output
