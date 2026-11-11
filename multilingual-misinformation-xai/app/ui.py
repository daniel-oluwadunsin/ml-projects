"""Shared local artifact loading and UI provenance helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts/models/fever-tfidf-logreg/model.joblib"
MODEL_METADATA_PATH = ROOT / "artifacts/models/fever-tfidf-logreg/metadata.json"
METRICS_PATH = ROOT / "results/fever/latest/metrics.json"
RUN_METADATA_PATH = ROOT / "results/fever/latest/metadata.json"
PREDICTIONS_PATH = ROOT / "results/fever/latest/predictions.jsonl"
DATA_SUMMARY_PATH = ROOT / "results/fever/data_summary.json"
TRAIN_PATH = ROOT / "data/raw/fever_train.jsonl"
DEV_PATH = ROOT / "data/raw/fever_dev.jsonl"


@st.cache_resource
def load_baseline():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path_string: str):
    path = Path(path_string)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_prediction_rows(path_string: str):
    path = Path(path_string)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def show_missing_model() -> None:
    st.info(
        "The local FEVER baseline has not been trained yet. Download the official FEVER train "
        "and labelled development JSONL files, then run "
        "`uv run python scripts/run_baseline.py --config configs/baseline.yaml`."
    )


def show_task_notice() -> None:
    st.caption(
        "Task: English claim-only classification of FEVER dataset labels. A prediction is not "
        "independent verification and does not establish whether a real-world claim is true."
    )


def read_docs(relative_path: str) -> str:
    path = ROOT / relative_path
    if not path.is_file():
        return f"Documentation file missing: {relative_path}"
    return path.read_text(encoding="utf-8")
