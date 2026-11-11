"""Select a TF-IDF Logistic Regression baseline on FEVER validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from ml.data.loaders import load_fever_jsonl
from ml.data.splits import grouped_train_validation_indices
from ml.evaluation.metrics import classification_metrics, probability_metrics
from ml.models.baseline import build_baseline

LABELS = ["NOT_ENOUGH_INFORMATION", "REFUTED", "SUPPORTED"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    parser.add_argument("--max-train-rows", type=int, default=None, help="Exploratory subset only")
    parser.add_argument("--run-id", default="latest")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    train_path = Path(config["train_file"])
    dev_path = Path(config["dev_file"])
    train_records = load_fever_jsonl(train_path, "train", include_evidence=False)
    dev_records = load_fever_jsonl(dev_path, "official_labelled_dev")
    source_train_rows = len(train_records)
    cross_partition = {row.group_id for row in train_records} & {
        row.group_id for row in dev_records
    }
    overlap_rows = sum(row.group_id in cross_partition for row in train_records)
    if cross_partition:
        train_records = [row for row in train_records if row.group_id not in cross_partition]
        print(
            f"Removed {len(cross_partition)} duplicate claim groups from training; "
            "held-out dev unchanged."
        )

    if args.max_train_rows:
        train_records = train_records[: args.max_train_rows]
    texts = np.asarray([row.claim for row in train_records], dtype=object)
    labels = np.asarray([row.label for row in train_records], dtype=object)
    groups = np.asarray([row.group_id for row in train_records], dtype=object)
    fit_indices, validation_indices = grouped_train_validation_indices(
        labels.tolist(),
        groups.tolist(),
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    if set(labels[fit_indices]) != set(LABELS):
        raise RuntimeError("Training partition does not contain all canonical FEVER classes")

    shared = {
        "max_features": int(config["max_features"]),
        "min_df": int(config["min_df"]),
        "class_weight": config["class_weight"],
        "max_iter": int(config["max_iter"]),
        "seed": int(config["seed"]),
    }
    candidates: list[dict[str, Any]] = []
    for candidate in config["selection_candidates"]:
        candidate_config = {**shared, **candidate}
        model = build_baseline(
            analyzer=candidate_config["analyzer"],
            ngram_range=tuple(candidate_config["ngram_range"]),
            c=float(candidate_config["C"]),
            **shared,
        )
        start = time.perf_counter()
        model.fit(texts[fit_indices].tolist(), labels[fit_indices].tolist())
        fit_seconds = time.perf_counter() - start
        validation_probabilities = model.predict_proba(texts[validation_indices].tolist())
        predicted = model.classes_[validation_probabilities.argmax(axis=1)]
        validation_metrics = classification_metrics(
            labels[validation_indices].tolist(), predicted.tolist(), LABELS
        )
        validation_metrics["probability_diagnostics"] = probability_metrics(
            labels[validation_indices].tolist(), validation_probabilities, list(model.classes_)
        )
        candidates.append(
            {
                "name": candidate_config["name"],
                "configuration": candidate_config,
                "fit_seconds": fit_seconds,
                "validation": validation_metrics,
            }
        )
        print(
            f"{candidate_config['name']}: validation macro-F1={validation_metrics['macro_f1']:.4f}"
        )

    if not candidates:
        raise RuntimeError("No model candidate was evaluated")
    selected_name = max(candidates, key=lambda row: row["validation"]["macro_f1"])["name"]
    best_config = next(row["configuration"] for row in candidates if row["name"] == selected_name)
    # Refit from scratch on all official training rows after validation selection.
    final_model = build_baseline(
        analyzer=best_config["analyzer"],
        ngram_range=tuple(best_config["ngram_range"]),
        c=float(best_config["C"]),
        **shared,
    )
    final_fit_start = time.perf_counter()
    final_model.fit([row.claim for row in train_records], [row.label for row in train_records])
    final_fit_seconds = time.perf_counter() - final_fit_start

    dev_texts = [row.claim for row in dev_records]
    inference_start = time.perf_counter()
    dev_probabilities = final_model.predict_proba(dev_texts)
    dev_predictions = final_model.classes_[dev_probabilities.argmax(axis=1)]
    inference_seconds = time.perf_counter() - inference_start
    heldout_metrics = classification_metrics(
        [row.label for row in dev_records], dev_predictions.tolist(), LABELS
    )
    heldout_metrics["probability_diagnostics"] = probability_metrics(
        [row.label for row in dev_records], dev_probabilities, list(final_model.classes_)
    )
    result_dir = Path(config["results_dir"]) / args.run_id
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "model.joblib", compress=3)
    prediction_path = result_dir / "predictions.jsonl"
    result_dir.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row, predicted, probabilities in zip(
            dev_records, dev_predictions, dev_probabilities, strict=True
        ):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row.sample_id,
                        "dataset": row.dataset,
                        "source_split": row.source_split,
                        "language": row.language,
                        "true_label": row.label,
                        "original_label": row.original_label,
                        "predicted_label": str(predicted),
                        "probabilities": {
                            label: float(probabilities[index])
                            for index, label in enumerate(final_model.classes_)
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    run_metadata = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "FEVER",
        "task": "English claim-only label classification",
        "selected_candidate": selected_name,
        "selected_configuration": best_config,
        "label_mapping": {
            "SUPPORTS": "SUPPORTED",
            "REFUTES": "REFUTED",
            "NOT ENOUGH INFO": "NOT_ENOUGH_INFORMATION",
        },
        "language_coverage": ["en"],
        "selection_metric": "validation macro_f1",
        "seed": int(config["seed"]),
        "split_strategy": (
            "StratifiedGroupKFold (7 folds, first fold used for validation) on NFC, "
            "whitespace-normalized, casefolded claim text"
        ),
        "source_train_rows": len(train_records),
        "source_train_rows_before_overlap_filter": source_train_rows,
        "internal_fit_rows": len(fit_indices),
        "internal_validation_rows": len(validation_indices),
        "heldout_partition": "official labelled development set; not the blind test",
        "heldout_rows": len(dev_records),
        "heldout_sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row.sample_id for row in dev_records)).encode("utf-8")
        ).hexdigest(),
        "source_partition_duplicate_groups_removed_from_training": len(cross_partition),
        "source_partition_rows_removed_from_training": overlap_rows,
        "train_file_sha256": _sha256(train_path),
        "heldout_file_sha256": _sha256(dev_path),
        "library_versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "final_fit_seconds": final_fit_seconds,
            "heldout_prediction_seconds": inference_seconds,
            "heldout_milliseconds_per_example": 1000 * inference_seconds / len(dev_records),
        },
        "exploratory_subset": args.max_train_rows,
    }
    _write_json(
        result_dir / "metrics.json",
        {
            "dataset": "FEVER",
            "task": "English claim-only label classification",
            "validation_candidates": candidates,
            "selected_candidate": selected_name,
            "heldout": heldout_metrics,
            "languages": {"en": heldout_metrics},
            "runtime": run_metadata["runtime"],
            "status": "measured",
        },
    )
    _write_json(result_dir / "metadata.json", run_metadata)
    _write_json(model_dir / "metadata.json", run_metadata)
    _write_json(result_dir / "configuration.json", config)
    print(json.dumps({"heldout": heldout_metrics, "metadata": run_metadata}, indent=2))
    print(f"Saved model to {model_dir / 'model.joblib'}")
    print(f"Saved compact results to {result_dir}")


if __name__ == "__main__":
    main()
