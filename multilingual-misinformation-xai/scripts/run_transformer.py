"""Fine-tune XLM-R on an explicitly mapped dataset configuration."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ml.data.loaders import load_authorized_hf_split
from ml.data.splits import grouped_train_validation_indices
from ml.evaluation.metrics import classification_metrics, probability_metrics
from ml.models.transformer import available_device


def _evidence_text(evidence: Any) -> str:
    if isinstance(evidence, str):
        return evidence.strip()
    if isinstance(evidence, (list, tuple)) and all(isinstance(item, str) for item in evidence):
        return " ".join(item.strip() for item in evidence if item.strip())
    raise ValueError(
        "Evidence mode requires a verified passage field containing text (or a list of text "
        "passages). Inspect and adapt structured source evidence explicitly before training."
    )


def _dataset(records, *, text_mode: str, tokenizer, max_length: int, label_to_id: dict[str, int]):
    from datasets import Dataset

    texts = []
    for record in records:
        claim = record.claim
        if text_mode == "claim_plus_evidence":
            if record.evidence is None:
                raise ValueError(f"Evidence is missing for sample {record.sample_id}")
            texts.append(f"Claim: {claim} </s> Evidence: {_evidence_text(record.evidence)}")
        else:
            texts.append(claim)
    dataset = Dataset.from_dict(
        {"text": texts, "labels": [label_to_id[record.label] for record in records]}
    )
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
    return dataset, label_to_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/transformer.yaml"))
    parser.add_argument("--columns", type=Path, required=True, help="JSON mapping verified source fields")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Exploratory subset only")
    args = parser.parse_args()
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install optional dependencies with `uv sync --extra transformer`") from exc
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not config.get("dataset_configuration"):
        raise SystemExit("Set dataset_configuration after authorized dataset inspection")
    for key in ("train_split", "test_split"):
        if not config.get(key):
            raise SystemExit(f"Set {key} to a source-verified split name; the script will not guess")
    columns = json.loads(args.columns.read_text(encoding="utf-8"))
    if config.get("text_mode") == "claim_plus_evidence" and not columns.get("evidence"):
        raise SystemExit("Evidence mode requires a source-verified evidence field mapping")
    train_records = load_authorized_hf_split(
        repository=config["dataset_repository"],
        configuration=config["dataset_configuration"],
        split=config["train_split"],
        columns=columns,
    )
    test_records = load_authorized_hf_split(
        repository=config["dataset_repository"],
        configuration=config["dataset_configuration"],
        split=config["test_split"],
        columns=columns,
    )
    overlap = {row.group_id for row in train_records} & {row.group_id for row in test_records}
    if overlap:
        train_records = [row for row in train_records if row.group_id not in overlap]
        print(f"Removed {len(overlap)} duplicate groups from training; held-out test unchanged.")
    if args.max_train_rows:
        train_records = train_records[: args.max_train_rows]
    groups = np.asarray([record.group_id for record in train_records], dtype=object)
    fit_ids, validation_ids = grouped_train_validation_indices(
        [row.label for row in train_records],
        groups.tolist(),
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    fit_records = [train_records[index] for index in fit_ids]
    validation_records = [train_records[index] for index in validation_ids]
    all_labels = sorted({record.label for record in train_records})
    unknown_test_labels = {record.label for record in test_records}.difference(all_labels)
    if unknown_test_labels:
        raise SystemExit(f"Test set has labels absent from training: {sorted(unknown_test_labels)}")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = available_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    label_to_id = {label: index for index, label in enumerate(all_labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=len(all_labels), label2id=label_to_id, id2label=id_to_label
    )
    train_data, _ = _dataset(
        fit_records, text_mode=config["text_mode"], tokenizer=tokenizer,
        max_length=int(config["max_length"]), label_to_id=label_to_id
    )
    validation_data, _ = _dataset(
        validation_records, text_mode=config["text_mode"], tokenizer=tokenizer,
        max_length=int(config["max_length"]), label_to_id=label_to_id
    )
    test_data, _ = _dataset(
        test_records, text_mode=config["text_mode"], tokenizer=tokenizer,
        max_length=int(config["max_length"]), label_to_id=label_to_id
    )
    if device == "cuda":
        fp16 = True
    else:
        fp16 = False

    def compute_metrics(prediction):
        logits, targets = prediction
        predicted = np.argmax(logits, axis=-1)
        report = classification_metrics(
            [id_to_label[int(value)] for value in targets],
            [id_to_label[int(value)] for value in predicted],
            all_labels,
        )
        return {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"], "weighted_f1": report["weighted_f1"]}

    output_dir = args.output_dir or Path(config["output_dir"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_train_epochs=float(config["epochs"]),
        weight_decay=float(config["weight_decay"]),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=int(config["eval_steps"]),
        save_steps=int(config["save_steps"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=max(1, int(config["eval_steps"]) // 5),
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=fp16,
        use_cpu=device == "cpu",
        label_names=["labels"],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"]))],
    )
    started = time.perf_counter()
    training_result = trainer.train()
    training_seconds = time.perf_counter() - started
    test_prediction = trainer.predict(test_data, metric_key_prefix="test")
    test_ids = np.argmax(test_prediction.predictions, axis=-1)
    test_labels = [record.label for record in test_records]
    test_metrics = classification_metrics(
        test_labels, [id_to_label[int(value)] for value in test_ids], all_labels
    )
    test_probabilities = torch.softmax(torch.tensor(test_prediction.predictions), dim=-1).numpy()
    test_metrics["probability_diagnostics"] = probability_metrics(
        test_labels, test_probabilities, all_labels
    )
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(output_dir / "final")
    result_dir = Path("results") / config["dataset_repository"].split("/")[-1].lower() / "xlmr"
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": config["dataset_repository"],
        "task": config["text_mode"],
        "model_name": config["model_name"],
        "labels": all_labels,
        "train_rows": len(fit_records),
        "validation_rows": len(validation_records),
        "test_rows": len(test_records),
        "overlap_groups_removed_from_training": len(overlap),
        "validation": trainer.evaluate(validation_data),
        "heldout": test_metrics,
        "training_seconds": training_seconds,
        "device": device,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "exploratory_subset": args.max_train_rows,
        "status": "measured",
    }
    (result_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (result_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record, pred_id, logits in zip(test_records, test_ids, test_prediction.predictions):
            probs = torch.softmax(torch.tensor(logits), dim=-1).tolist()
            stream.write(json.dumps({
                "sample_id": record.sample_id,
                "language": record.language,
                "true_label": record.label,
                "predicted_label": id_to_label[int(pred_id)],
                "probabilities": {label: float(probs[index]) for label, index in label_to_id.items()},
            }, ensure_ascii=False) + "\n")
    (result_dir / "metadata.json").write_text(json.dumps({
        "config": config,
        "columns": columns,
        "label_to_id": label_to_id,
        "train_result": training_result.metrics,
        "status": "fine_tuned",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
