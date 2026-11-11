"""Load FEVER JSONL and explicitly mapped authorized Hugging Face datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ml.data.preprocessing import duplicate_group_id, normalize_text
from ml.data.schemas import ClaimRecord
from ml.data.validation import validate_records


FEVER_TO_CANONICAL = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFORMATION",
}
AFRIFACT_TO_CANONICAL = {
    "SUPPORT": "SUPPORTED",
    "SUPPORTS": "SUPPORTED",
    "SUPPORTED": "SUPPORTED",
    "REFUTE": "REFUTED",
    "REFUTES": "REFUTED",
    "REFUTED": "REFUTED",
    "NEI": "NOT_ENOUGH_INFORMATION",
    "NOT ENOUGH INFORMATION": "NOT_ENOUGH_INFORMATION",
    "NOT_ENOUGH_INFORMATION": "NOT_ENOUGH_INFORMATION",
}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")
            yield record


def _canonical_label(value: Any, mapping: Mapping[str, str], dataset: str) -> tuple[str, str]:
    if value is None:
        raise ValueError(f"Missing label in {dataset} record")
    original = normalize_text(str(value))
    canonical = mapping.get(original.upper())
    if canonical is None:
        raise ValueError(f"Unrecognized {dataset} label {original!r}; no implicit mapping applied")
    return original, canonical


def _required_string(value: Any, *, dataset: str, field: str) -> str:
    if value is None:
        raise ValueError(f"Missing required {dataset} field {field!r}")
    normalized = normalize_text(str(value))
    if not normalized:
        raise ValueError(f"Empty required {dataset} field {field!r}")
    return normalized


def load_fever_jsonl(
    path: str | Path, split: str, *, include_evidence: bool = True
) -> list[ClaimRecord]:
    """Load FEVER while preserving the official label and evidence payload."""
    records: list[ClaimRecord] = []
    for source in _iter_jsonl(Path(path)):
        for field in ("id", "claim", "label"):
            if field not in source:
                raise ValueError(f"FEVER record is missing required field {field!r}")
        sample_id = _required_string(source["id"], dataset="FEVER", field="id")
        claim = _required_string(source["claim"], dataset="FEVER", field="claim")
        original, canonical = _canonical_label(source["label"], FEVER_TO_CANONICAL, "FEVER")
        records.append(
            ClaimRecord(
                sample_id=sample_id,
                dataset="fever",
                claim=claim,
                original_label=original,
                label=canonical,
                source_split=split,
                group_id=duplicate_group_id(claim),
                language="en",
                evidence=source.get("evidence") if include_evidence else None,
            )
        )
    return validate_records(records)


def load_authorized_hf_split(
    *,
    repository: str,
    configuration: str,
    split: str,
    columns: Mapping[str, str],
    token: str | None = None,
) -> list[ClaimRecord]:
    """Load a Hugging Face split using a user-verified source field mapping.

    `columns` requires `id`, `claim`, `label`, and `language`; `evidence` is
    optional. This deliberately does not guess a repository config or schema.
    """
    required = {"id", "claim", "label", "language"}
    missing = required.difference(columns)
    invalid = {key for key in required if not isinstance(columns.get(key), str) or not columns[key]}
    if missing or invalid:
        details = sorted(missing | invalid)
        raise ValueError(f"Explicit column mapping is missing/invalid: {', '.join(details)}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies with `uv sync` first") from exc
    try:
        dataset = load_dataset(repository, configuration, split=split, token=token)
    except Exception as exc:  # library errors can vary by Hub policy/config/version
        raise RuntimeError(
            f"Could not load {repository!r} configuration {configuration!r}, split {split!r}. "
            "Confirm you accepted its access conditions, authenticated with `hf auth login`, "
            "and inspected the available configuration/features. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    available = set(dataset.column_names)
    absent = {field for field in columns.values() if field}.difference(available)
    if absent:
        raise ValueError(
            f"Mapped source fields are absent: {sorted(absent)}. Actual fields: {sorted(available)}"
        )
    records: list[ClaimRecord] = []
    for row in dataset:
        sample_id = _required_string(row[columns["id"]], dataset="AfrIFact", field=columns["id"])
        claim = _required_string(row[columns["claim"]], dataset="AfrIFact", field=columns["claim"])
        original, canonical = _canonical_label(
            row[columns["label"]], AFRIFACT_TO_CANONICAL, "AfrIFact"
        )
        evidence_column = columns.get("evidence")
        group_column = columns.get("group_id")
        source_group = row.get(group_column) if group_column else None
        language_value = row.get(columns["language"])
        records.append(
            ClaimRecord(
                sample_id=sample_id,
                dataset=f"{repository}/{configuration}",
                claim=claim,
                original_label=original,
                label=canonical,
                source_split=split,
                group_id=(
                    f"{repository}/{configuration}:{source_group}"
                    if source_group is not None
                    else duplicate_group_id(claim)
                ),
                language=(normalize_text(str(language_value)).lower() or None)
                if language_value is not None
                else None,
                evidence=row.get(evidence_column) if evidence_column else None,
            )
        )
    return validate_records(records)
