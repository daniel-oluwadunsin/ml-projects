"""Input validation and data-leakage checks."""

from __future__ import annotations

from collections.abc import Iterable

from ml.data.schemas import ClaimRecord
from ml.data.preprocessing import normalize_text


def validate_records(records: Iterable[ClaimRecord]) -> list[ClaimRecord]:
    """Reject unusable rows while keeping all valid source records unchanged."""
    valid: list[ClaimRecord] = []
    seen: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        if not record.sample_id:
            raise ValueError(f"Row {row_number} has no stable sample_id")
        if not record.dataset or not record.source_split:
            raise ValueError(f"Row {row_number} is missing dataset/split provenance")
        claim = normalize_text(record.claim)
        label = normalize_text(record.label)
        if not claim:
            raise ValueError(f"Row {row_number} has an empty claim")
        if not label or not record.original_label:
            raise ValueError(f"Row {row_number} has a missing label")
        if record.sample_id in seen:
            raise ValueError(f"Duplicate sample_id {record.sample_id!r}")
        seen.add(record.sample_id)
        valid.append(record)
    if not valid:
        raise ValueError("No valid labeled claim records were loaded")
    return valid


def assert_group_disjoint(*groups: Iterable[str]) -> None:
    """Raise if any normalized claim group occurs in more than one split."""
    observed: set[str] = set()
    for index, values in enumerate(groups):
        current = set(values)
        overlap = observed.intersection(current)
        if overlap:
            preview = ", ".join(sorted(overlap)[:3])
            raise ValueError(f"Duplicate groups cross split {index}: {preview}")
        observed.update(current)

