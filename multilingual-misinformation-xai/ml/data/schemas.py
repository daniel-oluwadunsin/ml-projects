"""Canonical data records that retain source provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """One labeled claim, with source labels and optional evidence kept intact."""

    sample_id: str
    dataset: str
    claim: str
    original_label: str
    label: str
    source_split: str
    group_id: str
    language: str | None = None
    evidence: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

