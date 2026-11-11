"""Dataset summaries based only on records actually loaded by the user."""

from __future__ import annotations

from collections import Counter
from statistics import median

from ml.data.schemas import ClaimRecord


def summarize_records(records: list[ClaimRecord]) -> dict[str, object]:
    lengths = [len(record.claim) for record in records]
    group_counts = Counter(row.group_id for row in records)

    def has_textual_evidence(record: ClaimRecord) -> bool:
        if isinstance(record.evidence, str):
            return bool(record.evidence.strip())
        return isinstance(record.evidence, (list, tuple)) and any(
            isinstance(item, str) and item.strip() for item in record.evidence
        )

    return {
        "dataset": records[0].dataset if records else None,
        "n_examples": len(records),
        "label_counts": dict(sorted(Counter(row.label for row in records).items())),
        "language_counts": dict(
            sorted(Counter(row.language or "unknown" for row in records).items())
        ),
        "source_split_counts": dict(sorted(Counter(row.source_split for row in records).items())),
        "dataset_fingerprints": sorted(
            {row.dataset_fingerprint for row in records if row.dataset_fingerprint}
        ),
        "evidence_annotation_present": sum(row.evidence is not None for row in records),
        "evidence_text_available": sum(has_textual_evidence(row) for row in records),
        "duplicate_claim_groups": sum(count > 1 for count in group_counts.values()),
        "duplicate_claim_rows_beyond_first": sum(count - 1 for count in group_counts.values()),
        "claim_character_length": {
            "min": min(lengths) if lengths else None,
            "median": median(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
    }
