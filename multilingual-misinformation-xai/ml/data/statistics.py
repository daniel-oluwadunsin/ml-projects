"""Dataset summaries based only on records actually loaded by the user."""

from __future__ import annotations

from collections import Counter

from ml.data.schemas import ClaimRecord


def summarize_records(records: list[ClaimRecord]) -> dict[str, object]:
    lengths = [len(record.claim) for record in records]
    return {
        "dataset": records[0].dataset if records else None,
        "n_examples": len(records),
        "label_counts": dict(sorted(Counter(row.label for row in records).items())),
        "language_counts": dict(
            sorted(Counter(row.language or "unknown" for row in records).items())
        ),
        "source_split_counts": dict(sorted(Counter(row.source_split for row in records).items())),
        "missing_evidence": sum(row.evidence is None for row in records),
        "duplicate_claim_groups": len(records) - len({row.group_id for row in records}),
        "claim_character_length": {
            "min": min(lengths) if lengths else None,
            "median": sorted(lengths)[len(lengths) // 2] if lengths else None,
            "max": max(lengths) if lengths else None,
        },
    }
