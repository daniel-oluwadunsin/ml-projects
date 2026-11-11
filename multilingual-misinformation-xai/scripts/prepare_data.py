"""Validate FEVER local files and save observed, license-safe aggregate statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data.loaders import load_fever_jsonl
from ml.data.statistics import summarize_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["fever"], default="fever")
    parser.add_argument("--train", type=Path, default=Path("data/raw/fever_train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/raw/fever_dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/fever/data_summary.json"))
    args = parser.parse_args()
    train = load_fever_jsonl(args.train, "train")
    dev = load_fever_jsonl(args.dev, "official_labelled_dev")
    source_train_rows = len(train)
    overlap = {row.group_id for row in train} & {row.group_id for row in dev}
    overlap_rows = sum(row.group_id in overlap for row in train)
    if overlap:
        train = [row for row in train if row.group_id not in overlap]
    summary = {
        "source": "FEVER official JSONL files",
        "train": summarize_records(train),
        "official_labelled_dev": summarize_records(dev),
        "partition_group_overlap_removed_from_training": len(overlap),
        "partition_rows_removed_from_training": overlap_rows,
        "source_train_rows_before_overlap_filter": source_train_rows,
        "source_development_rows": len(dev),
        "protocol": (
            "Source development split reserved as held-out evaluation; never used for selection."
        ),
        "note": (
            "Statistics computed from local files by this run; raw records are not copied here."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
