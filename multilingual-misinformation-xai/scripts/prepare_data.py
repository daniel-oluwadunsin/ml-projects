"""Validate FEVER local files and save observed, license-safe aggregate statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data.loaders import load_fever_jsonl
from ml.data.statistics import summarize_records
from ml.data.validation import assert_group_disjoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["fever"], default="fever")
    parser.add_argument("--train", type=Path, default=Path("data/raw/fever_train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/raw/fever_dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/fever/data_summary.json"))
    args = parser.parse_args()
    train = load_fever_jsonl(args.train, "train")
    dev = load_fever_jsonl(args.dev, "official_labelled_dev")
    overlap = {row.group_id for row in train} & {row.group_id for row in dev}
    if overlap:
        train = [row for row in train if row.group_id not in overlap]
    summary = {
        "source": "FEVER official JSONL files",
        "train": summarize_records(train),
        "official_labelled_dev": summarize_records(dev),
        "partition_group_overlap_removed_from_training": len(overlap),
        "note": "Statistics computed from local files by this run; raw records are not copied here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
