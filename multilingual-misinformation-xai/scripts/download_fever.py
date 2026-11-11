"""Download official FEVER train and labelled development JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import certifi

URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    request = Request(url, headers={"User-Agent": "ml-projects-research/0.1"})
    with urlopen(request, context=context, timeout=60) as response:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            temporary_path = Path(temp.name)
            while chunk := response.read(1024 * 1024):
                temp.write(chunk)
    try:
        with temporary_path.open(encoding="utf-8") as stream:
            first = json.loads(stream.readline())
        if not {"id", "label", "claim"}.issubset(first):
            raise ValueError(f"Unexpected FEVER schema in download from {url}")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Downloaded {destination} ({destination.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=["all", "train", "dev"], default="all")
    args = parser.parse_args()
    selected = URLS if args.split == "all" else {args.split: URLS[args.split]}
    for split, url in selected.items():
        destination = args.output_dir / f"fever_{split}.jsonl"
        if destination.exists():
            print(f"Already present; leaving unchanged: {destination}")
        else:
            download(url, destination)


if __name__ == "__main__":
    main()
