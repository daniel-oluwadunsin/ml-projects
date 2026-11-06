"""Download and verify the standard AG News CSV train/test files."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
FILES = {
    "train.csv": {
        "url": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "md5": "b1a00f826fdfbd249f79597b59e1dc12",
    },
    "test.csv": {
        "url": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
        "md5": "d52ea96a97a2d943681189a97654912d",
    },
}


def download_data(force: bool = False) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: RAW_DIR / name for name in FILES}
    for name, details in FILES.items():
        path = paths[name]
        valid_existing = path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == details["md5"]
        if valid_existing and not force:
            continue
        request = Request(details["url"], headers={"User-Agent": "news-classification-ml-project/1.0"})
        with urlopen(request, timeout=120) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != details["md5"]:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {name}: got {digest}, expected {details['md5']}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download both files again")
    args = parser.parse_args()
    for name, path in download_data(force=args.force).items():
        print(f"{name}: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
