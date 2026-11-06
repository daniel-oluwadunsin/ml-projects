"""Download and unpack the UCI Student Performance dataset."""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "student-performance.zip"
DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def download_data(force: bool = False) -> Path:
    """Download the UCI archive if needed and return the math CSV path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / "student-mat.csv"
    if csv_path.exists() and not force:
        return csv_path

    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "student-performance-ml-project/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with ARCHIVE_PATH.open("wb") as output:
            shutil.copyfileobj(response, output)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        member = next((name for name in archive.namelist() if name.endswith("student-mat.csv")), None)
        if member is not None:
            csv_path.write_bytes(archive.read(member))
        else:
            nested_name = next((name for name in archive.namelist() if name.endswith("student.zip")), None)
            if nested_name is None:
                raise RuntimeError("The UCI archive did not contain student.zip or student-mat.csv")
            with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested_archive:
                nested_member = next(
                    (name for name in nested_archive.namelist() if name.endswith("student-mat.csv")),
                    None,
                )
                if nested_member is None:
                    raise RuntimeError("The nested UCI archive did not contain student-mat.csv")
                csv_path.write_bytes(nested_archive.read(nested_member))
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download again even if data exists")
    args = parser.parse_args()
    path = download_data(force=args.force)
    print(f"Dataset ready: {path}")


if __name__ == "__main__":
    main()
