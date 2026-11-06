"""Download and prepare the WiLI-2018 language identification data."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "wili-2018.zip"
DATA_URL = "https://zenodo.org/records/841984/files/wili-2018.zip?download=1"
EXPECTED_MD5 = "3dc5bd41587811ad6b0d04ae2f235f84"
EXPECTED_FILES = ("x_train.txt", "y_train.txt", "x_test.txt", "y_test.txt")


def download_data(force: bool = False) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: RAW_DIR / name for name in EXPECTED_FILES}
    if all(path.exists() for path in paths.values()) and not force:
        return paths

    if force or not ARCHIVE_PATH.exists():
        request = Request(
            DATA_URL, headers={"User-Agent": "language-identification-ml-project/1.0"}
        )
        with (
            urlopen(request, timeout=180) as response,
            ARCHIVE_PATH.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)

    digest = hashlib.md5(ARCHIVE_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise RuntimeError(
            f"WiLI archive checksum mismatch: got {digest}, expected {EXPECTED_MD5}"
        )

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        missing = set(EXPECTED_FILES) - set(members)
        if missing:
            raise RuntimeError(
                f"WiLI archive is missing expected files: {sorted(missing)}"
            )
        for name, destination in paths.items():
            destination.write_bytes(archive.read(members[name]))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Download the archive again"
    )
    args = parser.parse_args()
    paths = download_data(force=args.force)
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
