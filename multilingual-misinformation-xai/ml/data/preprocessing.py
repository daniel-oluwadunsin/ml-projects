"""Conservative Unicode normalization and duplicate grouping."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply NFC and whitespace normalization without stripping diacritics."""
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def duplicate_group_id(text: str) -> str:
    """Return a deterministic normalized key used to keep duplicates in one split."""
    return normalize_text(text).casefold()
