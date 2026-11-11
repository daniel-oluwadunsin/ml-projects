from __future__ import annotations

import pytest

from scripts.run_transformer import _evidence_text


def test_evidence_input_accepts_nonempty_passages():
    assert _evidence_text(["First passage.", "Second passage."]) == "First passage. Second passage."


@pytest.mark.parametrize("value", [None, "", [], ["", "  "], {"text": "passage"}])
def test_evidence_input_rejects_missing_or_unverified_structures(value):
    with pytest.raises(ValueError):
        _evidence_text(value)
