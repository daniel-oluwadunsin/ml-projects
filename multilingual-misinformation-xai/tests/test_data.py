from __future__ import annotations

import json

import pytest

from ml.data.loaders import load_fever_jsonl
from ml.data.preprocessing import duplicate_group_id, normalize_text
from ml.data.splits import grouped_train_validation_indices
from ml.data.validation import assert_group_disjoint


def test_unicode_normalization_preserves_yoruba_tone_marks():
    decomposed = "O\u0323\u0300r\u00f2"
    normalized = normalize_text(decomposed)
    assert normalized == "Ọ̀rò"
    assert duplicate_group_id("  Ọ̀rò   test ") == duplicate_group_id("Ọ̀rò test")


def test_fever_loader_preserves_source_label_and_maps_task_label(tmp_path):
    path = tmp_path / "fever.jsonl"
    path.write_text(
        json.dumps({"id": 17, "label": "NOT ENOUGH INFO", "claim": "A claim", "evidence": []}) + "\n",
        encoding="utf-8",
    )
    [record] = load_fever_jsonl(path, "dev")
    assert record.sample_id == "17"
    assert record.original_label == "NOT ENOUGH INFO"
    assert record.label == "NOT_ENOUGH_INFORMATION"
    assert record.language == "en"
    assert record.source_split == "dev"


def test_loader_rejects_unknown_source_label(tmp_path):
    path = tmp_path / "fever.jsonl"
    path.write_text(json.dumps({"id": 1, "label": "TRUE", "claim": "A claim"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized FEVER label"):
        load_fever_jsonl(path, "train")


def test_split_guard_rejects_cross_partition_duplicate():
    with pytest.raises(ValueError, match="cross split"):
        assert_group_disjoint(["a", "b"], ["b", "c"])


def test_grouped_holdout_is_reproducible_and_keeps_groups_together():
    labels = ["yes", "no"] * 35
    groups = [f"group-{index // 2}" for index in range(70)]
    first_train, first_validation = grouped_train_validation_indices(
        labels, groups, validation_fraction=0.2, seed=19
    )
    second_train, second_validation = grouped_train_validation_indices(
        labels, groups, validation_fraction=0.2, seed=19
    )
    assert first_train.tolist() == second_train.tolist()
    assert first_validation.tolist() == second_validation.tolist()
    assert_group_disjoint(
        [groups[index] for index in first_train],
        [groups[index] for index in first_validation],
    )


def test_fever_loader_rejects_duplicate_sample_ids(tmp_path):
    path = tmp_path / "fever.jsonl"
    rows = [
        {"id": 1, "label": "SUPPORTS", "claim": "first"},
        {"id": 1, "label": "REFUTES", "claim": "second"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_fever_jsonl(path, "train")
