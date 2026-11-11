"""Deterministic duplicate-group-aware train/validation splitting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ml.data.validation import assert_group_disjoint


def grouped_train_validation_indices(
    labels: Sequence[str], groups: Sequence[str], *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create an approximately stratified holdout with no duplicate group leakage."""
    if len(labels) != len(groups) or len(labels) < 4:
        raise ValueError("labels and groups must have equal length with at least four records")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    n_splits = max(2, round(1 / validation_fraction))
    if len(set(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique groups for grouped validation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fit_indices, validation_indices = next(
        splitter.split(np.zeros(len(labels)), labels, groups)
    )
    assert_group_disjoint(
        [groups[index] for index in fit_indices],
        [groups[index] for index in validation_indices],
    )
    if set(labels).difference(labels[index] for index in fit_indices):
        raise ValueError("Grouped training partition is missing one or more classes")
    if not len(validation_indices):
        raise ValueError("Grouped validation partition is empty")
    return fit_indices, validation_indices
