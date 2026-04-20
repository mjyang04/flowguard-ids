"""Torch-independent tests for the selection-metric dispatcher."""
from __future__ import annotations

import numpy as np
import pytest

from nids.training.selection import compute_selection_metric


def test_selection_metric_macro_f1_for_multiclass():
    y_true = np.array([0, 1, 2, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1, 2, 1])
    value = compute_selection_metric(
        "macro_f1",
        y_true=y_true,
        y_pred=y_pred,
        y_score=None,
        num_classes=3,
    )
    assert 0.0 < value <= 1.0


def test_selection_metric_weighted_f1_for_multiclass():
    y_true = np.array([0, 0, 1, 2, 2, 3, 3, 3])
    y_pred = np.array([0, 1, 1, 2, 2, 3, 0, 3])
    value = compute_selection_metric(
        "weighted_f1",
        y_true=y_true,
        y_pred=y_pred,
        y_score=None,
        num_classes=4,
    )
    assert 0.0 < value <= 1.0


def test_selection_metric_binary_recall_at_far_still_works():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    value = compute_selection_metric(
        "recall_at_far_1pct",
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        num_classes=2,
    )
    assert value == pytest.approx(1.0)


def test_selection_metric_rejects_binary_metric_in_multiclass():
    with pytest.raises(ValueError, match="binary-only"):
        compute_selection_metric(
            "recall_at_far_1pct",
            y_true=np.array([0, 1, 2]),
            y_pred=np.array([0, 1, 2]),
            y_score=None,
            num_classes=3,
        )


def test_selection_metric_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown selection_metric"):
        compute_selection_metric(
            "mystery_metric",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_score=None,
            num_classes=2,
        )
