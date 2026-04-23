"""Unit tests for nids.evaluation.two_stage (cascade inference).

The cascade module is intentionally pure-numpy so these tests do not need
torch, sklearn, or trained artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from nids.evaluation.two_stage import (
    TwoStageResult,
    binarize_scores,
    combine_stages,
    gating_stats,
    run_two_stage,
)


# ---------------------------------------------------------------------------
# binarize_scores
# ---------------------------------------------------------------------------

def test_binarize_scores_default_threshold():
    scores = np.array([0.1, 0.4, 0.5, 0.6, 0.9])
    preds = binarize_scores(scores)
    # default threshold 0.5, and the boundary is inclusive
    np.testing.assert_array_equal(preds, np.array([0, 0, 1, 1, 1], dtype=np.int64))


def test_binarize_scores_custom_threshold():
    scores = np.array([0.1, 0.4, 0.5, 0.6, 0.9])
    preds = binarize_scores(scores, threshold=0.7)
    np.testing.assert_array_equal(preds, np.array([0, 0, 0, 0, 1], dtype=np.int64))


def test_binarize_scores_returns_int64():
    preds = binarize_scores(np.array([0.2, 0.8]))
    assert preds.dtype == np.int64


# ---------------------------------------------------------------------------
# combine_stages
# ---------------------------------------------------------------------------

def test_combine_stages_gates_to_benign_when_stage1_negative():
    stage1 = np.array([0, 0, 0, 0])
    stage2 = np.array([3, 5, 7, 9])
    merged = combine_stages(stage1, stage2, benign_class=0)
    np.testing.assert_array_equal(merged, np.zeros(4, dtype=np.int64))


def test_combine_stages_passes_stage2_when_stage1_positive():
    stage1 = np.array([1, 1, 1])
    stage2 = np.array([2, 3, 4])
    merged = combine_stages(stage1, stage2, benign_class=0)
    np.testing.assert_array_equal(merged, np.array([2, 3, 4], dtype=np.int64))


def test_combine_stages_mixed_gating():
    stage1 = np.array([1, 0, 1, 0, 1])
    stage2 = np.array([3, 5, 7, 9, 2])
    merged = combine_stages(stage1, stage2, benign_class=0)
    # stage1=0 positions are gated to benign(0); stage1=1 positions keep stage2
    np.testing.assert_array_equal(merged, np.array([3, 0, 7, 0, 2], dtype=np.int64))


def test_combine_stages_respects_non_zero_benign():
    stage1 = np.array([0, 1, 0])
    stage2 = np.array([4, 6, 8])
    merged = combine_stages(stage1, stage2, benign_class=99)
    np.testing.assert_array_equal(merged, np.array([99, 6, 99], dtype=np.int64))


def test_combine_stages_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="align"):
        combine_stages(np.array([0, 1]), np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# gating_stats
# ---------------------------------------------------------------------------

def test_gating_stats_perfect_classifier():
    y_true = np.array([0, 0, 1, 1, 2, 3])  # multiclass: 0=benign
    stage1 = np.array([0, 0, 1, 1, 1, 1])
    stats = gating_stats(y_true, stage1, benign_class=0)
    assert stats["stage1_tp"] == 4.0
    assert stats["stage1_fp"] == 0.0
    assert stats["stage1_tn"] == 2.0
    assert stats["stage1_fn"] == 0.0
    assert stats["stage1_recall"] == pytest.approx(1.0)
    assert stats["stage1_precision"] == pytest.approx(1.0)
    assert stats["stage1_far"] == pytest.approx(0.0)


def test_gating_stats_counts_false_alarms():
    # 3 benign, 3 attacks; stage-1 wrongly fires on 1 benign, misses 1 attack
    y_true = np.array([0, 0, 0, 1, 2, 3])
    stage1 = np.array([0, 0, 1, 1, 1, 0])
    stats = gating_stats(y_true, stage1, benign_class=0)
    assert stats["stage1_tp"] == 2.0
    assert stats["stage1_fp"] == 1.0
    assert stats["stage1_tn"] == 2.0
    assert stats["stage1_fn"] == 1.0
    assert stats["stage1_recall"] == pytest.approx(2 / 3)
    assert stats["stage1_precision"] == pytest.approx(2 / 3)
    assert stats["stage1_far"] == pytest.approx(1 / 3)


def test_gating_stats_precision_zero_when_no_positive_predictions():
    y_true = np.array([1, 1, 2])
    stage1 = np.array([0, 0, 0])
    stats = gating_stats(y_true, stage1, benign_class=0)
    assert stats["stage1_precision"] == 0.0
    assert stats["stage1_recall"] == 0.0
    assert stats["stage1_tp"] == 0.0


def test_gating_stats_empty_inputs():
    stats = gating_stats(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    for key in (
        "stage1_attack_rate",
        "stage1_tp",
        "stage1_fp",
        "stage1_tn",
        "stage1_fn",
        "stage1_recall",
        "stage1_precision",
        "stage1_far",
    ):
        assert stats[key] == 0.0


def test_gating_stats_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="align"):
        gating_stats(np.array([0, 1]), np.array([0]))


# ---------------------------------------------------------------------------
# run_two_stage
# ---------------------------------------------------------------------------

def test_run_two_stage_with_scores_thresholds_correctly():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    stage2 = np.array([0, 3, 0, 7])
    result = run_two_stage(
        stage1_scores=scores,
        stage1_binary=None,
        stage2_multiclass=stage2,
        threshold=0.5,
    )
    assert isinstance(result, TwoStageResult)
    np.testing.assert_array_equal(result.stage1_predictions, np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(result.final_predictions, np.array([0, 3, 0, 7]))
    np.testing.assert_allclose(result.stage1_scores, scores)
    assert result.threshold == 0.5
    # Without y_true, stats stay empty.
    assert result.stats == {}


def test_run_two_stage_with_prebinarized_stage1():
    stage1 = np.array([1, 0, 1])
    stage2 = np.array([4, 5, 6])
    result = run_two_stage(
        stage1_scores=None,
        stage1_binary=stage1,
        stage2_multiclass=stage2,
    )
    np.testing.assert_array_equal(result.final_predictions, np.array([4, 0, 6]))
    assert result.stage1_scores is None


def test_run_two_stage_records_gating_stats_when_y_true_given():
    stage1 = np.array([1, 0, 1, 0])
    stage2 = np.array([2, 3, 4, 5])
    y_true = np.array([2, 0, 0, 5])  # stage1 is right twice, wrong twice
    result = run_two_stage(
        stage1_scores=None,
        stage1_binary=stage1,
        stage2_multiclass=stage2,
        y_true=y_true,
    )
    assert result.stats["stage1_tp"] == 1.0  # (stage1=1, y=2)
    assert result.stats["stage1_fp"] == 1.0  # (stage1=1, y=0)
    assert result.stats["stage1_tn"] == 1.0  # (stage1=0, y=0)
    assert result.stats["stage1_fn"] == 1.0  # (stage1=0, y=5)


def test_run_two_stage_raises_when_no_stage1_input():
    with pytest.raises(ValueError, match="stage1"):
        run_two_stage(
            stage1_scores=None,
            stage1_binary=None,
            stage2_multiclass=np.array([0, 1]),
        )


def test_run_two_stage_threshold_affects_gating():
    # Score 0.6 is "attack" at threshold 0.5 but "benign" at threshold 0.7
    scores = np.array([0.6, 0.6])
    stage2 = np.array([3, 4])
    low = run_two_stage(
        stage1_scores=scores, stage1_binary=None, stage2_multiclass=stage2, threshold=0.5
    )
    high = run_two_stage(
        stage1_scores=scores, stage1_binary=None, stage2_multiclass=stage2, threshold=0.7
    )
    np.testing.assert_array_equal(low.final_predictions, np.array([3, 4]))
    np.testing.assert_array_equal(high.final_predictions, np.array([0, 0]))
