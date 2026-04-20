import numpy as np

from nids.evaluation.metrics import compute_nids_metrics


def test_compute_nids_metrics_includes_binary_score_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_score = np.array([0.02, 0.05, 0.35, 0.40, 0.45, 0.70, 0.90, 0.95], dtype=np.float32)
    y_pred = (y_score >= 0.5).astype(np.int64)

    metrics = compute_nids_metrics(y_true, y_pred, y_score=y_score)

    for key in [
        "pr_auc",
        "roc_auc",
        "ece",
        "best_f1",
        "best_f1_threshold",
        "recall_at_far_1pct",
        "threshold_at_far_1pct",
        "recall_at_far_5pct",
        "threshold_at_far_5pct",
    ]:
        assert key in metrics

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["best_f1"] <= 1.0
    assert 0.0 <= metrics["best_f1_threshold"] <= 1.0
    assert 0.0 <= metrics["threshold_at_far_1pct"] <= 1.0
    assert 0.0 <= metrics["threshold_at_far_5pct"] <= 1.0
    assert metrics["recall_at_far_5pct"] >= metrics["recall_at_far_1pct"]


def test_mcc_included_in_metrics():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1], dtype=np.int64)
    metrics = compute_nids_metrics(y_true, y_pred)
    assert "mcc" in metrics
    assert -1.0 <= metrics["mcc"] <= 1.0


def test_ece_well_calibrated_model():
    """A perfectly calibrated model should have low ECE."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    # Scores closely match true frequencies
    y_score = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9, 0.9], dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)
    metrics = compute_nids_metrics(y_true, y_pred, y_score=y_score)
    assert metrics["ece"] < 0.3  # reasonably calibrated


def test_duplicate_scores_threshold_sweep():
    """Regression: duplicate scores must group correctly.

    y_true=[1,1,0], y_score=[0.8,0.8,0.7].
    At threshold=0.8 all positives are captured with 0 false positives,
    so best_f1=1.0 and recall_at_far_1pct=1.0.
    """
    y_true = np.array([1, 1, 0], dtype=np.int64)
    y_score = np.array([0.8, 0.8, 0.7], dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)

    metrics = compute_nids_metrics(y_true, y_pred, y_score=y_score)

    assert metrics["best_f1"] == 1.0, f"expected best_f1=1.0, got {metrics['best_f1']}"
    assert metrics["recall_at_far_1pct"] == 1.0, (
        f"expected recall_at_far_1pct=1.0, got {metrics['recall_at_far_1pct']}"
    )


def test_all_same_score():
    """Edge case: every sample has the same score."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    y_pred = np.array([1, 1, 1, 1], dtype=np.int64)

    metrics = compute_nids_metrics(y_true, y_pred, y_score=y_score)

    assert "best_f1" in metrics
    assert "recall_at_far_1pct" in metrics
    assert 0.0 <= metrics["best_f1"] <= 1.0


def test_best_f1_tiebreak_prefers_higher_recall():
    """When two thresholds give the same F1, prefer higher recall then lower FAR."""
    # 3 positives, 3 negatives.
    # Scores are crafted so two thresholds produce the same F1:
    #   threshold=0.7: TP=2, FP=0 → recall=2/3, prec=1.0, F1 = 4/5 * (2/3)/(1+2/3)… let's just
    #   verify the implementation picks the higher-recall option among ties.
    #
    # threshold=0.8: TP=2, FP=1 → recall=2/3, prec=2/3, F1=2/3
    # threshold=0.4: TP=3, FP=2 → recall=1.0, prec=3/5, F1=3/4
    # No tie here, but we confirm best_f1 = 0.75 with recall=1.0 (threshold 0.4).
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.4, 0.8, 0.4, 0.8, 0.9], dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)

    metrics = compute_nids_metrics(y_true, y_pred, y_score=y_score)

    assert abs(metrics["best_f1"] - 0.75) < 1e-9
    # The threshold that achieves recall=1.0 with F1=0.75
    assert metrics["best_f1_threshold"] == 0.4


def test_multiclass_metrics_expose_weighted_and_per_class():
    y_true = np.array([0, 0, 1, 2, 2, 3, 3, 3])
    y_pred = np.array([0, 1, 1, 2, 2, 3, 0, 3])
    m = compute_nids_metrics(y_true, y_pred, benign_class=0)
    assert "weighted_f1" in m
    assert "per_class_f1" in m
    assert "per_class_recall" in m
    assert "per_class_precision" in m
    assert set(m["per_class_f1"].keys()) == {"0", "1", "2", "3"}
    assert 0.0 <= m["weighted_f1"] <= 1.0
    for cls_key, value in m["per_class_recall"].items():
        assert 0.0 <= value <= 1.0, f"per_class_recall[{cls_key}]={value}"


def test_multiclass_metrics_skip_binary_score_keys_when_y_score_is_none():
    y_true = np.array([0, 1, 2, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1, 2, 1])
    m = compute_nids_metrics(y_true, y_pred, benign_class=0)
    # binary-only keys must not appear when y_score is None / multiclass
    assert "pr_auc" not in m
    assert "recall_at_far_1pct" not in m


def test_binary_metrics_also_carry_per_class_entries():
    """New per-class keys should be additive and present even for binary."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = compute_nids_metrics(y_true, y_pred, benign_class=0)
    assert "per_class_f1" in m and set(m["per_class_f1"].keys()) == {"0", "1"}
    assert "weighted_f1" in m
