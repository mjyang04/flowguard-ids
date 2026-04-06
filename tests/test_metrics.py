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
    assert 0.0 <= metrics["best_f1"] <= 1.0
    assert 0.0 <= metrics["best_f1_threshold"] <= 1.0
    assert 0.0 <= metrics["threshold_at_far_1pct"] <= 1.0
    assert 0.0 <= metrics["threshold_at_far_5pct"] <= 1.0
    assert metrics["recall_at_far_5pct"] >= metrics["recall_at_far_1pct"]
