from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)


def _coerce_binary_scores(y_score: np.ndarray | None) -> np.ndarray | None:
    if y_score is None:
        return None

    scores = np.asarray(y_score, dtype=np.float64)
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            scores = scores[:, 0]
        elif scores.shape[1] == 2:
            scores = scores[:, 1]
        else:
            return None
    return scores.reshape(-1)


def _compute_binary_score_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray | None,
    benign_class: int,
) -> dict:
    scores = _coerce_binary_scores(y_score)
    if scores is None or len(scores) != len(y_true):
        return {}

    y_true_bin = (np.asarray(y_true) != benign_class).astype(np.int64)
    if len(np.unique(y_true_bin)) < 2:
        return {}

    nan_mask = np.isnan(scores)
    if nan_mask.any():
        scores = np.where(nan_mask, 0.0, scores)
    scores = np.clip(scores, 0.0, 1.0)

    # Vectorized threshold sweep via sorted cumulative sums.
    # Descending sort lets us compute TP/FP at every unique threshold
    # in O(n log n) instead of O(n * #thresholds).
    total_pos = int(y_true_bin.sum())
    total_neg = len(y_true_bin) - total_pos

    desc_idx = np.argsort(-scores, kind="stable")
    sorted_labels = y_true_bin[desc_idx]
    sorted_scores = scores[desc_idx]

    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(1 - sorted_labels)

    # For threshold t, we predict positive for all samples with score >= t.
    # When multiple samples share the same score they are all predicted
    # positive together, so we must take the *last* position in each group
    # of identical scores (i.e. where the next score differs or end-of-array).
    unique_mask = np.concatenate(
        [sorted_scores[:-1] != sorted_scores[1:], np.array([True])]
    )
    tp_at = tp_cum[unique_mask]
    fp_at = fp_cum[unique_mask]
    thresholds = sorted_scores[unique_mask]

    recall = tp_at / max(1, total_pos)
    # Compute precision and F1 safely to avoid RuntimeWarning from division
    denom = tp_at + fp_at
    precision = np.divide(tp_at, denom, out=np.zeros_like(tp_at, dtype=np.float64), where=denom > 0)
    far = fp_at / max(1, total_neg)
    pr_sum = precision + recall
    f1 = np.divide(
        2.0 * precision * recall, pr_sum,
        out=np.zeros_like(pr_sum, dtype=np.float64), where=pr_sum > 0,
    )

    # Best F1 — tie-break: max recall, then min FAR, then max threshold
    best_f1_order = np.lexsort((thresholds, -far, recall, f1))
    best_f1_idx = int(best_f1_order[-1])
    best_f1_val = float(f1[best_f1_idx])
    best_f1_thr = float(thresholds[best_f1_idx])

    # Best recall under FAR constraint
    def _best_under_far(max_far: float) -> tuple[float, float]:
        eligible = far <= max_far + 1e-12
        if not eligible.any():
            return 0.0, 1.0
        idx = np.where(eligible)[0]
        # Among eligible, pick highest recall; break ties by lowest FAR
        best_i = idx[int(np.argmax(recall[idx]))]
        return float(recall[best_i]), float(thresholds[best_i])

    recall_1, thr_1 = _best_under_far(0.01)
    recall_5, thr_5 = _best_under_far(0.05)

    return {
        "pr_auc": float(average_precision_score(y_true_bin, scores)),
        "roc_auc": float(roc_auc_score(y_true_bin, scores)),
        "ece": _expected_calibration_error(y_true_bin, scores),
        "best_f1": best_f1_val,
        "best_f1_threshold": best_f1_thr,
        "recall_at_far_1pct": recall_1,
        "threshold_at_far_1pct": thr_1,
        "recall_at_far_5pct": recall_5,
        "threshold_at_far_5pct": thr_5,
    }


def compute_nids_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benign_class: int = 0,
    y_score: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    unique_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    attack_classes = [c for c in unique_classes if c != benign_class]

    attack_recalls: list[float] = []
    attack_precisions: list[float] = []
    for cls in attack_classes:
        key = str(cls)
        if key in report:
            attack_recalls.append(report[key]["recall"])
            attack_precisions.append(report[key]["precision"])

    benign_total = max(1, int((y_true == benign_class).sum()))
    benign_false_alarms = int(((y_true == benign_class) & (y_pred != benign_class)).sum())

    avg_attack_recall = float(np.mean(attack_recalls)) if attack_recalls else 0.0
    attack_macro_precision = float(np.mean(attack_precisions)) if attack_precisions else 0.0

    metrics = {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "avg_attack_recall": avg_attack_recall,
        "attack_macro_precision": attack_macro_precision,
        "benign_false_alarm_rate": float(benign_false_alarms / benign_total),
        "attack_miss_rate": float(1.0 - avg_attack_recall),
        "confusion_matrix": cm.tolist(),
    }
    metrics.update(_compute_binary_score_metrics(y_true, y_score, benign_class))
    return metrics
