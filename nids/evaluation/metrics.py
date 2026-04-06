from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


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

    scores = np.clip(scores, 0.0, 1.0)
    thresholds = np.unique(np.concatenate(([1.0], scores)))
    thresholds = np.sort(thresholds)[::-1]

    sweep_records: list[dict] = []
    best_f1_record: dict | None = None
    best_f1_key: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        y_pred_bin = (scores >= threshold).astype(np.int64)
        tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
        fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
        tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())

        recall = float(tp / max(1, tp + fn))
        precision = float(tp / max(1, tp + fp))
        false_alarm_rate = float(fp / max(1, fp + tn))
        f1 = float(0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall))

        record = {
            "threshold": float(threshold),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "false_alarm_rate": false_alarm_rate,
        }
        sweep_records.append(record)

        record_key = (f1, recall, -false_alarm_rate, float(threshold))
        if best_f1_key is None or record_key > best_f1_key:
            best_f1_key = record_key
            best_f1_record = record

    def _best_under_far(max_far: float) -> dict:
        eligible = [record for record in sweep_records if record["false_alarm_rate"] <= max_far + 1e-12]
        if not eligible:
            return {"recall": 0.0, "threshold": 1.0}
        return max(
            eligible,
            key=lambda record: (
                record["recall"],
                -record["false_alarm_rate"],
                record["precision"],
                record["threshold"],
            ),
        )

    far_1 = _best_under_far(0.01)
    far_5 = _best_under_far(0.05)

    return {
        "pr_auc": float(average_precision_score(y_true_bin, scores)),
        "roc_auc": float(roc_auc_score(y_true_bin, scores)),
        "best_f1": float(best_f1_record["f1"]) if best_f1_record is not None else 0.0,
        "best_f1_threshold": float(best_f1_record["threshold"]) if best_f1_record is not None else 1.0,
        "recall_at_far_1pct": float(far_1["recall"]),
        "threshold_at_far_1pct": float(far_1["threshold"]),
        "recall_at_far_5pct": float(far_5["recall"]),
        "threshold_at_far_5pct": float(far_5["threshold"]),
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
        "avg_attack_recall": avg_attack_recall,
        "attack_macro_precision": attack_macro_precision,
        "benign_false_alarm_rate": float(benign_false_alarms / benign_total),
        "attack_miss_rate": float(1.0 - avg_attack_recall),
        "confusion_matrix": cm.tolist(),
    }
    metrics.update(_compute_binary_score_metrics(y_true, y_score, benign_class))
    return metrics
