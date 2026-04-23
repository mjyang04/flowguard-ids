"""Two-stage cascade inference for FlowGuard IDS.

Stage-1 is a binary attack/benign gate. Stage-2 is a multiclass attack-type
classifier. The cascade output is:

    final[i] = benign_class               if stage1[i] == benign
    final[i] = stage2[i]                  otherwise

This mirrors the XI2S-IDS design (MDPI FI 2025, doi.org/10.3390/fi17010025):
a strong binary filter keeps false-positive rate low, and the multiclass
classifier specializes on the attack population.

The functions here are intentionally pure-numpy so the cascade logic can be
unit-tested without torch, sklearn, or trained artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TwoStageResult:
    """Aggregate artefacts produced by a single cascade inference pass."""

    final_predictions: np.ndarray
    stage1_predictions: np.ndarray
    stage2_predictions: np.ndarray
    stage1_scores: np.ndarray | None = None
    threshold: float = 0.5
    benign_class: int = 0
    # Light-weight gating statistics (safe to JSON-serialize).
    stats: dict[str, float] = field(default_factory=dict)


def binarize_scores(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold Stage-1 attack probabilities into {0, 1}.

    Args:
        scores: attack probabilities of shape ``(N,)``.
        threshold: decision threshold on ``P(attack)``; samples with
            ``score >= threshold`` are predicted as attack.

    Returns:
        ``np.ndarray[int64]`` of shape ``(N,)`` with values in ``{0, 1}``.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return (scores >= float(threshold)).astype(np.int64)


def combine_stages(
    stage1_binary: np.ndarray,
    stage2_multiclass: np.ndarray,
    benign_class: int = 0,
) -> np.ndarray:
    """Merge Stage-1 binary gating with Stage-2 multiclass refinement.

    Args:
        stage1_binary: Stage-1 predictions of shape ``(N,)``; values ``0`` or
            ``1`` (``0`` means benign, ``1`` means attack).
        stage2_multiclass: Stage-2 predictions of shape ``(N,)``; values in
            ``{0, ..., K-1}`` where ``benign_class`` is reserved.
        benign_class: integer label reserved for the benign class in the
            multiclass target space (default 0).

    Returns:
        Merged multiclass predictions of shape ``(N,)``.
    """
    stage1_binary = np.asarray(stage1_binary, dtype=np.int64).reshape(-1)
    stage2_multiclass = np.asarray(stage2_multiclass, dtype=np.int64).reshape(-1)
    if stage1_binary.shape != stage2_multiclass.shape:
        raise ValueError(
            "stage1 and stage2 predictions must align: "
            f"{stage1_binary.shape} vs {stage2_multiclass.shape}"
        )
    gated = np.where(stage1_binary == 0, benign_class, stage2_multiclass)
    return gated.astype(np.int64)


def gating_stats(
    y_true: np.ndarray,
    stage1_binary: np.ndarray,
    benign_class: int = 0,
) -> dict[str, float]:
    """Compute Stage-1 gating statistics against multiclass ground truth.

    The ground truth is multiclass; we binarize it as ``(y_true != benign)``
    to measure how well Stage-1 sifts attacks from benign traffic.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    stage1_binary = np.asarray(stage1_binary, dtype=np.int64).reshape(-1)
    if y_true.shape != stage1_binary.shape:
        raise ValueError(
            f"y_true and stage1 must align: {y_true.shape} vs {stage1_binary.shape}"
        )
    y_true_bin = (y_true != benign_class).astype(np.int64)
    n = int(y_true_bin.size)
    if n == 0:
        return {
            "stage1_attack_rate": 0.0,
            "stage1_tp": 0.0,
            "stage1_fp": 0.0,
            "stage1_tn": 0.0,
            "stage1_fn": 0.0,
            "stage1_recall": 0.0,
            "stage1_precision": 0.0,
            "stage1_far": 0.0,
        }
    tp = int(((stage1_binary == 1) & (y_true_bin == 1)).sum())
    fp = int(((stage1_binary == 1) & (y_true_bin == 0)).sum())
    tn = int(((stage1_binary == 0) & (y_true_bin == 0)).sum())
    fn = int(((stage1_binary == 0) & (y_true_bin == 1)).sum())
    pos = max(1, tp + fn)
    neg = max(1, tn + fp)
    pred_pos = tp + fp
    return {
        "stage1_attack_rate": float((tp + fp) / n),
        "stage1_tp": float(tp),
        "stage1_fp": float(fp),
        "stage1_tn": float(tn),
        "stage1_fn": float(fn),
        "stage1_recall": float(tp / pos),
        "stage1_precision": float(tp / pred_pos) if pred_pos > 0 else 0.0,
        "stage1_far": float(fp / neg),
    }


def run_two_stage(
    stage1_scores: np.ndarray | None,
    stage1_binary: np.ndarray | None,
    stage2_multiclass: np.ndarray,
    y_true: np.ndarray | None = None,
    threshold: float = 0.5,
    benign_class: int = 0,
) -> TwoStageResult:
    """Run the full cascade on pre-computed stage outputs.

    Provide either ``stage1_scores`` (attack probability, thresholded here)
    or a pre-thresholded ``stage1_binary`` vector.
    """
    if stage1_binary is None and stage1_scores is None:
        raise ValueError("Provide stage1_binary or stage1_scores")

    if stage1_binary is None:
        stage1_binary = binarize_scores(stage1_scores, threshold=threshold)
    stage1_binary = np.asarray(stage1_binary, dtype=np.int64).reshape(-1)
    stage2_multiclass = np.asarray(stage2_multiclass, dtype=np.int64).reshape(-1)

    final = combine_stages(stage1_binary, stage2_multiclass, benign_class=benign_class)

    stats: dict[str, float] = {}
    if y_true is not None:
        stats.update(gating_stats(y_true, stage1_binary, benign_class=benign_class))

    return TwoStageResult(
        final_predictions=final,
        stage1_predictions=stage1_binary,
        stage2_predictions=stage2_multiclass,
        stage1_scores=(
            np.asarray(stage1_scores, dtype=np.float64).reshape(-1)
            if stage1_scores is not None
            else None
        ),
        threshold=float(threshold),
        benign_class=int(benign_class),
        stats=stats,
    )
