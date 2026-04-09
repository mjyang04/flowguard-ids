"""Pairwise AUC surrogate loss for cross-dataset ranking improvement.

Based on Xin & Xu (2025) "Cross-Dataset Transformer-IDS with Calibration
and AUC Optimization", which combines BCE with a hinge-based pairwise loss
that optimizes the Wilcoxon-Mann-Whitney statistic.

The auxiliary loss encourages attack samples to have higher scores than
benign samples, improving ranking performance across decision thresholds.
"""

from __future__ import annotations

import torch


def pairwise_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    num_neg: int = 5,
) -> torch.Tensor:
    """Compute pairwise hinge AUC surrogate loss for binary classification.

    For each positive (attack) sample, randomly sample `num_neg` negative
    (benign) samples and compute a hinge loss encouraging the positive
    logit to exceed the negative logit by at least `margin`.

    Args:
        logits: Raw model output scores, shape (batch,).
        labels: Binary labels (0=benign, 1=attack), shape (batch,).
        margin: Hinge margin for the pairwise comparison.
        num_neg: Number of negative samples per positive.

    Returns:
        Scalar pairwise AUC loss. Returns 0 if no valid pairs exist.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]

    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return logits.new_tensor(0.0)

    # Sample negatives for each positive
    n_neg = min(num_neg, neg_logits.numel())
    indices = torch.randint(0, neg_logits.numel(), (pos_logits.numel(), n_neg), device=logits.device)
    sampled_neg = neg_logits[indices]  # (n_pos, n_neg)

    # Hinge loss: max(0, margin - (pos - neg))
    diff = pos_logits.unsqueeze(1) - sampled_neg  # (n_pos, n_neg)
    loss = torch.clamp(margin - diff, min=0.0).mean()

    return loss
