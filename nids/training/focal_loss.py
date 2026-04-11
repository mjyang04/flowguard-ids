"""Focal Loss for class-imbalanced binary and multi-class classification.

Based on Lin et al. (2017) "Focal Loss for Dense Object Detection".
Reduces the relative loss for well-classified examples, focusing training
on hard negatives. Widely adopted in IDS literature (Zhang et al. 2025,
Li & Li 2025 TCNSE) for handling class imbalance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification with logits input.

    Args:
        alpha: Weighting factor for the positive class (default 0.25).
        gamma: Focusing parameter (default 2.0). Higher values down-weight
            easy examples more aggressively.
        pos_weight: Optional positive class weight (like BCEWithLogitsLoss).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # p_t = p for positive, (1-p) for negative
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t = alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Use BCE with logits for numerical stability
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )

        loss = focal_weight * bce
        return loss.mean()
