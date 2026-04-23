"""CLAN loss — contrastive learning with augmented negatives.

Ported from https://github.com/jackwilkie/CLAN/blob/main/loss/clan_loss.py
(Apache-2.0).

Given original features ``x`` and augmented features ``x_aug`` (both
L2-normalised in the last dim), the loss has two terms:

1. *Intra-class* (original-to-original): pulls benign samples toward a
   shared centroid.
2. *Inter-class* (original-to-augmented): pushes the augmented (treated
   as negative) away up to a margin ``m``.

The final loss is the ``alpha``-weighted average of the two.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nids.training.distance import cosdist, edist


def _resolve_distance(name: str) -> Callable[..., Tensor]:
    if name == "cosine":
        return cosdist
    if name == "euclidean":
        return edist
    raise ValueError(f"Invalid distance metric: {name!r}. Expected 'cosine' or 'euclidean'.")


def clan_loss(
    z: Tensor,
    z_aug: Tensor,
    m: float = 1.0,
    alpha: float = 0.5,
    squared: bool = False,
    distance_metric: Callable[..., Tensor] = cosdist,
    eps: float = 1e-16,
    return_frac_pos: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    """Functional CLAN loss.

    See :class:`CLANLoss` for details.
    """
    intra = distance_metric(z)
    if squared:
        intra = intra.pow(2)
    n_sim = torch.greater(intra, eps).float().sum()
    sim_loss = intra.sum() / (n_sim + eps)

    inter = distance_metric(z, z_aug)
    dissim = F.relu(m - inter)
    if squared:
        dissim = dissim.pow(2)
    n_dissim = torch.greater(dissim, eps).float().sum()
    dissim_loss = dissim.sum() / (n_dissim + eps)

    loss = alpha * sim_loss + (1.0 - alpha) * dissim_loss

    if return_frac_pos:
        frac = n_dissim / (z.size(0) * z_aug.size(0) + eps)
        return loss, frac
    return loss


class CLANLoss(nn.Module):
    """Module wrapper around :func:`clan_loss`."""

    def __init__(
        self,
        m: float = 1.0,
        loss_alpha: float = 0.5,
        squared: bool = False,
        distance_metric: str = "cosine",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 <= loss_alpha <= 1.0:
            raise ValueError(f"loss_alpha must be in [0, 1]; got {loss_alpha}")
        self.m = m
        self.loss_alpha = loss_alpha
        self.squared = squared
        self.eps = eps
        self.distance_metric_name = distance_metric
        self._distance_fn = _resolve_distance(distance_metric)

    def forward(self, x: Tensor, x_aug: Tensor) -> tuple[Tensor, Tensor]:
        return clan_loss(  # type: ignore[return-value]
            z=x,
            z_aug=x_aug,
            m=self.m,
            alpha=self.loss_alpha,
            squared=self.squared,
            distance_metric=self._distance_fn,
            eps=self.eps,
            return_frac_pos=True,
        )
