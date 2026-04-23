"""Distance metrics used by CLANLoss and the centroid-based anomaly score.

Ported from https://github.com/jackwilkie/CLAN/blob/main/util/distance.py
(Apache-2.0). Minor cleanups:

* guard ``edist`` against ``b is None``
* type hints everywhere
* allow ``chunk_size=None`` to mean "one-shot computation"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def cosdist(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Cosine distance rescaled to [0, 1] (1 means opposite directions)."""
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1) if b is not None else a_norm
    similarity = torch.mm(a_norm, b_norm.T)
    return (1.0 - similarity) / 2.0


def cossim(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Cosine similarity rescaled to [0, 1]."""
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1) if b is not None else a_norm
    similarity = torch.mm(a_norm, b_norm.T)
    return (1.0 + similarity) / 2.0


def edist(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Pairwise Euclidean distance."""
    b = b if b is not None else a
    return torch.cdist(a, b)


def esim(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Euclidean similarity (negative distance)."""
    return -1.0 * edist(a, b)


def cosdist_inference(centroid: Tensor, x: Tensor) -> Tensor:
    """Negative cosine similarity to a single centroid (anomaly score)."""
    return -1.0 * F.cosine_similarity(centroid.unsqueeze(0), x, dim=1).squeeze().detach()


def chunked_centroid_sims(
    embeddings: Tensor,
    centroid: Tensor,
    chunk_size: Optional[int] = 1024,
) -> np.ndarray:
    """Cosine similarity between every row in ``embeddings`` and ``centroid``.

    Falls back to a one-shot computation when ``chunk_size`` is None.
    """
    if chunk_size is None:
        sims = F.cosine_similarity(centroid.unsqueeze(0), embeddings, dim=1).squeeze()
        return sims.cpu().detach().numpy()

    n_rows = embeddings.size(0)
    out: list[np.ndarray] = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = embeddings[start:end]
        chunk_sims = F.cosine_similarity(centroid.unsqueeze(0), chunk, dim=1).squeeze()
        out.append(chunk_sims.cpu().detach().numpy())
    return np.concatenate(out)
