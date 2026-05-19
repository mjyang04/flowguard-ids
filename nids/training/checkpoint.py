"""Checkpoint save / load helpers.

Simplified from https://github.com/jackwilkie/CLAN/blob/main/util/checkpoint.py
(Apache-2.0). The DDP-specific branch and the ``distributed_to_local``
state-dict remapping have been dropped — neither is relevant to single-GPU
3060 training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _ensure_extension(path: str | Path, extension: str) -> Path:
    """Return ``path`` with the checkpoint extension appended when missing."""
    p = Path(path)
    if not str(p).endswith(extension):
        p = Path(str(p) + extension)
    return p


def make_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    stats: dict[str, Any] | None = None,
    **extra: Any,
) -> Path:
    """Save model weights + (optional) optimiser/scheduler/stats to ``path``."""
    p = _ensure_extension(path, ".pt.tar")
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "stats": stats,
    }
    payload.update(extra)
    torch.save(payload, p)
    return p


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights (and optional optimiser/scheduler/scaler state) into the
    in-memory objects. Returns the raw payload for callers that want stats."""
    p = _ensure_extension(path, ".pt.tar")
    payload = torch.load(p, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    return payload
