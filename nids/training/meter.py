"""Running-average meter used by the training loop.

Ported from https://github.com/jackwilkie/CLAN/blob/main/util/meter.py
(Apache-2.0).
"""

from __future__ import annotations


class AverageMeter:
    """Tracks current value and running mean of a scalar."""

    def __init__(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
