"""Warmup-cosine learning rate schedule.

Ported from https://github.com/jackwilkie/CLAN/blob/main/util/schedules.py
(Apache-2.0).
"""

from __future__ import annotations

import math
from typing import Any


class Schedule:
    """Generic step-based schedule. Subclasses implement :meth:`update`."""

    def __init__(
        self,
        start_val: float,
        end_val: float,
        T_max: int,
        step_every: int = 1,
    ) -> None:
        """Initialise a schedule bounded by start/end values and max steps."""
        if step_every < 1:
            raise ValueError(f"step_every must be >= 1, got {step_every}")
        self._steps = 0
        self.start_val = start_val
        self.end_val = end_val
        self.T_max = T_max
        self.step_every = step_every
        self.val = start_val

    def step(self, n_steps: int = 1) -> float:
        """Advance the internal counter and return the current schedule value."""
        for _ in range(n_steps):
            self._steps += 1
            if self._steps % self.step_every == 0:
                self.val = self.update()
        return self.val

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state for checkpointing the schedule."""
        return {
            "_steps": self._steps,
            "start_val": self.start_val,
            "end_val": self.end_val,
            "T_max": self.T_max,
            "val": self.val,
            "step_every": self.step_every,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore schedule state from a checkpoint payload."""
        self._steps = state_dict["_steps"]
        self.start_val = state_dict["start_val"]
        self.end_val = state_dict["end_val"]
        self.T_max = state_dict["T_max"]
        self.val = state_dict["val"]
        self.step_every = state_dict["step_every"]

    def reset(self) -> None:
        """Reset the schedule to its initial value and zero steps."""
        self._steps = 0
        self.val = self.start_val

    def update(self) -> float:
        """Compute the next value; subclasses must implement this method."""
        raise NotImplementedError


class WarmupCosineSchedule(Schedule):
    """Linear warmup followed by cosine annealing to ``end_val``."""

    def __init__(
        self,
        start_val: float,
        end_val: float,
        T_max: int,
        ref_val: float,
        warmup_steps: int,
        step_every: int = 1,
        plateau: bool = True,
    ) -> None:
        """Create a warmup-cosine schedule with optional post-run plateau."""
        super().__init__(
            start_val=start_val, end_val=end_val, T_max=T_max, step_every=step_every
        )
        self.ref_val = ref_val
        self.warmup_steps = warmup_steps
        self.plateau = plateau

    def update(self) -> float:
        """Compute the current warmup or cosine-annealed value."""
        if self.plateau and self._steps > self.T_max:
            return self.end_val
        if self._steps < self.warmup_steps:
            progress = float(self._steps) / float(max(1, self.warmup_steps))
            return self.start_val + progress * (self.ref_val - self.start_val)
        progress = float(self._steps - self.warmup_steps) / float(
            max(1, self.T_max - self.warmup_steps)
        )
        val = self.end_val + (self.ref_val - self.end_val) * 0.5 * (
            1.0 + math.cos(progress * math.pi)
        )
        if self.ref_val <= self.end_val:
            val = min(self.end_val, val)
        else:
            val = max(self.end_val, val)
        return val


class LRSchedule:
    """Apply a :class:`Schedule` to optimiser LRs."""

    def __init__(self, optimizer: Any, schedule: Schedule) -> None:
        """Bind a scalar schedule to all non-excluded optimiser param groups."""
        self.optimizer = optimizer
        self.schedule = schedule
        self.scale = 1.0
        self._apply(schedule.start_val)

    def _apply(self, lr: float) -> None:
        """Write a learning rate to optimiser groups, respecting local scales."""
        for group in self.optimizer.param_groups:
            if not group.get("lr_exclude", False):
                ls = group.get("layer_scale", 1.0)
                group["lr"] = lr * self.scale * ls

    def step(self, n_steps: int = 1) -> float:
        """Advance the schedule and apply the resulting learning rate."""
        val = self.schedule.step(n_steps)
        self._apply(val)
        return val

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable scheduler wrapper state for checkpoints."""
        return {"scale": self.scale, "schedule": self.schedule.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore wrapper and base-schedule state, then reapply the LR."""
        self.scale = state_dict["scale"]
        self.schedule.load_state_dict(state_dict["schedule"])
        self._apply(self.schedule.val)
