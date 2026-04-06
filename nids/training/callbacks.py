from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 5
    delta: float = 1e-4
    mode: str = "max"

    def __post_init__(self) -> None:
        self.best = float("-inf") if self.mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.mode == "max":
            improved = value > self.best + self.delta
        else:
            improved = value < self.best - self.delta

        if improved:
            self.best = value
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved
