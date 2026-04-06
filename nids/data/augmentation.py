from __future__ import annotations

import numpy as np


def gaussian_noise(X: np.ndarray, std: float = 0.01, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, std, size=X.shape)
    return (X + noise).astype(np.float32)
