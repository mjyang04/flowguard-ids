from __future__ import annotations

import torch


def build_optimizer(
    model: torch.nn.Module,
    name: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "plateau",
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 2,
    min_learning_rate: float = 1e-6,
):
    key = name.lower()
    if key == "none":
        return None
    if key == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_learning_rate,
        )
    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    raise ValueError(f"Unsupported scheduler: {name}")
