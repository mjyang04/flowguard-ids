from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_nids_metrics


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
    criterion: torch.nn.Module | None = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_scores: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            if criterion is not None:
                if num_classes == 2:
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs, labels)
                total_loss += float(loss.item())

            if num_classes == 2:
                probs = torch.sigmoid(outputs)
                probs = probs.reshape(-1)
                preds = (probs >= 0.5).long()
                all_scores.extend(probs.cpu().numpy().tolist())
            else:
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_score = np.array(all_scores, dtype=np.float32) if all_scores else None
    metrics = compute_nids_metrics(np.array(all_labels), np.array(all_preds), y_score=y_score)
    if criterion is not None and len(data_loader) > 0:
        metrics["loss"] = total_loss / len(data_loader)
    return metrics
