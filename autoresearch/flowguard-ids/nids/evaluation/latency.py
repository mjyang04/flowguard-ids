from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader


def measure_inference_latency(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
) -> dict:
    model.eval()
    latencies = []
    batch_sizes = []

    with torch.no_grad():
        for i, (features, _) in enumerate(dataloader):
            if i >= n_batches:
                break

            features = features.to(device)
            batch_sizes.append(features.shape[0])

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(features)

            if device.type == "cuda":
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start) * 1000)

    if not latencies:
        return {"mean_latency_ms": 0.0, "p50_latency_ms": 0.0, "p99_latency_ms": 0.0, "throughput": 0.0}

    total_samples = float(np.sum(batch_sizes))
    total_time_s = float(np.sum(latencies) / 1000.0)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput": float(total_samples / total_time_s) if total_time_s > 0 else 0.0,
    }
