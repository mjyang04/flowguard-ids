from __future__ import annotations

import numpy as np


def compute_feature_importance(shap_values) -> np.ndarray:
    # shap_values: (n_samples, n_features) or list[(n_samples, n_features)]
    if isinstance(shap_values, list):
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        values = np.asarray(shap_values)
        if values.dtype == object:
            values = values.astype(np.float32)
        if values.ndim == 3:
            # Common SHAP layouts:
            # (samples, features, outputs) -> reduce over samples/outputs
            # (outputs, samples, features) -> reduce over outputs/samples
            if values.shape[-1] <= 8:
                importance = np.abs(values).mean(axis=(0, 2))
            elif values.shape[0] <= 8:
                importance = np.abs(values).mean(axis=(0, 1))
            else:
                importance = np.abs(values).mean(axis=0)
        else:
            importance = np.abs(values).mean(axis=0)
    return importance
