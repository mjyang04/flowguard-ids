from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .feature_selector import select_top_k_features
from .importance import compute_feature_importance


class SHAPAnalyzer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def sample_data(
        self, X_train: np.ndarray, y_train: np.ndarray, n_samples: int = 2000
    ) -> np.ndarray:
        np.random.seed(42)
        classes = np.unique(y_train)
        samples_per_class = max(1, n_samples // max(1, len(classes)))

        indices = []
        for cls in classes:
            cls_indices = np.where(y_train == cls)[0]
            n_select = min(samples_per_class, len(cls_indices))
            selected = np.random.choice(cls_indices, size=n_select, replace=False)
            indices.extend(selected.tolist())

        if len(indices) < n_samples and len(indices) < len(X_train):
            remaining = list(set(range(len(X_train))) - set(indices))
            add_n = min(n_samples - len(indices), len(remaining))
            additional = np.random.choice(remaining, size=add_n, replace=False)
            indices.extend(additional.tolist())

        return X_train[np.array(indices, dtype=int)]

    def compute_shap_values(
        self, X_samples: np.ndarray, background_size: int = 100
    ):
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is not installed. Please install shap first.") from exc

        self.model.eval()
        background_size = min(background_size, len(X_samples))
        background = torch.from_numpy(X_samples[:background_size]).float().to(self.device)
        inputs = torch.from_numpy(X_samples).float().to(self.device)

        def _wrap_for_binary_output(base_model: nn.Module, device: torch.device, bg: torch.Tensor) -> nn.Module:
            wrapped: nn.Module = base_model
            with torch.no_grad():
                probe = base_model(bg[:1])
                if probe.dim() == 1:
                    class _SHAPBinaryOutputWrapper(nn.Module):
                        def __init__(self, model: nn.Module):
                            super().__init__()
                            self.model = model

                        def forward(self, x: torch.Tensor) -> torch.Tensor:
                            out = self.model(x)
                            if out.dim() == 1:
                                out = out.unsqueeze(1)
                            return out

                    wrapped = _SHAPBinaryOutputWrapper(base_model).to(device)
            wrapped.eval()
            return wrapped

        def _has_recurrent_module(model: nn.Module) -> bool:
            recurrent_types = (nn.RNN, nn.GRU, nn.LSTM, nn.RNNBase)
            return any(isinstance(module, recurrent_types) for module in model.modules())

        def _run_gradient_explainer(model: nn.Module, bg: torch.Tensor, xs: torch.Tensor):
            explainer = shap.GradientExplainer(model, bg)
            return explainer.shap_values(xs)

        def _run_kernel_explainer(model: nn.Module, bg: torch.Tensor, xs: torch.Tensor):
            model_cpu = model.to("cpu").eval()

            def _predict_fn(arr: np.ndarray) -> np.ndarray:
                tensor = torch.from_numpy(arr.astype(np.float32))
                with torch.no_grad():
                    out = model_cpu(tensor)
                    if out.dim() == 1:
                        out = out.unsqueeze(1)
                    return out.cpu().numpy()

            bg_np = bg.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
            if xs_np.shape[0] > 200:
                xs_np = xs_np[:200]
            explainer = shap.KernelExplainer(_predict_fn, bg_np)
            return explainer.shap_values(xs_np, nsamples=min(100, max(20, xs_np.shape[0])))

        if _has_recurrent_module(self.model):
            # DeepExplainer is fragile on recurrent modules; use more stable explainers directly.
            model_cpu = copy.deepcopy(self.model).to("cpu")
            model_cpu.eval()
            background_cpu = torch.from_numpy(X_samples[:background_size]).float()
            inputs_cpu = torch.from_numpy(X_samples).float()
            model_cpu = _wrap_for_binary_output(model_cpu, torch.device("cpu"), background_cpu)
            try:
                return _run_gradient_explainer(model_cpu, background_cpu, inputs_cpu)
            except Exception:
                return _run_kernel_explainer(model_cpu, background_cpu, inputs_cpu)

        model_for_shap = _wrap_for_binary_output(self.model, self.device, background)

        try:
            # DeepExplainer is preferred for neural networks.
            with torch.backends.cudnn.flags(enabled=False):
                explainer = shap.DeepExplainer(model_for_shap, background)
                # LSTM/unsupported ops may fail strict additivity checks in SHAP.
                return explainer.shap_values(inputs, check_additivity=False)
        except (RuntimeError, AssertionError):
            # Fallback: run SHAP on CPU for recurrent models that still fail on CUDA.
            model_cpu = copy.deepcopy(self.model).to("cpu")
            model_cpu.eval()
            background_cpu = torch.from_numpy(X_samples[:background_size]).float()
            inputs_cpu = torch.from_numpy(X_samples).float()
            model_cpu = _wrap_for_binary_output(model_cpu, torch.device("cpu"), background_cpu)

            try:
                explainer = shap.DeepExplainer(model_cpu, background_cpu)
                return explainer.shap_values(inputs_cpu, check_additivity=False)
            except (RuntimeError, AssertionError):
                try:
                    return _run_gradient_explainer(model_cpu, background_cpu, inputs_cpu)
                except Exception:
                    # Final fallback for unsupported internals.
                    return _run_kernel_explainer(model_cpu, background_cpu, inputs_cpu)

    def compute_feature_importance(self, shap_values) -> np.ndarray:
        return compute_feature_importance(shap_values)

    def select_top_k(
        self, importance: np.ndarray, feature_names: List[str], k: int = 30
    ) -> List[str]:
        selected_features, _, _ = select_top_k_features(feature_names, importance, k=k)
        return selected_features
