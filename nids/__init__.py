"""Lightweight NIDS package: CNN-BiLSTM-SE + Transformer variants + two-stage cascade + SHAP-driven Top-K + Platt calibration."""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
