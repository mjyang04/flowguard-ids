from .metrics import compute_nids_metrics

__all__ = ["compute_nids_metrics"]

try:
    from .latency import measure_inference_latency
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("measure_inference_latency")

try:
    from .evaluator import evaluate_model
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.insert(0, "evaluate_model")
