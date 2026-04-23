from .metrics import compute_nids_metrics

__all__ = ["compute_nids_metrics"]

try:
    from .latency import measure_inference_latency
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("measure_inference_latency")
