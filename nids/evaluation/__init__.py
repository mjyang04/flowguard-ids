from .metrics import compute_nids_metrics
from .two_stage import (
    TwoStageResult,
    binarize_scores,
    combine_stages,
    gating_stats,
    run_two_stage,
)

__all__ = [
    "compute_nids_metrics",
    "TwoStageResult",
    "binarize_scores",
    "combine_stages",
    "gating_stats",
    "run_two_stage",
]

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

try:
    from .calibration import PlattCalibrator, collect_logits
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(["PlattCalibrator", "collect_logits"])
