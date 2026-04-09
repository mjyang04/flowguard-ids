__all__: list[str] = []

try:
    from .auc_loss import pairwise_auc_loss
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("pairwise_auc_loss")

try:
    from .trainer import EvaluationResult, Trainer, TrainingSummary
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(["Trainer", "TrainingSummary", "EvaluationResult"])
