"""Data loading for Lycos2017 (CLAN reproduction)."""

from .lycos import DataSplits, get_data
from .utils import sample_data

__all__ = ["DataSplits", "get_data", "sample_data"]

try:
    from .loaders import tabular_dl
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    tabular_dl = None  # type: ignore[assignment]
else:
    __all__.append("tabular_dl")
