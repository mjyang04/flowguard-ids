"""Data loading for NIDS experiments.

Supports two datasets for the label-quality comparison study:

* ``lycos2017`` — Rosay et al.'s cleaned relabel of CICIDS2017.
* ``cicids2017`` — original CIC-IDS2017 with its well-documented label
  and feature-extraction issues preserved intentionally as a control.
"""

from ._core import DataSplits, prepare_splits
from .cicids import CICIDS_META_COLS, get_data_cicids, load_cicids_dataframe
from .lycos import get_data
from .utils import sample_data

__all__ = [
    "DataSplits",
    "prepare_splits",
    "get_data",
    "get_data_cicids",
    "load_cicids_dataframe",
    "CICIDS_META_COLS",
    "sample_data",
]

try:
    from .loaders import tabular_dl
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    tabular_dl = None  # type: ignore[assignment]
else:
    __all__.append("tabular_dl")


def get_data_by_name(dataset: str, *args, **kwargs) -> DataSplits:
    """Dispatch to the dataset-specific loader by name."""
    name = dataset.lower().replace("-", "").replace("_", "")
    if name in {"lycos", "lycos2017", "lycosids2017"}:
        return get_data(*args, **kwargs)
    if name in {"cicids", "cicids2017", "cicids17"}:
        return get_data_cicids(*args, **kwargs)
    raise ValueError(
        f"Unknown dataset: {dataset!r}. Supported: 'lycos2017', 'cicids2017'."
    )


__all__.append("get_data_by_name")
