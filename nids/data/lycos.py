"""Lycos2017 CSV loader.

Thin wrapper around :func:`nids.data._core.prepare_splits`. The upstream
CLAN scripts expect this exact call signature:

    x_train, y_train, x_val, y_val, x_test, y_test, x_zd, y_zd = get_data(
        data_path, target, drop, class_zero, sample_thres, split_seed,
        test_ratio, val_ratio, anomaly_detection,
    )

Re-exports :class:`DataSplits` for backwards compatibility with tests that
imported it from this module before the core was extracted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ._core import DataSplits, prepare_splits

__all__ = ["DataSplits", "get_data"]


def get_data(
    data_path: str | Path,
    *,
    target: str = "label",
    drop: Iterable[str] = (
        "flow_id",
        "src_addr",
        "src_port",
        "dst_addr",
        "dst_port",
        "ip_prot",
        "timestamp",
    ),
    class_zero: str = "benign",
    sample_thres: int = 100,
    split_seed: int = 39058032,
    test_ratio: float = 0.5,
    val_ratio: float = 0.0,
    anomaly_detection: bool = True,
    standardise: bool = True,
) -> DataSplits:
    """Load the Lycos2017 CSV and produce the 8 splits consumed by CLAN scripts.

    Behaviour mirrors the paper (Rosay et al., 2022) and the upstream CLAN
    reference implementation: the 7 metadata columns are dropped,
    ``class_zero`` is mapped to class 0, attacks below ``sample_thres``
    become the zero-day holdout, and features are standardised using
    training-split statistics.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Lycos CSV not found at {path}")

    df = pd.read_csv(path)
    return prepare_splits(
        df,
        target=target,
        drop=drop,
        class_zero=class_zero,
        sample_thres=sample_thres,
        split_seed=split_seed,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        anomaly_detection=anomaly_detection,
        standardise=standardise,
        drop_zero_cols=True,
    )
