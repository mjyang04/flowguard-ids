"""Contract tests for the Lycos2017 CSV loader.

These tests use pandas + numpy + scikit-learn only (no torch), so they
run locally on the Mac dev environment as well as on the Windows training
rig. They write a tiny synthetic CSV to ``tmp_path`` and assert that
``nids.data.get_data`` honours the documented contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids.data.lycos import DataSplits, get_data
from nids.data.utils import sample_data


def _make_synthetic_csv(
    path: Path,
    n_benign: int = 200,
    n_ddos: int = 60,
    n_botnet: int = 60,
    n_heartbleed: int = 5,
) -> None:
    """Write a minimised Lycos-style CSV to ``path``."""
    rng = np.random.default_rng(0)
    rows = []
    for i, label in enumerate(
        ["benign"] * n_benign
        + ["ddos"] * n_ddos
        + ["botnet"] * n_botnet
        + ["heartbleed"] * n_heartbleed
    ):
        rows.append(
            {
                "flow_id": f"f_{i:06d}",
                "src_addr": "10.0.0.1",
                "src_port": 54321,
                "dst_addr": "10.0.0.2",
                "dst_port": 80,
                "ip_prot": 6,
                "timestamp": "2017-07-03 09:15:00",
                "flow_duration": float(rng.uniform(0, 1000)),
                "tot_fwd_pkts": float(rng.integers(1, 100)),
                "flow_iat_mean": float(rng.uniform(0, 10)),
                "active_mean": float(rng.uniform(0, 1)),
                "label": label,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_get_data_returns_datasplits(tmp_path: Path) -> None:
    csv = tmp_path / "lycos.csv"
    _make_synthetic_csv(csv)
    splits = get_data(
        csv, split_seed=42, test_ratio=0.5, val_ratio=0.0, anomaly_detection=True
    )
    assert isinstance(splits, DataSplits)


def test_get_data_drops_metadata_columns(tmp_path: Path) -> None:
    csv = tmp_path / "lycos.csv"
    _make_synthetic_csv(csv)
    splits = get_data(csv, split_seed=42, anomaly_detection=False)
    for meta in (
        "flow_id",
        "src_addr",
        "src_port",
        "dst_addr",
        "dst_port",
        "ip_prot",
        "timestamp",
    ):
        assert meta not in splits.feature_names


def test_get_data_benign_label_maps_to_zero(tmp_path: Path) -> None:
    csv = tmp_path / "lycos.csv"
    _make_synthetic_csv(csv)
    splits = get_data(csv, split_seed=42, anomaly_detection=False)
    assert splits.class_names[0] == "benign"


def test_get_data_anomaly_detection_keeps_only_benign_in_train(tmp_path: Path) -> None:
    csv = tmp_path / "lycos.csv"
    _make_synthetic_csv(csv)
    splits = get_data(csv, split_seed=42, anomaly_detection=True)
    assert np.all(
        splits.y_train == 0
    ), "train split must contain only benign when anomaly_detection=True"


def test_get_data_zero_day_holdout(tmp_path: Path) -> None:
    """heartbleed has only 5 rows, below the threshold of 100 -> zero-day."""
    csv = tmp_path / "lycos.csv"
    _make_synthetic_csv(csv, n_heartbleed=5)
    splits = get_data(csv, split_seed=42, sample_thres=100, anomaly_detection=False)
    assert splits.x_zd.shape[0] >= 1
    zd_classes = set(int(c) for c in splits.y_zd)
    train_classes = set(int(c) for c in splits.y_train)
    test_classes = set(int(c) for c in splits.y_test)
    assert zd_classes.isdisjoint(train_classes)
    assert zd_classes.isdisjoint(test_classes)


def test_get_data_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_data(tmp_path / "does_not_exist.csv")


def test_get_data_standardises_when_requested(tmp_path: Path) -> None:
    csv = tmp_path / "lycos.csv"
    _make_synthetic_csv(csv)
    splits = get_data(csv, split_seed=42, standardise=True, anomaly_detection=False)
    means = splits.x_train.mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-4)


def test_sample_data_balanced_subset(tmp_path: Path) -> None:
    csv = tmp_path / "lycos.csv"
    _make_synthetic_csv(csv)
    # sample_thres=1 so ddos/botnet (60 rows each) stay in dev instead of zero-day.
    splits = get_data(csv, split_seed=42, sample_thres=1, anomaly_detection=False)
    x_sub, y_sub = sample_data(
        splits.x_train,
        splits.y_train,
        num_benign=8,
        num_mal=8,
        sample_seed=42,
    )
    assert x_sub.shape[0] == 8 + 8
    assert y_sub.shape[0] == 8 + 8
    assert (y_sub == 0).sum() == 8
