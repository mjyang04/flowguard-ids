"""Contract tests for :mod:`nids.data.cicids`.

Uses a tiny synthetic fixture that mirrors the real CICIDS2017 schema
quirks: leading-space ``" Label"`` column, upper-case label values, an
en-dash unicode hyphen in ``Web Attack – Brute Force``, and Inf values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids.data.cicids import (
    _canonicalise_label,
    _normalise_column_name,
    get_data_cicids,
    load_cicids_dataframe,
)


def _write_synthetic_cicids_csv(path: Path, n_benign: int = 200, n_attack: int = 80) -> None:
    """Write a small CICIDS2017-shaped CSV with quirky schema."""
    rng = np.random.default_rng(0)

    rows: list[dict] = []
    # Benign rows.
    for i in range(n_benign):
        rows.append({
            "Flow ID": f"flow-{i}",
            "Source IP": "10.0.0.1",
            " Source Port": 1000 + (i % 50),
            " Destination IP": "10.0.0.2",
            " Destination Port": 80,
            " Protocol": 6,
            " Timestamp": "01/01/2017 10:00:00",
            " Flow Duration": float(rng.integers(0, 1000000)),
            " Total Fwd Packets": int(rng.integers(1, 100)),
            " Total Backward Packets": int(rng.integers(1, 100)),
            "Flow Bytes/s": float(rng.uniform(100, 10000)),
            " Flow Packets/s": float(rng.uniform(1, 1000)),
            " Always Zero": 0.0,                       # all-zero column to drop
            " Label": "BENIGN",
        })
    # Attack rows with an en-dash label (Unicode en-dash U+2013).
    for i in range(n_attack):
        rows.append({
            "Flow ID": f"flow-att-{i}",
            "Source IP": "10.0.0.3",
            " Source Port": 2000 + (i % 30),
            " Destination IP": "10.0.0.4",
            " Destination Port": 80,
            " Protocol": 6,
            " Timestamp": "01/01/2017 11:00:00",
            " Flow Duration": float(rng.integers(0, 100)),
            " Total Fwd Packets": int(rng.integers(1, 50)),
            " Total Backward Packets": int(rng.integers(1, 50)),
            "Flow Bytes/s": float("inf") if i == 0 else float(rng.uniform(100, 10000)),
            " Flow Packets/s": float(rng.uniform(1, 1000)),
            " Always Zero": 0.0,
            " Label": "Web Attack – Brute Force",
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")


def test_normalise_column_name_strips_and_snake_cases() -> None:
    assert _normalise_column_name(" Source IP") == "source_ip"
    assert _normalise_column_name("Flow Bytes/s") == "flow_bytes_per_s"
    assert _normalise_column_name(" Label") == "label"
    assert _normalise_column_name("Fwd PSH Flags") == "fwd_psh_flags"


def test_canonicalise_label_handles_endash_and_mojibake() -> None:
    assert _canonicalise_label("BENIGN") == "benign"
    assert _canonicalise_label("DDoS") == "ddos"
    assert _canonicalise_label("FTP-Patator") == "ftp_patator"
    # Unicode en-dash.
    assert (
        _canonicalise_label("Web Attack – Brute Force") == "web_attack_brute_force"
    )
    # Replacement character (mojibake).
    assert (
        _canonicalise_label("Web Attack � Brute Force") == "web_attack_brute_force"
    )


def test_load_cicids_dataframe_normalises_columns(tmp_path: Path) -> None:
    csv = tmp_path / "cic.csv"
    _write_synthetic_cicids_csv(csv)
    df = load_cicids_dataframe(csv)

    assert "label" in df.columns
    assert " Label" not in df.columns
    assert "source_ip" in df.columns
    assert "flow_bytes_per_s" in df.columns
    assert set(df["label"].unique()) == {"benign", "web_attack_brute_force"}


def test_load_cicids_dataframe_prefers_extracted_csvs(tmp_path: Path) -> None:
    csv = tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    _write_synthetic_cicids_csv(csv)
    (tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.zip").write_text(
        "not a zip",
        encoding="utf-8",
    )
    df = load_cicids_dataframe(tmp_path)

    assert "label" in df.columns
    assert set(df["label"].unique()) == {"benign", "web_attack_brute_force"}


def test_get_data_cicids_returns_datasplits(tmp_path: Path) -> None:
    csv = tmp_path / "cic.csv"
    _write_synthetic_cicids_csv(csv)
    splits = get_data_cicids(csv, sample_thres=10, split_seed=42, anomaly_detection=False)

    assert splits.x_train.ndim == 2
    assert splits.x_train.shape[1] == splits.input_dim
    assert splits.input_dim > 0
    assert "benign" in splits.class_names


def test_get_data_cicids_drops_zero_columns(tmp_path: Path) -> None:
    csv = tmp_path / "cic.csv"
    _write_synthetic_cicids_csv(csv)
    splits = get_data_cicids(csv, sample_thres=10, split_seed=42, anomaly_detection=False)

    # " Always Zero" normalises to "always_zero"; should have been dropped.
    assert "always_zero" not in splits.feature_names


def test_get_data_cicids_handles_inf_values(tmp_path: Path) -> None:
    csv = tmp_path / "cic.csv"
    _write_synthetic_cicids_csv(csv)
    splits = get_data_cicids(csv, sample_thres=10, split_seed=42, anomaly_detection=False)

    # Inf was inserted into flow_bytes_per_s for attack row 0; the loader
    # should treat it as missing and still return finite model inputs.
    assert np.isfinite(splits.x_train).all()
    assert np.isfinite(splits.x_test).all()


def test_get_data_cicids_log_transforms_extreme_numeric_values(tmp_path: Path) -> None:
    csv = tmp_path / "cic.csv"
    _write_synthetic_cicids_csv(csv)
    df = pd.read_csv(csv)
    df.loc[0, " Flow Duration"] = 1.0e30
    df.to_csv(csv, index=False, encoding="utf-8")

    splits = get_data_cicids(csv, sample_thres=10, split_seed=42, anomaly_detection=False)

    assert np.isfinite(splits.x_train).all()
    assert np.isfinite(splits.x_test).all()


def test_get_data_cicids_anomaly_detection_filters_benign(tmp_path: Path) -> None:
    csv = tmp_path / "cic.csv"
    _write_synthetic_cicids_csv(csv)
    splits = get_data_cicids(csv, sample_thres=10, split_seed=42, anomaly_detection=True)

    assert (splits.y_train == 0).all()


def test_get_data_cicids_missing_source_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_data_cicids(tmp_path / "does-not-exist.csv")
