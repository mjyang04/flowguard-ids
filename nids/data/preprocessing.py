from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight as sklearn_compute_class_weight


LABEL_MAPPING_BINARY = {
    "benign": 0,
    "bot": 1,
    "ddos": 1,
    "dos goldeneye": 1,
    "dos hulk": 1,
    "dos slowhttptest": 1,
    "dos slowloris": 1,
    "ftp-patator": 1,
    "heartbleed": 1,
    "infiltration": 1,
    "portscan": 1,
    "ssh-patator": 1,
    "web attack - brute force": 1,
    "web attack - sql injection": 1,
    "web attack - xss": 1,
}

LABEL_MAPPING_MULTI = {
    "benign": 0,
    "bot": 1,
    "ddos": 2,
    "dos goldeneye": 3,
    "dos hulk": 4,
    "dos slowhttptest": 5,
    "dos slowloris": 6,
    "ftp-patator": 7,
    "heartbleed": 8,
    "infiltration": 9,
    "portscan": 10,
    "ssh-patator": 11,
    "web attack - brute force": 12,
    "web attack - sql injection": 13,
    "web attack - xss": 14,
}

UNSW_TO_BINARY = {
    "normal": 0,
    "analysis": 1,
    "backdoor": 1,
    "dos": 1,
    "exploits": 1,
    "fuzzers": 1,
    "generic": 1,
    "reconnaissance": 1,
    "shellcode": 1,
    "worms": 1,
}


def normalize_column_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("∞", "inf")
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: normalize_column_name(col) for col in df.columns}
    return df.rename(columns=renamed)


def normalize_label_text(label: str) -> str:
    s = str(label).strip().lower().replace("�", "-")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("web attack -", "web attack - ")
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _list_dataset_files(dataset_name: str, data_dir: Path) -> list[Path]:
    key = dataset_name.strip().lower()
    if key == "cicids2017":
        candidates = [data_dir / "CICDS2017", data_dir / "cicids2017", data_dir / "CICIDS2017"]
        for root in candidates:
            if root.exists():
                return sorted(root.glob("*.csv"))
        raise FileNotFoundError(f"CICIDS2017 directory not found under {data_dir}")

    if key == "unsw_nb15":
        candidates = [data_dir / "UNSW_NB15", data_dir / "unsw_nb15"]
        for root in candidates:
            if root.exists():
                files = sorted(root.glob("*.csv"))
                ignored = {"nusw-nb15_features.csv", "unsw-nb15_list_events.csv"}
                filtered = [f for f in files if f.name.lower() not in ignored]
                preferred = [
                    f
                    for f in filtered
                    if f.name.lower() in {"unsw_nb15_training-set.csv", "unsw_nb15_testing-set.csv"}
                ]
                return preferred if preferred else filtered
        raise FileNotFoundError(f"UNSW_NB15 directory not found under {data_dir}")

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def load_dataset(dataset_name: str, data_dir: Path | str, max_rows: int | None = None) -> pd.DataFrame:
    data_path = Path(data_dir)
    files = _list_dataset_files(dataset_name, data_path)
    if not files:
        raise FileNotFoundError(f"No CSV files found for dataset: {dataset_name}")

    frames: list[pd.DataFrame] = []
    rows_left = max_rows
    for file in files:
        nrows = None
        if rows_left is not None:
            if rows_left <= 0:
                break
            nrows = rows_left
        df = pd.read_csv(file, low_memory=False, nrows=nrows)
        frames.append(df)
        if rows_left is not None:
            rows_left -= len(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    return normalize_columns(merged)


def clean_nan_inf(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    return cleaned.fillna(0)


def handle_outliers(
    df: pd.DataFrame, columns: list[str], threshold: float = 1e6
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].clip(upper=threshold, lower=-threshold)
    return out


def remove_invalid_features(df: pd.DataFrame, corr_threshold: float = 0.95) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_cols:
        return out

    constant_cols = [col for col in numeric_cols if out[col].nunique(dropna=False) <= 1]
    if constant_cols:
        out = out.drop(columns=constant_cols)
        numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    if len(numeric_cols) < 2:
        return out

    sample_df = out[numeric_cols]
    if len(sample_df) > 50000:
        sample_df = sample_df.sample(n=50000, random_state=42)

    corr_matrix = sample_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
    if to_drop:
        out = out.drop(columns=to_drop)
    return out


def clean_data_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Clean without correlation-based feature removal.

    Use this for test/evaluation datasets so that features present in the
    common feature schema are not zeroed out by a correlation filter fitted
    on test rows.
    """
    out = df.drop_duplicates().copy()
    out = clean_nan_inf(out)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    return handle_outliers(out, numeric_cols, threshold=1e6)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates().copy()
    out = clean_nan_inf(out)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out = handle_outliers(out, numeric_cols, threshold=1e6)
    reserved = {"label", "attack_cat"}
    features_only = out[[c for c in out.columns if c not in reserved]]
    reduced = remove_invalid_features(features_only)
    for target in reserved:
        if target in out.columns:
            reduced[target] = out[target].values
    return reduced


def _ensure_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {normalize_column_name(k): v for k, v in mapping.items()}


def _encode_cicids_labels(series: pd.Series, label_mode: str) -> np.ndarray:
    normalized = series.astype(str).map(normalize_label_text)
    mapping = LABEL_MAPPING_BINARY if label_mode == "binary" else LABEL_MAPPING_MULTI
    y = normalized.map(mapping).fillna(1 if label_mode == "binary" else -1).astype(int).values
    return y


def _encode_unsw_labels(df: pd.DataFrame, label_mode: str) -> np.ndarray:
    if label_mode == "binary":
        if "attack_cat" in df.columns:
            normalized = df["attack_cat"].astype(str).map(normalize_label_text)
            y = normalized.map(UNSW_TO_BINARY)
            if y.isna().any() and "label" in df.columns:
                y = y.fillna(df["label"])
            return y.fillna(1).astype(int).values
        if "label" in df.columns:
            return pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int).values
    if "attack_cat" in df.columns:
        le = LabelEncoder()
        return le.fit_transform(df["attack_cat"].astype(str).values)
    if "label" in df.columns:
        return pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values
    raise ValueError("UNSW labels not found")


def encode_labels(
    df: pd.DataFrame, label_col: str, mapping: dict | None
) -> Tuple[np.ndarray, LabelEncoder]:
    if mapping is not None:
        normalized_map = {normalize_label_text(k): v for k, v in mapping.items()}
        mapped = (
            df[label_col]
            .astype(str)
            .map(normalize_label_text)
            .map(normalized_map)
            .fillna(0)
            .astype(int)
        )
        y = mapped.values
    else:
        le = LabelEncoder()
        y = le.fit_transform(df[label_col].astype(str).values)
        return y, le

    le = LabelEncoder()
    le.fit(y)
    return y, le


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must satisfy 0 < train < 1 and train+val < 1")

    test_ratio = 1 - train_ratio - val_ratio
    strat = y if stratify else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=strat
    )

    val_relative = val_ratio / (val_ratio + test_ratio)
    strat_tmp = y_tmp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_relative), random_state=random_state, stratify=strat_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def sample_dataframe_by_percentage(
    df: pd.DataFrame,
    y: np.ndarray,
    percentage: float = 100.0,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    pct = float(percentage)
    if pct <= 0 or pct > 100:
        raise ValueError("data_percentage must be in (0, 100].")
    if pct >= 100:
        return df.reset_index(drop=True), np.asarray(y)

    y_arr = np.asarray(y)
    if len(df) != len(y_arr):
        raise ValueError("df and y must have the same length.")
    if len(df) == 0:
        return df.copy(), y_arr

    train_size = pct / 100.0
    indices = np.arange(len(df))
    strat = y_arr if stratify and len(np.unique(y_arr)) > 1 else None
    try:
        picked, _ = train_test_split(
            indices,
            train_size=train_size,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError:
        # Fallback for tiny datasets where stratified split may be infeasible.
        picked, _ = train_test_split(
            indices,
            train_size=train_size,
            random_state=random_state,
            stratify=None,
        )
    picked = np.sort(picked)
    sampled_df = df.iloc[picked].reset_index(drop=True)
    sampled_y = y_arr[picked]
    return sampled_df, sampled_y


def fit_scaler(X_train: np.ndarray, scaler_type: str = "minmax"):
    key = scaler_type.lower()
    if key == "minmax":
        scaler = MinMaxScaler()
    elif key == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")
    scaler.fit(X_train)
    return scaler


def transform_features(X: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    return sklearn_compute_class_weight(class_weight="balanced", classes=classes, y=y_train)


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, strategy: str = "auto"):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("imblearn is required for SMOTE") from exc
    smote = SMOTE(sampling_strategy=strategy, random_state=42)
    return smote.fit_resample(X_train, y_train)


def apply_oversampling(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random oversampling: duplicate minority class samples to match majority class size."""
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError as exc:
        raise ImportError("imblearn is required for oversampling") from exc
    ros = RandomOverSampler(random_state=42)
    return ros.fit_resample(X_train, y_train)


def apply_undersampling(X_train: np.ndarray, y_train: np.ndarray):
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as exc:
        raise ImportError("imblearn is required for undersampling") from exc
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X_train, y_train)


def build_feature_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    numeric_df = pd.DataFrame(index=df.index)
    for feature in feature_names:
        if feature not in df.columns:
            numeric_df[feature] = 0.0
        else:
            numeric_df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    return numeric_df.values.astype(np.float32)


def _align_single_dataset(
    df: pd.DataFrame,
    renaming_map: dict[str, str],
    common_features: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    out = df.copy()
    mapping = _ensure_mapping(renaming_map)
    out = out.rename(columns={c: mapping[c] for c in out.columns if c in mapping})

    # Derived fields for UNSW to approximate NetFlow fields.
    if dataset_name == "unsw_nb15":
        if "flow_bytes_per_sec" not in out.columns and {"sload", "dload"}.issubset(out.columns):
            out["flow_bytes_per_sec"] = (
                pd.to_numeric(out["sload"], errors="coerce").fillna(0)
                + pd.to_numeric(out["dload"], errors="coerce").fillna(0)
            )
        if "flow_packets_per_sec" not in out.columns and "rate" in out.columns:
            out["flow_packets_per_sec"] = pd.to_numeric(out["rate"], errors="coerce").fillna(0)
        if "fwd_packets_per_sec" not in out.columns and {"spkts", "dur"}.issubset(out.columns):
            dur = pd.to_numeric(out["dur"], errors="coerce").replace(0, np.nan)
            out["fwd_packets_per_sec"] = pd.to_numeric(out["spkts"], errors="coerce").div(dur).fillna(0)
        if "bwd_packets_per_sec" not in out.columns and {"dpkts", "dur"}.issubset(out.columns):
            dur = pd.to_numeric(out["dur"], errors="coerce").replace(0, np.nan)
            out["bwd_packets_per_sec"] = pd.to_numeric(out["dpkts"], errors="coerce").div(dur).fillna(0)

    aligned = pd.DataFrame(index=out.index)
    for feature in common_features:
        if feature in out.columns:
            aligned[feature] = pd.to_numeric(out[feature], errors="coerce").fillna(0.0)
        else:
            aligned[feature] = 0.0
    return aligned


def align_features(
    cicids_df: pd.DataFrame, unsw_df: pd.DataFrame, config: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if hasattr(config, "common_features"):
        common_features = list(config.common_features)
        cicids_map = dict(config.cicids_renaming_map)
        unsw_map = dict(config.unsw_renaming_map)
    else:
        common_features = list(config["common_features"])
        cicids_map = dict(config["cicids_renaming_map"])
        unsw_map = dict(config["unsw_renaming_map"])

    cicids_aligned = _align_single_dataset(cicids_df, cicids_map, common_features, "cicids2017")
    unsw_aligned = _align_single_dataset(unsw_df, unsw_map, common_features, "unsw_nb15")
    return cicids_aligned, unsw_aligned


def prepare_labels(df: pd.DataFrame, dataset_name: str, label_mode: str = "binary") -> np.ndarray:
    dataset = dataset_name.strip().lower()
    if dataset == "cicids2017":
        if "label" not in df.columns:
            raise ValueError("CICIDS2017 requires `label` column")
        return _encode_cicids_labels(df["label"], label_mode)
    if dataset == "unsw_nb15":
        return _encode_unsw_labels(df, label_mode)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")
