from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import load_config
from nids.data.preprocessing import (
    clean_data,
    fit_scaler,
    load_dataset,
    prepare_labels,
    sample_dataframe_by_percentage,
    split_data,
    transform_features,
)
from nids.utils.io import save_json, save_pickle
from nids.utils.logging import get_logger


def _encode_categorical_inplace(df, ignore_columns: set[str]) -> None:
    for col in df.columns:
        if col in ignore_columns:
            continue
        if df[col].dtype == "object":
            df[col], _ = df[col].factorize(sort=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess single dataset for NIDS training")
    parser.add_argument("--dataset", required=True, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-mode", choices=["binary", "multi"], default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--data-percentage", type=float, default=None, help="Percent of dataset to use (0, 100]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("preprocess")

    dataset_name = args.dataset
    data_dir = Path(args.data_dir or cfg.data.data_dir)
    output_root = Path(args.output_dir or cfg.data.processed_dir)
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    label_mode = args.label_mode or cfg.data.label_mode
    max_rows = args.max_rows if args.max_rows is not None else cfg.data.max_rows
    data_percentage = args.data_percentage if args.data_percentage is not None else cfg.data.data_percentage

    logger.info("Loading dataset=%s from %s", dataset_name, data_dir)
    df = load_dataset(dataset_name, data_dir, max_rows=max_rows)
    df = clean_data(df)
    y = prepare_labels(df, dataset_name=dataset_name, label_mode=label_mode)
    full_rows = len(df)
    df, y = sample_dataframe_by_percentage(
        df=df,
        y=y,
        percentage=data_percentage,
        random_state=cfg.data.random_state,
        stratify=cfg.data.stratify,
    )
    logger.info(
        "Dataset sampling | percentage=%.2f%% rows=%s/%s",
        data_percentage,
        len(df),
        full_rows,
    )

    ignore_cols = {"label", "attack_cat"}
    _encode_categorical_inplace(df, ignore_cols)
    feature_cols = [
        c
        for c in df.columns
        if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X=X,
        y=y,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        stratify=cfg.data.stratify,
        random_state=cfg.data.random_state,
    )

    scaler = fit_scaler(X_train, scaler_type=cfg.data.scaler_type)
    X_train = transform_features(X_train, scaler)
    X_val = transform_features(X_val, scaler)
    X_test = transform_features(X_test, scaler)

    np.savez_compressed(
        output_dir / "data.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=np.array(feature_cols),
    )
    save_pickle(scaler, output_dir / "scaler.pkl")
    save_json(
        {
            "dataset": dataset_name,
            "label_mode": label_mode,
            "data_percentage": float(data_percentage),
            "source_rows": int(full_rows),
            "used_rows": int(len(df)),
            "n_features": len(feature_cols),
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "test_size": int(len(y_test)),
            "feature_names": feature_cols,
        },
        output_dir / "metadata.json",
    )
    logger.info("Saved preprocessed data to %s", output_dir)


if __name__ == "__main__":
    main()
