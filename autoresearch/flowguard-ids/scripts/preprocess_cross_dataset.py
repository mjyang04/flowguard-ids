from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import load_config
from nids.data.preprocessing import (
    align_features,
    clean_data,
    clean_data_basic,
    fit_scaler,
    load_dataset,
    prepare_labels,
    sample_dataframe_by_percentage,
    transform_features,
)
from nids.utils.io import save_json, save_pickle
from nids.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset preprocessing for NIDS")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-dataset", default=None, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--test-dataset", default=None, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--label-mode", choices=["binary", "multi"], default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--data-percentage", type=float, default=None, help="Percent of each dataset to use (0, 100]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("preprocess_cross")

    train_dataset = args.train_dataset or cfg.data.train_dataset
    test_dataset = args.test_dataset or cfg.data.test_dataset
    label_mode = args.label_mode or cfg.data.label_mode
    data_dir = Path(args.data_dir or cfg.data.data_dir)
    output_dir = Path(args.output_dir or cfg.data.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rows = args.max_rows if args.max_rows is not None else cfg.data.max_rows
    data_percentage = args.data_percentage if args.data_percentage is not None else cfg.data.data_percentage

    logger.info("Loading CICIDS2017 and UNSW-NB15")
    # Apply full cleaning (incl. correlation removal) only to the training dataset.
    # The test dataset uses basic cleaning to avoid zeroing out valid features.
    if train_dataset == "cicids2017":
        cicids_df = clean_data(load_dataset("cicids2017", data_dir, max_rows=max_rows))
        unsw_df = clean_data_basic(load_dataset("unsw_nb15", data_dir, max_rows=max_rows))
    else:
        cicids_df = clean_data_basic(load_dataset("cicids2017", data_dir, max_rows=max_rows))
        unsw_df = clean_data(load_dataset("unsw_nb15", data_dir, max_rows=max_rows))
    y_cicids = prepare_labels(cicids_df, "cicids2017", label_mode=label_mode)
    y_unsw = prepare_labels(unsw_df, "unsw_nb15", label_mode=label_mode)
    cicids_full = len(cicids_df)
    unsw_full = len(unsw_df)
    cicids_df, y_cicids = sample_dataframe_by_percentage(
        df=cicids_df,
        y=y_cicids,
        percentage=data_percentage,
        random_state=cfg.data.random_state,
        stratify=cfg.data.stratify,
    )
    unsw_df, y_unsw = sample_dataframe_by_percentage(
        df=unsw_df,
        y=y_unsw,
        percentage=data_percentage,
        random_state=cfg.data.random_state,
        stratify=cfg.data.stratify,
    )
    logger.info(
        "Dataset sampling | percentage=%.2f%% cicids=%s/%s unsw=%s/%s",
        data_percentage,
        len(cicids_df),
        cicids_full,
        len(unsw_df),
        unsw_full,
    )

    X_cicids, X_unsw = align_features(cicids_df, unsw_df, cfg.alignment)
    if train_dataset == "cicids2017":
        X_train_raw = X_cicids.values.astype(np.float32)
        y_train_raw = y_cicids.astype(np.int64)
    else:
        X_train_raw = X_unsw.values.astype(np.float32)
        y_train_raw = y_unsw.astype(np.int64)

    if test_dataset == "cicids2017":
        X_test_raw = X_cicids.values.astype(np.float32)
        y_test = y_cicids.astype(np.int64)
    else:
        X_test_raw = X_unsw.values.astype(np.float32)
        y_test = y_unsw.astype(np.int64)

    val_ratio = cfg.data.val_ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=val_ratio,
        random_state=cfg.data.random_state,
        stratify=y_train_raw if cfg.data.stratify else None,
    )

    scaler = fit_scaler(X_train, scaler_type=cfg.data.scaler_type)
    X_train = transform_features(X_train, scaler)
    X_val = transform_features(X_val, scaler)
    X_test = transform_features(X_test_raw, scaler)

    tag = f"cross_{train_dataset}_to_{test_dataset}"
    np.savez_compressed(
        output_dir / f"{tag}.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=np.array(cfg.alignment.common_features),
    )
    save_pickle(scaler, output_dir / f"{tag}_scaler.pkl")
    save_json(
        {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "label_mode": label_mode,
            "data_percentage": float(data_percentage),
            "source_rows": {
                "cicids2017": int(cicids_full),
                "unsw_nb15": int(unsw_full),
            },
            "used_rows": {
                "cicids2017": int(len(cicids_df)),
                "unsw_nb15": int(len(unsw_df)),
            },
            "n_features": len(cfg.alignment.common_features),
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "test_size": int(len(y_test)),
            "feature_names": cfg.alignment.common_features,
        },
        output_dir / f"{tag}.json",
    )
    logger.info("Saved cross-dataset artifact: %s", output_dir / f"{tag}.npz")


if __name__ == "__main__":
    main()
