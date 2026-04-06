from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import (
    align_features,
    clean_data,
    clean_data_basic,
    fit_scaler,
    load_dataset,
    prepare_labels,
    transform_features,
)


def prepare_cross_dataset(
    cicids_path: str | Path,
    unsw_path: str | Path,
    feature_config: Any,
    train_dataset: str = "cicids2017",
    test_dataset: str = "unsw_nb15",
    max_rows: int | None = None,
    label_mode: str = "binary",
):
    train_key = train_dataset.lower()
    test_key = test_dataset.lower()

    # Apply full cleaning (incl. correlation removal) only to the training dataset.
    # The test dataset uses basic cleaning only to avoid zeroing out valid features
    # that happen to be correlated within the test set.
    if train_key == "cicids2017":
        cicids_df = clean_data(load_dataset("cicids2017", cicids_path, max_rows=max_rows))
        unsw_df = clean_data_basic(load_dataset("unsw_nb15", unsw_path, max_rows=max_rows))
    else:
        cicids_df = clean_data_basic(load_dataset("cicids2017", cicids_path, max_rows=max_rows))
        unsw_df = clean_data(load_dataset("unsw_nb15", unsw_path, max_rows=max_rows))

    y_cicids = prepare_labels(cicids_df, "cicids2017", label_mode=label_mode)
    y_unsw = prepare_labels(unsw_df, "unsw_nb15", label_mode=label_mode)

    cicids_features, unsw_features = align_features(cicids_df, unsw_df, feature_config)

    if train_key == "cicids2017":
        X_train_raw = cicids_features.values.astype(np.float32)
        y_train = y_cicids
    else:
        X_train_raw = unsw_features.values.astype(np.float32)
        y_train = y_unsw

    if test_key == "cicids2017":
        X_test_raw = cicids_features.values.astype(np.float32)
        y_test = y_cicids
    else:
        X_test_raw = unsw_features.values.astype(np.float32)
        y_test = y_unsw

    scaler = fit_scaler(X_train_raw, scaler_type="minmax")
    X_train = transform_features(X_train_raw, scaler)
    X_test = transform_features(X_test_raw, scaler)

    return {
        "X_train": X_train,
        "y_train": y_train.astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.astype(np.int64),
        "feature_names": np.array(list(feature_config.common_features)),
        "scaler": scaler,
    }