# NIDS API Reference

## `nids.data.preprocessing`

- `load_dataset(dataset_name: str, data_dir: Path, max_rows: int | None = None) -> pd.DataFrame`
- `clean_data(df: pd.DataFrame) -> pd.DataFrame`
- `align_features(cicids_df: pd.DataFrame, unsw_df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame]`
- `encode_labels(df: pd.DataFrame, label_col: str, mapping: dict | None) -> Tuple[np.ndarray, LabelEncoder]`
- `split_data(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float, stratify: bool = True, random_state: int = 42) -> Tuple[np.ndarray, ...]`
- `fit_scaler(X_train: np.ndarray, scaler_type: str = "minmax")`
- `transform_features(X: np.ndarray, scaler) -> np.ndarray`

## `nids.models.cnn_bilstm_se`

- `CNNBiLSTMSE(input_dim, num_classes, conv_channels, lstm_hidden_size=128, lstm_num_layers=2, dropout=0.3, bidirectional=True, use_attention=False, use_se=True)`
- `forward(inputs: torch.Tensor) -> torch.Tensor`
- `get_metadata() -> dict`

## `nids.features.shap_analysis`

- `SHAPAnalyzer(model: nn.Module, device: torch.device)`
- `sample_data(X_train: np.ndarray, y_train: np.ndarray, n_samples: int = 2000) -> np.ndarray`
- `compute_shap_values(X_samples: np.ndarray, background_size: int = 100)`
- `compute_feature_importance(shap_values) -> np.ndarray`
- `select_top_k(importance: np.ndarray, feature_names: List[str], k: int = 30) -> List[str]`

## `nids.training.trainer`

- `Trainer(config: TrainingConfig, output_dir: Path)`
- `fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_classes: int = 2, class_weights: np.ndarray | None = None) -> TrainingSummary`
- `evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module | None = None, device: torch.device | None = None, num_classes: int = 2) -> EvaluationResult`
- `measure_latency(model: nn.Module, data_loader: DataLoader) -> dict`

## `nids.data.dataset`

- `NIDSDataset(features: np.ndarray, labels: np.ndarray)`
- `create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=128, num_workers=0)`
