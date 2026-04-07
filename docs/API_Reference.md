# NIDS API Reference

## `nids.config`

- `load_config(config_path: str | Path | None = None) -> ExperimentConfig`
- `save_config(config: ExperimentConfig, output_path: str | Path) -> None`
- `ExperimentConfig` — top-level frozen dataclass containing:
  - `data: DataConfig`
  - `model: ModelConfig`
  - `training: TrainingConfig`
  - `shap: ShapConfig`
  - `alignment: AlignmentConfig`
  - `runtime: RuntimeConfig`

## `nids.data.preprocessing`

- `load_dataset(dataset_name: str, data_dir: Path, max_rows: int | None = None) -> pd.DataFrame`
- `clean_data(df: pd.DataFrame) -> pd.DataFrame`
- `align_features(cicids_df: pd.DataFrame, unsw_df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame]`
- `encode_labels(df: pd.DataFrame, label_col: str, mapping: dict | None) -> Tuple[np.ndarray, LabelEncoder]`
- `split_data(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float, stratify: bool = True, random_state: int = 42) -> Tuple[np.ndarray, ...]`
- `fit_scaler(X_train: np.ndarray, scaler_type: str = "minmax")`
- `transform_features(X: np.ndarray, scaler) -> np.ndarray`

## `nids.data.dataset`

- `NIDSDataset(features: np.ndarray, labels: np.ndarray)`
- `create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=128, num_workers=0)`

## `nids.models.cnn_bilstm_se`

- `CNNBiLSTMSE(input_dim, num_classes, conv_channels, lstm_hidden_size=128, lstm_num_layers=2, dropout=0.3, bidirectional=True, use_attention=False, use_se=True)`
- `forward(inputs: torch.Tensor) -> torch.Tensor`
- `get_metadata() -> dict`

## `nids.models.classical`

- Random Forest and XGBoost wrappers with a unified `fit(X, y)` / `predict(X)` interface.

## `nids.models.registry`

- `get_model(name: str, **kwargs) -> nn.Module | object` — returns model instance by name.

## `nids.training.trainer`

- `Trainer(config: TrainingConfig, output_dir: Path)`
- `fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_classes: int = 2, class_weights: np.ndarray | None = None) -> TrainingSummary`
- `evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module | None = None, device: torch.device | None = None, num_classes: int = 2) -> EvaluationResult`

### Data classes

- `EvaluationResult(loss, metrics, predictions, labels, scores)`
- `TrainingSummary(best_metric, best_epoch, best_model_path, history, test_metrics)`

## `nids.training.optimizers`

- `build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer`
- `build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> _LRScheduler | None`

## `nids.training.callbacks`

- `EarlyStopping(patience: int, delta: float)`

## `nids.evaluation.metrics`

- `compute_nids_metrics(y_true: np.ndarray, y_pred: np.ndarray, benign_class: int = 0, y_score: np.ndarray | None = None) -> dict`

Returns a dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `accuracy` | float | Overall accuracy |
| `macro_f1` | float | Macro-averaged F1 |
| `avg_attack_recall` | float | Mean recall across attack classes |
| `attack_macro_precision` | float | Mean precision across attack classes |
| `benign_false_alarm_rate` | float | FAR for benign class |
| `attack_miss_rate` | float | 1 - avg_attack_recall |
| `confusion_matrix` | list | Confusion matrix as nested list |
| `pr_auc` | float | Precision-Recall AUC (requires `y_score`) |
| `roc_auc` | float | ROC AUC (requires `y_score`) |
| `best_f1` | float | Best F1 across all thresholds (requires `y_score`) |
| `best_f1_threshold` | float | Threshold for best F1 |
| `recall_at_far_1pct` | float | Best recall at FAR ≤ 1% (requires `y_score`) |
| `threshold_at_far_1pct` | float | Corresponding threshold |
| `recall_at_far_5pct` | float | Best recall at FAR ≤ 5% (requires `y_score`) |
| `threshold_at_far_5pct` | float | Corresponding threshold |

## `nids.evaluation.evaluator`

- `evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int = 2, criterion: nn.Module | None = None) -> dict`

## `nids.evaluation.latency`

- `measure_inference_latency(model: nn.Module, dataloader: DataLoader, device: torch.device, n_batches: int = 10) -> dict`

## `nids.features.shap_analysis`

- `SHAPAnalyzer(model: nn.Module, device: torch.device)`
- `sample_data(X_train: np.ndarray, y_train: np.ndarray, n_samples: int = 2000) -> np.ndarray`
- `compute_shap_values(X_samples: np.ndarray, background_size: int = 100)`
- `compute_feature_importance(shap_values) -> np.ndarray` (delegated to `importance.py`)
- `select_top_k(importance: np.ndarray, feature_names: List[str], k: int = 30) -> List[str]`

## `nids.features.importance`

- `compute_feature_importance(shap_values) -> np.ndarray` — handles 2D/3D SHAP value arrays and list inputs.

## `nids.features.feature_selector`

- `select_top_k_features(feature_names: list[str], importance: np.ndarray, k: int = 30) -> Tuple[list[str], np.ndarray, np.ndarray]`
- `select_by_cumulative_importance(feature_names: list[str], importance: np.ndarray, threshold: float = 0.9) -> ...`

## `nids.utils`

- `nids.utils.logging.get_logger(name: str) -> logging.Logger`
- `nids.utils.io.save_json(data: dict, path: Path) -> None`
- `nids.utils.reproducibility` — seed setting for torch, numpy, random
- `nids.utils.visualization` — paper-ready figure generation (confusion matrix, training curves, key metrics)
