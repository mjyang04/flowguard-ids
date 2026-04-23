# NIDS API Reference

Module-level reference for the `nids` package. CLI entry points live under
`scripts/` and are not re-documented here — see each file's top docstring
for invocation examples.

## `nids.config`

- `load_config(config_path: str | Path | None = None) -> ExperimentConfig`
- `save_config(config: ExperimentConfig, output_path: str | Path) -> None`
- `ExperimentConfig` — top-level dataclass:
  - `data: DataConfig`
  - `model: ModelConfig` — includes `transformer_layers / transformer_heads /
    transformer_dim_feedforward` for `cnn_bilstm_se_transformer`, and
    `d_token / n_blocks / attention_heads / ffn_factor` for `ft_transformer`
  - `training: TrainingConfig` — includes `selection_metric / loss_type /
    use_auc_loss / use_platt_calibration / label_smoothing`
  - `shap: ShapConfig`
  - `alignment: AlignmentConfig`
  - `runtime: RuntimeConfig`
  - `pipeline: PipelineConfig` — defaults for
    `scripts/run_two_stage_pipeline.py`, `scripts/run_experiments.py`, and
    `scripts/train.py`. Fields: `models, seeds, directions, do_train, force,
    one_click, imbalance_strategy, cross_dataset_enhancements,
    stage1_threshold, benign_class, summary_path`.

## `nids.data.preprocessing`

- `load_dataset(dataset_name: str, data_dir: Path, max_rows: int | None = None) -> pd.DataFrame`
- `clean_data(df: pd.DataFrame) -> pd.DataFrame`
- `clean_data_basic(df: pd.DataFrame) -> pd.DataFrame`
- `align_features(cicids_df, unsw_df, config) -> Tuple[pd.DataFrame, pd.DataFrame]`
- `prepare_labels(df, dataset_name, label_mode: str = "binary") -> np.ndarray`
- `get_num_classes(dataset: str, label_mode: str) -> int` — returns 2 for
  binary, 15 for `cicids2017` multiclass, 10 for `unsw_nb15` multiclass.
- `apply_smote(X, y) -> Tuple[np.ndarray, np.ndarray]`
- `apply_oversampling(X, y) -> Tuple[np.ndarray, np.ndarray]`
- `compute_class_weights(y) -> np.ndarray`
- `split_data(X, y, train_ratio, val_ratio, stratify=True, random_state=42)`
- `fit_scaler(X_train, scaler_type: str = "minmax")`

## `nids.data.dataset`

- `NIDSDataset(features, labels)`
- `create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=128, num_workers=0, weighted_sampler=False)`

## `nids.data.cross_dataset`

Pipeline helpers for building the aligned 55-dimensional NetFlow feature
space from raw CICIDS2017 + UNSW-NB15 CSVs (re-exports preprocessing
primitives plus `prepare_cross_dataset_splits`).

## `nids.models.cnn_bilstm`

- `CNNBiLSTM(input_dim, num_classes, conv_channels, ..., use_attention=False)` — baseline without SE.

## `nids.models.cnn_bilstm_se`

- `CNNBiLSTMSE(input_dim, num_classes, conv_channels, ..., use_se=True, se_reduction=16)`
- Also exports reusable blocks: `ConvBlock`, `SqueezeExcitation`, `AttentionPooling`.

## `nids.models.cnn_bilstm_se_transformer`

- `CNNBiLSTMSETransformer(input_dim, num_classes, conv_channels, ...,
  transformer_layers=2, transformer_heads=4, transformer_dim_feedforward=512)`
- Three-scale fusion (CNN + SE → BiLSTM → Transformer encoder). Reuses
  `ConvBlock / SqueezeExcitation / AttentionPooling` from `cnn_bilstm_se`.
- `effective_heads` attribute: actual number of attention heads used after
  divisibility fallback.
- Primary backbone for the thesis.

## `nids.models.cnn_bilstm_at`

- `CNNBiLSTMAT(input_dim, num_classes, dropout)` — Najar et al. lightweight
  CNN-BiLSTM-AT external baseline.

## `nids.models.cnn_bilstm_attention`

- `CNNBiLSTMAttention(...)` — legacy variant, not in the active training
  matrix but kept for API-compatibility.

## `nids.models.ft_transformer`

- `FTTransformer(input_dim, num_classes, d_token=64, n_blocks=3, attention_heads=4, ffn_factor=2.0, dropout=0.1)`
- FT-Transformer tabular baseline (Gorishniy et al. 2021).

## `nids.models.classical`

- `train_random_forest(X_train, y_train, random_state=42) -> RandomForestClassifier`
- `train_xgboost(X_train, y_train) -> XGBClassifier`
- `predict_binary_scores(model, X) -> np.ndarray | None`
- `evaluate_classical_model(model, X_test, y_test) -> dict`

## `nids.models.registry`

- `create_model(config: ModelConfig) -> nn.Module` — dispatches by
  `config.name`. Supported deep-model keys: `cnn_bilstm, cnn_bilstm_se,
  cnn_bilstm_se_transformer, cnn_bilstm_attention, cnn_bilstm_at,
  ft_transformer`. Classical models (`random_forest, xgboost`) are trained
  via `nids.models.classical` directly, not through this registry.

## `nids.training.trainer`

- `Trainer(config: TrainingConfig, output_dir: Path)`
- `fit(model, train_loader, val_loader, num_classes=2, class_weights=None, resume_checkpoint=None) -> TrainingSummary`
- `evaluate(model, data_loader, criterion=None, device=None, num_classes=2) -> EvaluationResult`
- `measure_latency(model, data_loader) -> dict`

### Data classes

- `EvaluationResult(loss, metrics, predictions, labels, scores)`
- `TrainingSummary(best_metric, best_epoch, best_model_path, history, test_metrics, fit_seconds)`

## `nids.training.optimizers`

- `build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer`
- `build_scheduler(optimizer, config: TrainingConfig) -> _LRScheduler | None`

## `nids.training.callbacks`

- `EarlyStopping(patience: int, delta: float)`

## `nids.training.selection`

- `compute_selection_metric(name, *, y_true, y_pred, y_score, num_classes) -> float`
- Dispatches the validation-time selection metric by `name`
  (`recall_at_far_1pct / recall_at_far_5pct / pr_auc / roc_auc / best_f1`
  for binary; `macro_f1 / weighted_f1 / accuracy / mcc` for multiclass).
- Torch-free so it can be unit-tested standalone.

## `nids.training.auc_loss`

- `pairwise_auc_loss(logits, labels, num_neg=5, margin=1.0) -> torch.Tensor`
  — hinge-based pairwise approximation of Wilcoxon-Mann-Whitney (Yan et al. 2003).

## `nids.training.focal_loss`

- `BinaryFocalLoss(alpha=0.25, gamma=2.0)` — Lin et al. 2017.

## `nids.evaluation.metrics`

- `compute_nids_metrics(y_true, y_pred, benign_class=0, y_score=None) -> dict`

Returns a dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `accuracy` | float | Overall accuracy |
| `macro_f1` | float | Macro-averaged F1 |
| `weighted_f1` | float | Weighted-averaged F1 |
| `mcc` | float | Matthews correlation coefficient |
| `avg_attack_recall` | float | Mean recall across attack classes |
| `attack_macro_precision` | float | Mean precision across attack classes |
| `benign_false_alarm_rate` | float | FAR for benign class |
| `attack_miss_rate` | float | 1 − avg_attack_recall |
| `confusion_matrix` | list | Confusion matrix as nested list |
| `per_class_f1 / per_class_recall / per_class_precision` | dict | Per-class breakdown (multiclass) |
| `pr_auc` | float | Precision-Recall AUC (binary, requires `y_score`) |
| `roc_auc` | float | ROC AUC (binary, requires `y_score`) |
| `ece` | float | Expected Calibration Error (binary, requires `y_score`) |
| `best_f1 / best_f1_threshold` | float | Best F1 across thresholds + threshold |
| `recall_at_far_1pct / threshold_at_far_1pct` | float | Best recall under FAR ≤ 1% + threshold |
| `recall_at_far_5pct / threshold_at_far_5pct` | float | Same under FAR ≤ 5% |

## `nids.evaluation.evaluator`

- `evaluate_model(model, data_loader, device, num_classes=2, criterion=None) -> dict`

## `nids.evaluation.latency`

- `measure_inference_latency(model, dataloader, device, n_batches=10) -> dict`

## `nids.evaluation.calibration`

- `PlattCalibrator()` — Platt scaling (Platt 1999) on raw logits.
  - `.fit(logits, labels) -> CalibrationResult`
  - `.transform(logits) -> np.ndarray` (probabilities)
  - `.save(path) -> None` / `PlattCalibrator.load(path) -> PlattCalibrator`
- `collect_logits(model, data_loader, device) -> tuple[np.ndarray, np.ndarray]`
  — torch helper (imported lazily so the core math stays torch-free).
- `CalibrationResult(A, B, val_ece_before, val_ece_after)` — JSON-safe.

## `nids.evaluation.two_stage`

Pure-numpy XI2S-IDS cascade core (Al-Najjar et al., MDPI FI 2025). Torch-
and sklearn-free so cascade logic is unit-testable in isolation.

- `binarize_scores(scores, threshold: float = 0.5) -> np.ndarray`
- `combine_stages(stage1_binary, stage2_multiclass, benign_class=0) -> np.ndarray`
  — gating: `final = benign if stage1_binary == 0 else stage2_multiclass`.
- `gating_stats(y_true, stage1_binary, benign_class=0) -> dict` — TP/FP/TN/FN
  plus `stage1_recall / precision / far / attack_rate` against binarized truth.
- `run_two_stage(stage1_scores, stage1_binary, stage2_multiclass, y_true,
  threshold=0.5, benign_class=0) -> TwoStageResult`
- `TwoStageResult(final_predictions, stage1_predictions, stage2_predictions,
  stage1_scores, threshold, benign_class, stats)` — frozen dataclass.

## `nids.features.shap_analysis`

- `SHAPAnalyzer(model, device)`
- `sample_data(X_train, y_train, n_samples=2000) -> np.ndarray`
- `compute_shap_values(X_samples, background_size=100)`
- `compute_feature_importance(shap_values) -> np.ndarray`
- `select_top_k(importance, feature_names, k=30) -> list[str]`

## `nids.features.importance`

- `compute_feature_importance(shap_values) -> np.ndarray` — handles 2D / 3D
  SHAP value arrays and list inputs.

## `nids.features.feature_selector`

- `select_top_k_features(feature_names, importance, k=30) -> Tuple[list[str], np.ndarray, np.ndarray]`
- `select_by_cumulative_importance(feature_names, importance, threshold=0.9) -> ...`

## `nids.utils`

- `nids.utils.logging.get_logger(name: str) -> logging.Logger`
- `nids.utils.io.save_json(data: dict, path: Path) -> None`
- `nids.utils.process.run_command(cmd, logger) -> None` — subprocess helper
  used by the batch orchestrators.
- `nids.utils.reproducibility.seed_everything(seed: int)` — sets torch /
  numpy / random / CUDA seeds.
- `nids.utils.run_layout` — artifact-path helpers shared by the CLI scripts:
  `canonical_model_root, experiment_group_name, candidate_model_roots,
  find_latest_report, find_latest_best_model,
  find_latest_checkpoint_run, resolve_model_root_for_write` and their
  `*_in_roots` variants.
- `nids.utils.visualization` — paper-ready figure generation (confusion
  matrix, training curves, key metrics, split distribution).
- `nids.utils.paper_export.export_paper_results(artifacts_root, output_dir) -> dict`
  — scans artifact trees for `report.json` and emits CSV tables + publication
  figures.
