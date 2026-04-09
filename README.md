# FlowGuard IDS

Lightweight IDS research project for graduation design:
**CNN-BiLSTM-SE + SHAP explainability + cross-dataset generalization (CICIDS2017 / UNSW-NB15)**.

## Highlights

- Unified preprocessing pipeline for CICIDS2017 and UNSW-NB15 (55-dim shared feature space)
- 6 supported models:
  - `cnn_bilstm` — baseline deep model
  - `cnn_bilstm_se` — main model with Squeeze-Excitation attention
  - `cnn_bilstm_se_topk` — `cnn_bilstm_se` architecture trained on SHAP-reduced Top-K features
  - `cnn_bilstm_at` — Najar et al. lightweight CNN-BiLSTM-AT baseline
  - `random_forest` — lightweight classical model
  - `xgboost` — lightweight classical model
- `cnn_bilstm_attention` is kept in code but excluded from current training commands
- Imbalance-aware evaluation metrics designed for IDS: `recall_at_far_1pct`, `recall_at_far_5pct`, `best_f1`, `pr_auc`, `roc_auc` (see [Evaluation Metrics](#evaluation-metrics))
- One-click training and one-click experiments
- Skip trained models by default in one-click mode, with `--force` for full retrain
- Every training run is traceable with a unique folder name: `timestamp + model + strategy (+ tag)`
- Training visualization with `tqdm` (epoch + batch), live loss/lr/metric display
- Auto-generate paper-ready figures and structured run artifacts in each run folder
- Supports resume training from `checkpoint_last.pt` for interrupted deep-model runs
- Config-driven scheduler and early stopping switches
- Docker support for GPU training environments (CUDA 12.4)

## Repository Structure

```text
flowguard-ids/
|-- configs/                              # Experiment/runtime configuration files
|   |-- default.yaml                      # Default config (recommended start)
|   |-- experiment_cnn_bilstm_se.yaml     # Main paper-style experiment config
|   |-- cicids2017.yaml                   # Same-dataset config for CICIDS2017
|   `-- unsw_nb15.yaml                    # Same-dataset config for UNSW-NB15
|
|-- data/
|   |-- raw/                              # Raw datasets root (default data_dir)
|   |   |-- cicids2017/*.csv
|   |   `-- unsw_nb15/*.csv
|   |-- processed/                        # Preprocessed artifacts (*.npz, scaler, metadata)
|   `-- shap_samples/                     # Cached SHAP sample files
|
|-- docs/
|   |-- API_Reference.md                  # Core module interfaces
|   `-- NIDS_Technical_Design.md          # Technical design document (Chinese)
|
|-- nids/                                 # Main Python package
|   |-- config.py                         # Dataclass config definitions + loader
|   |-- data/                             # Loading/cleaning/alignment/dataset builders
|   |   |-- preprocessing.py              # Single-dataset preprocessing
|   |   |-- cross_dataset.py              # Cross-dataset feature alignment
|   |   |-- dataset.py                    # NIDSDataset + DataLoader factory
|   |   `-- augmentation.py               # SMOTE / resampling utilities
|   |-- models/                           # Deep models, classical models, model registry
|   |   |-- cnn_bilstm.py                # Baseline CNN-BiLSTM
|   |   |-- cnn_bilstm_se.py             # Main model with SE attention
|   |   |-- cnn_bilstm_at.py            # Najar et al. CNN-BiLSTM-AT lightweight model
|   |   |-- cnn_bilstm_attention.py      # Variant with attention (excluded from training)
|   |   |-- classical.py                  # Random Forest, XGBoost wrappers
|   |   |-- base.py                       # Abstract base model interface
|   |   `-- registry.py                   # Model name -> class mapping
|   |-- training/                         # Trainer loop, optimizer, scheduler, callbacks
|   |   |-- trainer.py                    # Main training loop + evaluation
|   |   |-- optimizers.py                 # Optimizer & scheduler factory
|   |   `-- callbacks.py                  # Early stopping
|   |-- evaluation/                       # Metrics and latency utilities
|   |   |-- metrics.py                    # NIDS-specific metrics (see Evaluation Metrics)
|   |   |-- evaluator.py                  # High-level model evaluation
|   |   `-- latency.py                    # Inference latency measurement
|   |-- features/                         # SHAP and feature selection pipeline
|   |   |-- shap_analysis.py              # SHAPAnalyzer (GradientExplainer / DeepExplainer)
|   |   |-- importance.py                 # Feature importance aggregation
|   |   `-- feature_selector.py           # Top-K and cumulative threshold selection
|   `-- utils/                            # Logging, IO, reproducibility, plotting
|       |-- logging.py                    # Structured logger
|       |-- io.py                         # JSON/artifact I/O helpers
|       |-- visualization.py              # Paper-ready figures
|       |-- reproducibility.py            # Seed setting
|       `-- process.py                    # Subprocess utilities
|
|-- scripts/                              # CLI entrypoints used by users
|   |-- preprocess.py                     # Single-dataset preprocessing
|   |-- preprocess_cross_dataset.py       # Cross-dataset preprocessing
|   |-- train.py                          # Train one model / all models / one-click
|   |-- train_lightweight.py              # Retrain lightweight final model from Top-K features
|   |-- run_experiments.py                # Batch experiment runner (same + cross)
|   |-- evaluate.py                       # Evaluate saved model on test artifact
|   |-- shap_analysis.py                  # SHAP explainability workflow
|   |-- feature_selection.py              # Top-K/cumulative feature selection
|   `-- export_model.py                   # Export to TorchScript / ONNX
|
|-- tests/                                # Unit tests (config, data, model, training, metrics)
|-- Dockerfile                            # CUDA training environment (PyTorch 2.5.1 + CUDA 12.4)
|-- requirements.txt                      # Python dependencies
|-- setup.py                              # Package metadata
`-- README.md
```

Recommended reading order for new contributors:

1. `README.md` (quick start and workflow)
2. `configs/default.yaml` (all key knobs)
3. `scripts/train.py` and `scripts/run_experiments.py` (entrypoint behavior)
4. `nids/data/preprocessing.py` and `nids/models/` (core pipeline + model logic)
5. `nids/training/trainer.py` (training details: scheduler, early stopping, progress)
6. `nids/evaluation/metrics.py` (NIDS-specific evaluation metrics)
7. `docs/API_Reference.md` (module interfaces)
8. `docs/NIDS_Technical_Design.md` (system design and architecture, Chinese)

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended

Tested and recommended versions (RTX 3060 Laptop GPU):

| Package | Minimum | Recommended |
|---------|---------|-------------|
| Python | 3.10+ | 3.10.20 |
| torch | 2.5+ | 2.5.1+cu124 |
| numpy | 1.26+ | 2.2.6 |
| pandas | 2.2+ | 2.3.3 |
| scikit-learn | 1.4+ | 1.7.2 |
| xgboost | 2.1+ | 3.2.0 |
| shap | 0.46+ | 0.49.1 |
| imbalanced-learn | 0.12+ | 0.14.1 |
| matplotlib | 3.8+ | 3.10.8 |
| tqdm | 4.66+ | 4.67.3 |
| pytest | 8.0+ | 9.0.2 |

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Placement

Default config expects:

```text
data/raw/
|-- cicids2017/*.csv
`-- unsw_nb15/*.csv
```

The loader also accepts alternative folder names:
- `CICDS2017/` for CICIDS
- `UNSW_NB15/` for UNSW

If your path is different, edit `data.data_dir` in config files.

## Quick Start

### 1) One-click training (single train/test pair, all models)

This command will:
- auto preprocess if needed
- train all supported models (or selected subset via `--models`)
- skip models already trained
- for `cnn_bilstm_se_topk`, auto-run SHAP + Top-K feature selection if reduced data is missing

```bash
python scripts/train.py --config configs/default.yaml --cross-dataset --train-dataset cicids2017 --test-dataset unsw_nb15 --one-click
```

Force full retraining:

```bash
python scripts/train.py --config configs/default.yaml --cross-dataset --train-dataset cicids2017 --test-dataset unsw_nb15 --one-click --force
```

### 2) One-click experiments (2 dataset settings x all models)

Runs:
- `cicids2017 -> cicids2017` (same-dataset baseline)
- `cicids2017 -> unsw_nb15` (cross-dataset generalization)

Each setting trains all supported models and skips existing results by default:

```bash
python scripts/run_experiments.py --config configs/default.yaml --one-click
```

`run_experiments.py` now reuses the same canonical training layout as `scripts/train.py`:

- model artifacts: `artifacts/<train_dataset>_to_<test_dataset>/<model_name>/runs/...`
- batch summary: `artifacts/experiments/experiment_status.json`

In non-`--force` one-click mode:
- finished models are skipped
- interrupted deep-model runs are resumed from `checkpoint_last.pt` automatically

Force full retraining for all experiment groups:

```bash
python scripts/run_experiments.py --config configs/default.yaml --one-click --force
```

### 2.1) RTX 3060 laptop preset (keep default training hyperparameters)

This preset keeps the baseline config from `configs/default.yaml` intact, including `batch_size: 512`, but reduces total work by:

- running same-dataset (`cicids2017 -> cicids2017`) and cross-dataset (`cicids2017 -> unsw_nb15`) experiments
- selecting `cnn_bilstm_se`, `random_forest`, and `xgboost`
- keeping one-click conveniences such as auto preprocess, skip existing results, and resume

```bash
python scripts/run_experiments.py --config configs/default.yaml --profile laptop_3060 --one-click
```

### 3) Final lightweight model (your deliverable model)

Pipeline:

1. Train base model (e.g., CNN-BiLSTM-SE with class weights)
2. Run SHAP analysis once on the shared reference model (`cicids2017 -> cicids2017`, `cnn_bilstm_se`)
3. Save fixed Top-K feature index lists
4. Retrain lightweight model on reduced features

Simplified command (recommended):

```bash
python scripts/train_lightweight.py --config configs/default.yaml --model random_forest --auto-preprocess --auto-feature-selection
```

This command will:
- ensure base cross-dataset artifact exists
- generate `reduced_data.npz` from SHAP outputs if missing
- train the lightweight model as final output
- create a new traceable run folder automatically

## Common Commands

Train a specific model:

```bash
python scripts/train.py --config configs/default.yaml --cross-dataset --train-dataset cicids2017 --test-dataset unsw_nb15 --model cnn_bilstm_se --imbalance-strategy class_weights --run-tag thesis_main --auto-preprocess
```

Train all models manually:

```bash
python scripts/train.py --config configs/default.yaml --cross-dataset --train-dataset cicids2017 --test-dataset unsw_nb15 --models all --auto-preprocess
```

Resume interrupted deep-model training (continue from latest checkpoint of that model):

```bash
python scripts/train.py --config configs/default.yaml --cross-dataset --train-dataset cicids2017 --test-dataset unsw_nb15 --model cnn_bilstm_se --resume
```

Train with SMOTE strategy (deep/classical supported in training stage):

```bash
python scripts/train.py --config configs/default.yaml --cross-dataset --train-dataset cicids2017 --test-dataset unsw_nb15 --model cnn_bilstm --imbalance-strategy smote --run-tag smote_try --auto-preprocess
```

Preprocess only:

```bash
python scripts/preprocess.py --dataset cicids2017 --config configs/default.yaml
python scripts/preprocess_cross_dataset.py --config configs/default.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --model artifacts/cicids2017_to_unsw_nb15/cnn_bilstm_se/runs/<run_name>/best_model.pt --config configs/default.yaml
```

SHAP analysis:

```bash
python scripts/shap_analysis.py --model artifacts/cicids2017_to_unsw_nb15/cnn_bilstm_se/runs/<run_name>/best_model.pt --config configs/default.yaml
```

Top-K feature selection:

```bash
python scripts/feature_selection.py --selected-idx artifacts/shap/shared/cicids2017_to_cicids2017/cnn_bilstm_se/top30_idx.npy --feature-names artifacts/shap/shared/cicids2017_to_cicids2017/cnn_bilstm_se/feature_names.npy --data-file data/processed/cross_cicids2017_to_unsw_nb15.npz --output-dir artifacts/feature_selection/cicids2017_to_unsw_nb15/cnn_bilstm_se_topk_top30
```

Train `cnn_bilstm_se_topk` (requires reduced features; add `--auto-feature-selection` to auto build):

```bash
python scripts/train.py --config configs/default.yaml --cross-dataset --train-dataset cicids2017 --test-dataset unsw_nb15 --model cnn_bilstm_se_topk --auto-preprocess --auto-feature-selection
```

Shared SHAP reference outputs are generated only once and reused later. The default reference is:
- train dataset: `cicids2017`
- test dataset: `cicids2017`
- model: `cnn_bilstm_se`

The shared SHAP directory will contain:
- `shap_values.npy`
- `feature_importance.npy`
- `sorted_feature_idx.npy`
- `feature_ranking.json`
- `top20_idx.npy`
- `top30_idx.npy`
- `top50_idx.npy`
- `topk_indices.json`

Complex (fully manual) lightweight/classical retraining command:

```bash
python scripts/train.py --config configs/default.yaml --data-file artifacts/feature_selection/cicids2017_to_unsw_nb15/cnn_bilstm_se_topk/reduced_data.npz --model random_forest --output-dir artifacts/lightweight/cicids2017_to_unsw_nb15/random_forest
```

## Output Layout

Training outputs are saved under:

```text
artifacts/<train_dataset>_to_<test_dataset>/<model_name>/
|-- latest_run.txt
|-- latest_report.txt
|-- latest_run.json
|-- runs_index.json
`-- runs/
    `-- <timestamp>_<ModelName>_<Strategy>[_<tag>]/
        |-- report.json
        |-- run_manifest.json
        |-- checkpoint_last.pt              # resume checkpoint
        |-- resolved_config.yaml
        |-- best_model.pt / best_model.pkl
        |-- training_summary.json            # deep models
        |-- training_history.csv
        |-- test_predictions.npz             # y_true / y_pred
        |-- key_metrics.json
        |-- data_profile.json
        |-- training_curves.png              # deep models
        `-- figures/
            |-- confusion_matrix.png
            |-- nids_key_metrics.png
            `-- split_distribution.png
```

The `<Strategy>` part is automatically inferred from your imbalance setting, e.g.:
- `ClassWeights`
- `SMOTE`
- `NoRebalance`

This design guarantees reproducibility and makes every run directly traceable in your thesis output.

When using `scripts/run_experiments.py`, the same layout is used for all model artifacts. The runner only writes experiment summaries to `artifacts/experiments/`.

## Evaluation Metrics

The evaluation system (`nids/evaluation/metrics.py`) computes IDS-specific metrics that account for class imbalance — in network intrusion detection, attack traffic is rare and missing an attack is far more costly than a false alarm.

### Core Metrics

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy |
| `macro_f1` | Macro-averaged F1 score across all classes |
| `avg_attack_recall` | Mean recall across attack classes (excludes benign) |
| `attack_macro_precision` | Mean precision across attack classes |
| `benign_false_alarm_rate` | Rate at which benign traffic is misclassified as attack (FAR) |
| `attack_miss_rate` | `1 - avg_attack_recall` — fraction of attacks missed |

### Imbalance-Aware Threshold Metrics

These metrics are computed via a vectorized threshold sweep over prediction scores (O(n log n)):

| Metric | Description |
|--------|-------------|
| `recall_at_far_1pct` | Best recall achievable while keeping FAR ≤ 1% |
| `recall_at_far_5pct` | Best recall achievable while keeping FAR ≤ 5% |
| `best_f1` | Optimal F1 score across all thresholds (tie-break: max recall, then min FAR) |
| `pr_auc` | Area under the Precision-Recall curve |
| `roc_auc` | Area under the ROC curve |

### Model Selection

The `training.selection_metric` config knob controls which metric is used to select the best checkpoint during training. The default is `recall_at_far_1pct` — this prioritizes models that detect attacks reliably while maintaining a low false alarm rate.

Available choices: `recall_at_far_1pct`, `recall_at_far_5pct`, `best_f1`, `pr_auc`, `avg_attack_recall`, or any key returned by `compute_nids_metrics()`.

## Key Config Knobs

Edit `configs/*.yaml`:

```yaml
data:
  data_percentage: 100   # use full dataset; e.g. 10 means use 10% for quick experiments
  batch_size: 256

training:
  num_epochs: 30
  learning_rate: 0.001
  optimizer: adamw
  use_scheduler: true
  use_early_stopping: true
  selection_metric: recall_at_far_1pct  # metric for best checkpoint selection
  use_tqdm: true
  show_eval_tqdm: false
```

Important:
- If preprocessed artifacts already exist (for example from 100% data), changing `data.data_percentage` alone will not trigger automatic re-preprocessing.
- To make a new percentage effective, re-run preprocessing explicitly (or remove the old processed artifact first).

## Docker

A `Dockerfile` is provided for reproducible GPU training environments:

```bash
# Build the image
docker build -t flowguard-ids .

# Run training (GPU required)
docker run --rm --gpus all -v $(pwd):/workspace flowguard-ids \
  python scripts/train.py --config configs/default.yaml --cross-dataset \
  --train-dataset cicids2017 --test-dataset unsw_nb15 --one-click

# Interactive shell
docker run -it --gpus all -v $(pwd):/workspace flowguard-ids bash
```

Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`.

## Testing

```bash
pytest -q
```

## Notes

- One-click mode is designed for incremental reruns.
- Use `--force` when you intentionally want clean full retraining.
