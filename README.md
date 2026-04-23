# FlowGuard IDS

Lightweight network intrusion detection research project for graduation
design:
**CNN-BiLSTM-SE-Transformer three-scale backbone + XI2S two-stage cascade (binary gate → multiclass refinement) + SHAP-driven Top-K feature selection** on CICIDS2017 / UNSW-NB15. Cross-dataset transfer is kept as a secondary comparison track.

## Highlights

- Unified 55-dimensional NetFlow feature space shared across CICIDS2017 and UNSW-NB15
- **Two-stage cascade**: binary Stage-1 gate + multiclass Stage-2 refinement, automated via `scripts/run_two_stage_pipeline.py`; cascade core is pure-numpy and unit-tested (`nids.evaluation.two_stage`, 18 tests)
- **Three-scale backbone** `cnn_bilstm_se_transformer`: CNN + SE → BiLSTM → 2-layer Transformer encoder (~1.2M params, fits on a 6GB laptop GPU)
- **SHAP-driven Top-K** feature selection workflow that retrains any backbone into an explicit lightweight variant
- **Multiclass artifact isolation**: multiclass runs live under `artifacts/<dir>_multiclass/<model>/`, so binary and multiclass runs for the same direction never overwrite each other
- **Configuration-driven CLI**: every pipeline knob (models, seeds, directions, thresholds, training flags) lives under the `pipeline:` section of `configs/default.yaml`; the headline command takes no other flags
- Per-seed artifact isolation (`artifacts/seed<S>/...`) for mean ± std reporting over seeds `42/43/44`
- Platt scaling + ECE for calibrated binary probabilities; macro-F1 / weighted-F1 / per-class F1 / MCC for multiclass
- Docker image pinned to PyTorch 2.5.1 + CUDA 12.4 for reproducible GPU training on RTX 3060 Laptop (6GB VRAM)

### Supported models

| Key | Role |
|---|---|
| `cnn_bilstm_se_transformer` | **Primary backbone** — three-scale CNN + SE + BiLSTM + Transformer |
| `cnn_bilstm_se` | Ablation baseline (drop Transformer to isolate its contribution) |
| `cnn_bilstm` | Pure baseline without SE |
| `cnn_bilstm_se_topk` | `cnn_bilstm_se` trained on SHAP-reduced Top-K features |
| `cnn_bilstm_at` | External baseline (Najar et al. 2025) |
| `ft_transformer` | FT-Transformer tabular baseline (Gorishniy et al. 2021) |
| `random_forest` | Lightweight classical baseline |
| `xgboost` | Lightweight classical baseline |

`cnn_bilstm_attention` is kept in code for backward compatibility but excluded from the active training matrix.

## Repository Structure

```text
flowguard-ids/
|-- configs/
|   `-- default.yaml                      # Single config file; pipeline: section drives every CLI default
|
|-- data/                                 # gitignored
|   |-- raw/{cicids2017,unsw_nb15}/*.csv
|   |-- processed/                        # *.npz (binary + _multiclass variants)
|   `-- shap_samples/
|
|-- docs/
|   |-- API_Reference.md                  # Module-level API surface
|   |-- NIDS_Technical_Design.md          # Technical design document (v1 details + v2 pointers)
|   `-- thesis_v2.md                      # Thesis main draft (gitignored)
|
|-- nids/                                 # Main Python package
|   |-- config.py                         # Dataclass config definitions + PipelineConfig + load/save
|   |-- data/
|   |   |-- preprocessing.py              # Single-dataset preprocessing + label mapping
|   |   |-- cross_dataset.py              # Cross-dataset feature alignment
|   |   |-- dataset.py                    # NIDSDataset + DataLoader factory
|   |   `-- augmentation.py               # SMOTE / resampling utilities
|   |-- models/
|   |   |-- base.py                       # Abstract base model interface
|   |   |-- cnn_bilstm.py                 # Baseline CNN-BiLSTM
|   |   |-- cnn_bilstm_se.py              # CNN-BiLSTM + SE attention
|   |   |-- cnn_bilstm_se_transformer.py  # Primary three-scale backbone (new in v2)
|   |   |-- cnn_bilstm_at.py              # Najar et al. lightweight baseline
|   |   |-- cnn_bilstm_attention.py       # Legacy, excluded from training
|   |   |-- ft_transformer.py             # FT-Transformer tabular baseline
|   |   |-- classical.py                  # Random Forest / XGBoost wrappers
|   |   `-- registry.py                   # create_model(name) dispatcher
|   |-- training/
|   |   |-- trainer.py                    # Main training loop + evaluation + latency
|   |   |-- optimizers.py                 # Optimizer + scheduler factory
|   |   |-- callbacks.py                  # Early stopping
|   |   |-- selection.py                  # Torch-free selection-metric dispatcher
|   |   |-- auc_loss.py                   # Pairwise AUC ranking loss (Yan et al. 2003)
|   |   `-- focal_loss.py                 # Focal loss (Lin et al. 2017)
|   |-- evaluation/
|   |   |-- metrics.py                    # compute_nids_metrics: binary + multiclass
|   |   |-- evaluator.py                  # High-level model evaluation
|   |   |-- latency.py                    # Inference latency measurement
|   |   |-- calibration.py                # Platt scaling (lazy torch import)
|   |   `-- two_stage.py                  # Pure-numpy XI2S cascade core (new in v2)
|   |-- features/
|   |   |-- shap_analysis.py              # SHAPAnalyzer (GradientExplainer / DeepExplainer)
|   |   |-- importance.py                 # Feature importance aggregation
|   |   `-- feature_selector.py           # Top-K / cumulative threshold selection
|   `-- utils/
|       |-- logging.py                    # Structured logger
|       |-- io.py                         # JSON / artifact I/O helpers
|       |-- process.py                    # Subprocess helper for orchestrators
|       |-- reproducibility.py            # Seed setting
|       |-- run_layout.py                 # Artifact-path helpers for skip-existing / resume
|       |-- paper_export.py               # Scan artifact tree -> CSV tables + figures
|       `-- visualization.py              # Paper-ready figures
|
|-- scripts/                              # CLI entry points
|   |-- preprocess.py                     # Single-dataset preprocessing
|   |-- preprocess_cross_dataset.py       # Aligned NPZ builder
|   |-- train.py                          # Train one / some / all models
|   |-- train_lightweight.py              # Retrain lightweight model from Top-K features
|   |-- run_experiments.py                # Batch experiment matrix (reads pipeline:)
|   |-- run_two_stage_pipeline.py         # Headline end-to-end cascade pipeline (new in v2)
|   |-- eval_two_stage.py                 # Cascade evaluator for a single (direction, model) pair (new in v2)
|   |-- evaluate.py                       # Evaluate a saved model on a test artifact
|   |-- shap_analysis.py                  # SHAP explainability workflow
|   |-- feature_selection.py              # Top-K / cumulative feature selection
|   |-- export_model.py                   # TorchScript / ONNX export
|   |-- export_paper_results.py           # Aggregate artifact tree -> CSVs + figures for thesis
|   |-- bench_model_size.py               # Parameter / memory budget audit
|   `-- md_to_word.py                     # thesis_v2.md -> .docx converter
|
|-- tests/                                # Unit tests (66 passing + 7 torch-skipped locally)
|-- Dockerfile                            # PyTorch 2.5.1 + CUDA 12.4
|-- requirements.txt
|-- setup.py
`-- README.md
```

Recommended reading order for new contributors:

1. `README.md` — quick start + workflow
2. `configs/default.yaml` — every knob (especially the `pipeline:` section)
3. `scripts/run_two_stage_pipeline.py` + `scripts/run_experiments.py` + `scripts/train.py` — entrypoint behavior
4. `nids/models/cnn_bilstm_se_transformer.py` + `nids/evaluation/two_stage.py` — core research artifacts
5. `nids/evaluation/metrics.py` — IDS-specific evaluation metrics
6. `docs/API_Reference.md` — module-level API surface
7. `docs/NIDS_Technical_Design.md` — v1 technical details + v2 direction change banner

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (training tested on RTX 3060 Laptop, 6GB VRAM)

Pinned versions (Docker image `flowguard-ids:latest`):

| Package | Pinned |
|---|---|
| Python | 3.10.20 |
| torch | 2.5.1+cu124 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |
| shap | 0.49.1 |
| imbalanced-learn | 0.14.1 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.3 |
| pytest | 9.0.2 |

```bash
pip install -r requirements.txt
# or for editable dev:  pip install -e .
```

## Dataset Placement

Default config expects:

```text
data/raw/
|-- cicids2017/*.csv
`-- unsw_nb15/*.csv
```

Alternative folder names (`CICDS2017/`, `UNSW_NB15/`) are auto-detected by
the loader. Override with `data.data_dir` in the YAML if your path differs.

## Quick Start

### Headline: one command does everything

Trains every model in `pipeline.models` × every direction in `pipeline.directions`
× every seed in `pipeline.seeds`, in **both** binary and multiclass label modes,
then runs the XI2S two-stage cascade evaluation and aggregates a summary JSON.

```bash
python scripts/run_two_stage_pipeline.py --config configs/default.yaml
```

Results land under `artifacts/seed<S>/<dir>[_multiclass]/<model>/...` and the
aggregated cascade summary is written to `artifacts/two_stage_summary.json`.

### Override without touching YAML

```bash
# Only train/evaluate a single backbone + single seed
python scripts/run_two_stage_pipeline.py --config configs/default.yaml \
  --models cnn_bilstm_se_transformer --seeds 42

# Skip training (reuse existing artifacts), only rerun the cascade eval
python scripts/run_two_stage_pipeline.py --config configs/default.yaml --no-do-train

# Custom directions
python scripts/run_two_stage_pipeline.py --config configs/default.yaml \
  --directions cicids2017:cicids2017,cicids2017:unsw_nb15
```

### Single-model debug (skips cascade eval)

```bash
# Binary (Stage-1) training of the primary backbone on CIC → CIC
python scripts/train.py --config configs/default.yaml \
  --models cnn_bilstm_se_transformer \
  --train-dataset cicids2017 --test-dataset cicids2017

# Multiclass (Stage-2). Artifact lands under artifacts/.../_multiclass/ automatically.
python scripts/train.py --config configs/default.yaml --label-mode multiclass \
  --models cnn_bilstm_se_transformer \
  --train-dataset cicids2017 --test-dataset cicids2017
```

Common `train.py` flags:

| Flag | Purpose |
|---|---|
| `--models NAME[,NAME...]` / `--models all` | Which model(s) to train; default reads from the config |
| `--train-dataset / --test-dataset` | Direction override (default from `data.train_dataset`/`data.test_dataset`) |
| `--label-mode {binary,multiclass}` | Task; multiclass automatically isolates artifacts under `_multiclass` suffix |
| `--seed S` | Override `cfg.runtime.seed`; run dir gets a `seed<S>` tag |
| `--resume` | Continue an interrupted deep-model run from `checkpoint_last.pt` |
| `--force` | Retrain finished models (skip-existing is on by default) |
| `--one-click` | All models + auto preprocess + auto feature-selection + resume |
| `--cross-dataset-enhancements / --no-cross-dataset-enhancements` | Enable/disable `{AUC aux loss, Platt scaling, label smoothing ε=0.05}` (auto-on for train ≠ test) |
| `--imbalance-strategy {auto,smote,oversampling,weighted_sampler,none}` | Default `auto` → SMOTE |

### Single-pair cascade evaluation

```bash
python scripts/eval_two_stage.py \
  --stage1-run artifacts/seed42/cicids2017_to_cicids2017/cnn_bilstm_se_transformer \
  --stage2-run artifacts/seed42/cicids2017_to_cicids2017_multiclass/cnn_bilstm_se_transformer \
  --data-file data/processed/cicids2017/data_multiclass.npz \
  --output-dir artifacts/seed42/cicids2017_to_cicids2017_multiclass/two_stage/cnn_bilstm_se_transformer
```

Outputs: `two_stage_report.json`, `two_stage_predictions.npz`,
`figures/two_stage_confusion_matrix.png`.

### Lightweight deliverable model

```bash
# 1) Ensure base model is trained; 2) auto-generate reduced_data.npz from SHAP;
# 3) retrain classical model on the SHAP-reduced feature space.
python scripts/train_lightweight.py --config configs/default.yaml --model random_forest
```

### SHAP analysis (manual)

SHAP is normally invoked implicitly when training a `*_topk` model. To run it
directly on a trained checkpoint:

```bash
python scripts/shap_analysis.py \
  --model artifacts/seed42/cicids2017_to_cicids2017/cnn_bilstm_se_transformer/runs/<ts>/best_model.pt \
  --config configs/default.yaml
```

Default reference: `cicids2017 → cicids2017` with `cnn_bilstm_se` (configurable
under `shap:` in the YAML). Outputs include `shap_values.npy`,
`feature_importance.npy`, `top{20,30,50}_idx.npy`, and `feature_ranking.json`.

## Configuration (`configs/default.yaml`)

The single config file drives everything. Key sections:

```yaml
data:
  train_dataset: cicids2017      # default direction (overridable via --train-dataset)
  test_dataset: unsw_nb15
  label_mode: binary             # or multiclass (overridable via --label-mode)
  batch_size: 512

model:
  name: cnn_bilstm_se
  input_dim: 55                  # unified 55-dim NetFlow feature space
  num_classes: 2                 # auto-resolved per (dataset, label_mode)
  transformer_layers: 2          # only consumed by cnn_bilstm_se_transformer
  transformer_heads: 4
  transformer_dim_feedforward: 512

training:
  num_epochs: 30
  learning_rate: 0.001
  optimizer: adamw
  selection_metric: recall_at_far_1pct   # model selection target (binary)
  loss_type: bce                         # bce | focal
  use_auc_loss: false                    # pairwise AUC aux loss (Yan et al. 2003)
  use_platt_calibration: false           # Platt scaling on source-validation logits

shap:
  top_k: 30                              # default Top-K for *_topk variants
  reference_model_name: cnn_bilstm_se
  reference_train_dataset: cicids2017
  reference_test_dataset: cicids2017

pipeline:
  # These fields become the defaults for run_two_stage_pipeline.py,
  # run_experiments.py, and train.py. Any CLI flag overrides the field for that invocation.
  models:
    - cnn_bilstm_se_transformer
    - cnn_bilstm_se
    - random_forest
    - xgboost
  seeds: [42, 43, 44]
  directions:
    - [cicids2017, cicids2017]
    - [unsw_nb15, unsw_nb15]
  do_train: true
  stage1_threshold: 0.5
  benign_class: 0
  summary_path: artifacts/two_stage_summary.json
```

## Output Layout

Artifact paths (multi-seed, both label modes):

```text
artifacts/
|-- seed<S>/
|   |-- <train>_to_<test>/                   # Stage-1 binary runs
|   |   `-- <model>/
|   |       |-- latest_run.txt, latest_report.txt, latest_run.json
|   |       |-- runs_index.json
|   |       `-- runs/
|   |           `-- <timestamp>_<Model>_<Strategy>_seed<S>/
|   |               |-- best_model.pt | best_model.pkl
|   |               |-- checkpoint_last.pt         # for --resume
|   |               |-- report.json
|   |               |-- run_manifest.json
|   |               |-- resolved_config.yaml
|   |               |-- training_history.csv
|   |               |-- test_predictions.npz
|   |               |-- platt_calibration.npz | .json   # when enabled
|   |               |-- training_curves.png
|   |               `-- figures/
|   |                   |-- confusion_matrix.png
|   |                   |-- nids_key_metrics.png
|   |                   `-- split_distribution.png
|   `-- <train>_to_<test>_multiclass/          # Stage-2 multiclass runs (same sub-tree)
|       |-- <model>/
|       |   `-- runs/...
|       `-- two_stage/                         # Cascade evaluation outputs
|           `-- <model>/
|               |-- two_stage_report.json      # baseline_multiclass_metrics / two_stage_metrics / delta / stage1_stats
|               |-- two_stage_predictions.npz  # y_true / y_baseline / y_two_stage / stage1_binary / stage1_scores
|               `-- figures/two_stage_confusion_matrix.png
|-- experiments/
|   `-- experiment_status.json                 # run_experiments.py batch summary
|-- two_stage_summary.json                     # run_two_stage_pipeline.py aggregate across all targets
|-- shap/shared/<ref_train>_to_<ref_test>/<ref_model>/
|   |-- shap_values.npy / feature_importance.npy / sorted_feature_idx.npy
|   `-- top{20,30,50}_idx.npy / topk_indices.json
`-- feature_selection/<train>_to_<test>/<model>_top<K>/
    `-- reduced_data.npz
```

## Evaluation Metrics

`nids.evaluation.metrics.compute_nids_metrics` returns both binary-aware and
multiclass-aware metrics.

### Binary (requires `y_score`)

| Metric | Description |
|---|---|
| `recall_at_far_1pct` | Best recall achievable while keeping FAR ≤ 1% (**default selection target**) |
| `recall_at_far_5pct` | Same under FAR ≤ 5% |
| `best_f1` / `best_f1_threshold` | Optimal F1 across thresholds + its threshold |
| `pr_auc` | Area under the Precision-Recall curve |
| `roc_auc` | Area under the ROC curve |
| `ece` | Expected Calibration Error (Naeini et al., 2015) |

### Multiclass

| Metric | Description |
|---|---|
| `accuracy` | Overall accuracy |
| `macro_f1` | Macro-averaged F1 across all classes |
| `weighted_f1` | Frequency-weighted F1 |
| `mcc` | Matthews correlation coefficient |
| `per_class_f1 / per_class_recall / per_class_precision` | Per-class breakdown (dict keyed by class label) |
| `confusion_matrix` | Nested-list confusion matrix |

### Shared

| Metric | Description |
|---|---|
| `avg_attack_recall` | Mean recall across non-benign classes |
| `attack_macro_precision` | Mean precision across non-benign classes |
| `benign_false_alarm_rate` | FAR for the benign class |
| `attack_miss_rate` | `1 − avg_attack_recall` |

### Model selection

`training.selection_metric` picks the validation-time checkpoint. Default
`recall_at_far_1pct` for binary;
`_apply_label_mode_overrides` auto-switches to `macro_f1` for multiclass when
the user has not set one explicitly.

## Docker

`Dockerfile` provides a reproducible GPU training environment. Base image:
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`.

```bash
# Build
docker build -t flowguard-ids .

# Headline pipeline (GPU required)
docker run --rm --gpus all -v $(pwd):/workspace flowguard-ids \
  python scripts/run_two_stage_pipeline.py --config configs/default.yaml

# Interactive shell
docker run -it --gpus all -v $(pwd):/workspace flowguard-ids bash
```

On Windows the volume mount is `-v E:\flowguard-ids:/workspace` (do NOT run
`docker build` over SSH — invoke it from a local Windows terminal).

## Testing

```bash
pytest -q
# Current baseline: 66 passed, 7 skipped (skips are torch-dependent tests
# when torch is not installed in the local env).
```

## Notes

- **Skip-existing is on by default**. Pass `--force` only when you actually want to retrain finished models.
- **Multiclass artifact isolation**: multiclass runs get a `_multiclass` suffix on their base output directory, so binary and multiclass runs for the same (direction, model) never overwrite each other.
- **Cross-dataset enhancement bundle** `{AUC aux loss, Platt scaling, label smoothing ε=0.05}` auto-activates on `train != test` directions; toggle per invocation with `--cross-dataset-enhancements` / `--no-cross-dataset-enhancements` or globally via `pipeline.cross_dataset_enhancements`.
- **Cross-dataset transfer is secondary**. To run it, add `[cicids2017, unsw_nb15]` to `pipeline.directions` in the YAML (same-dataset binary + multiclass remain the primary tracks).
