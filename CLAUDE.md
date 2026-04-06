# FlowGuard IDS

## Project Overview

Graduation design project: a lightweight network intrusion detection system using **CNN-BiLSTM-SE + SHAP explainability + cross-dataset generalization** on CICIDS2017 and UNSW-NB15.

- **Python**: 3.10.20
- **PyTorch**: 2.5.1+cu124 (CUDA 12.4, cuDNN 9.1.0)
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU
- **Core deps**: numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.2.0, shap 0.49.1, imbalanced-learn 0.14.1, matplotlib 3.10.8, tqdm 4.67.3, PyYAML 6.0.3, joblib 1.5.3, pytest 9.0.2
- **Package**: `nids/` (installed via `setup.py`)
- **Config**: YAML-driven (`configs/default.yaml`)

## Architecture

```
nids/
├── config.py          # Frozen dataclasses + YAML loader
├── data/              # Preprocessing, cross-dataset alignment, dataset builders
├── models/            # Deep models (CNN-BiLSTM variants), classical (RF, XGBoost), registry
├── training/          # Trainer loop, optimizer/scheduler, callbacks (early stopping)
├── evaluation/        # Metrics, latency, evaluator
├── features/          # SHAP analysis, feature selection (Top-K)
└── utils/             # Logging, IO, visualization, reproducibility
```

Entry points are in `scripts/` — not in `nids/`.

## Models

| Model | Type | Key |
|-------|------|-----|
| `cnn_bilstm` | Deep | Baseline |
| `cnn_bilstm_se` | Deep | Main model (SE attention) |
| `cnn_bilstm_se_topk` | Deep | Main model on reduced features |
| `random_forest` | Classical | Lightweight deliverable |
| `xgboost` | Classical | Lightweight deliverable |

`cnn_bilstm_attention` exists in code but is excluded from training commands.

## Key Commands

```bash
# Preprocess
python scripts/preprocess_cross_dataset.py --config configs/default.yaml

# One-click train all models
python scripts/train.py --config configs/default.yaml --cross-dataset \
  --train-dataset cicids2017 --test-dataset unsw_nb15 --one-click

# Run all experiments (3 settings x 5 models)
python scripts/run_experiments.py --config configs/default.yaml --one-click

# SHAP analysis
python scripts/shap_analysis.py --model <path>/best_model.pt --config configs/default.yaml

# Tests
pytest -q
```

## Development Rules

### Config

- All hyperparameters live in `configs/*.yaml` — never hardcode values in Python.
- Config is loaded into frozen dataclasses (`nids/config.py`). Do not mutate config objects.

### Data Pipeline

- Raw CSV data goes in `data/raw/`. Preprocessed `.npz` artifacts go in `data/processed/`.
- SHAP samples are cached in `data/shap_samples/`.
- The unified feature space is 55 dimensions (defined in `nids/config.py`).
- Both datasets are aligned to this shared feature set during cross-dataset preprocessing.

### Training Artifacts

All outputs land in `artifacts/<train>_to_<test>/<model>/runs/<timestamp>_<Model>_<Strategy>/`.
Each run folder is self-contained: model weights, config snapshot, metrics, figures.

### Code Style

- Follow PEP 8. Use type annotations on all function signatures.
- Use `logging` module, never `print()` for operational output (`tqdm` is fine for progress bars).
- Keep files under 400 lines. Extract helpers if a module grows large.
- Immutable-first: prefer frozen dataclasses, don't mutate function arguments.

### Testing

- Tests are in `tests/` using pytest.
- Fixtures are in `tests/conftest.py`.
- Run `pytest -q` before committing.

### Git

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Never commit `data/raw/`, `data/processed/`, `artifacts/`, or `*.pt`/`*.pkl` model files.

## Things to Know

- `--one-click` mode skips already-trained models. Use `--force` for clean retraining.
- `--resume` continues from `checkpoint_last.pt` for interrupted deep-model runs.
- SHAP reference model is always `cnn_bilstm_se` trained on `cicids2017 -> cicids2017`. Generated once, reused everywhere.
- `selection_metric` in config (`avg_attack_recall`) drives model checkpoint selection — this is intentional for IDS where per-attack-class recall matters more than overall accuracy.
- `--auto-preprocess` and `--auto-feature-selection` flags handle dependency resolution automatically.