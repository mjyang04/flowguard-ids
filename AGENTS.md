# FlowGuard IDS

This file adapts [CLAUDE.md](/Users/mj/flowguard-ids/CLAUDE.md) into project instructions that Codex can use directly. It supplements the global Codex and ECC rules with repo-specific guidance.

## Project Summary

Graduation design project: a lightweight network intrusion detection system using CNN-BiLSTM-SE, SHAP explainability, and cross-dataset generalization on CICIDS2017 and UNSW-NB15.

- Python: 3.10.20
- PyTorch: 2.5.1+cu124
- Main package: `nids/`
- Config source: `configs/default.yaml`
- Primary execution surface: `scripts/`

## Repository Map

```text
nids/
  config.py          Frozen dataclasses plus YAML loader
  data/              Preprocessing, cross-dataset alignment, dataset builders
  models/            Deep and classical models plus registry
  training/          Trainer loop, optimizers, callbacks
  evaluation/        Metrics, latency, evaluation
  features/          SHAP analysis and feature selection
  utils/             Logging, IO, reproducibility, visualization
scripts/             Entry points for preprocess, train, evaluate, SHAP
tests/               Pytest suite
```

Entry points live in `scripts/`, not `nids/`.

## Model Keys

- `cnn_bilstm`: baseline deep model
- `cnn_bilstm_se`: main model with SE attention
- `cnn_bilstm_se_topk`: main model on reduced features
- `random_forest`: lightweight deliverable
- `xgboost`: lightweight deliverable

`cnn_bilstm_attention` exists in code but is intentionally excluded from training commands unless the user explicitly asks for it.

## Core Commands

```bash
python scripts/preprocess_cross_dataset.py --config configs/default.yaml

python scripts/train.py --config configs/default.yaml --cross-dataset \
  --train-dataset cicids2017 --test-dataset unsw_nb15 --one-click

python scripts/run_experiments.py --config configs/default.yaml --one-click

python scripts/shap_analysis.py --model <path>/best_model.pt --config configs/default.yaml

pytest -q
```

## Development Workflow

- Edit code locally on the Mac workspace.
- Run local tests first whenever possible. Default verification is `pytest -q`.
- Use the Windows machine only for GPU, CUDA, Docker, or full-dataset work that cannot run locally.
- Ask the user before starting remote Windows work.
- Raw datasets already exist on the Windows machine under `E:\flowguard-ids\data\raw\`. Do not re-download them.

Standard remote flow:

1. Finish local edits.
2. Run the local checks that are feasible.
3. Ask the user before remote execution.
4. After confirmation, push local code, pull on Windows, then run inside Docker.

## Remote Training Machine

- SSH target: `ssh Lenovo@10.70.72.246`
- Project path: `E:\flowguard-ids`
- Docker image: `flowguard-ids:latest`
- Persistent container: `flowguard`

Before any SSH session, verify the host:

```bash
~/.claude/scripts/find-windows.sh 10.70.72.246
```

- If the script returns the IP, continue.
- If it returns `FAIL`, stop and ask the user for the current IP before doing anything remote.

Useful remote commands:

```bash
ssh Lenovo@<verified-ip> "<command>"
ssh Lenovo@<verified-ip> "git -C E:\flowguard-ids fetch && git -C E:\flowguard-ids status -uno"
ssh Lenovo@<verified-ip> "git -C E:\flowguard-ids pull"
docker exec -it flowguard bash
docker run --rm --gpus all -v E:\flowguard-ids:/workspace flowguard-ids <command>
docker run -d --gpus all -v E:\flowguard-ids:/workspace --name <name> flowguard-ids tmux new-session -d -s train '<cmd>'
```

Remote execution rules:

- Check whether the Windows checkout is up to date before training or evaluation.
- If `git status -uno` reports the branch is behind, pull first.
- If Docker daemon is down, tell the user to open Docker Desktop manually.
- `docker build` must be run on the Windows local desktop session, not over SSH.
- Notify the user when long-running training finishes.

## Configuration Rules

- All tunable values belong in `configs/*.yaml`. Do not hardcode hyperparameters in Python.
- Config loads into frozen dataclasses in [nids/config.py](/Users/mj/flowguard-ids/nids/config.py). Do not mutate config objects.
- The unified feature space is 55 dimensions and is defined in [nids/config.py](/Users/mj/flowguard-ids/nids/config.py).

## Data And Artifacts

- Raw CSV data: `data/raw/`
- Preprocessed arrays: `data/processed/`
- SHAP sample cache: `data/shap_samples/`
- Training outputs: `artifacts/<train>_to_<test>/<model>/runs/<timestamp>_<Model>_<Strategy>/`

Each run directory is expected to be self-contained with weights, config snapshot, metrics, and figures.

## Code And Testing Rules

- Follow PEP 8 and type annotate all function signatures.
- Use `logging`, not `print()`, for operational output. `tqdm` is fine for progress bars.
- Prefer immutable patterns. Do not mutate function arguments unless there is a strong reason already established in the code.
- Keep modules focused. Target under 400 lines when practical.
- Tests live in `tests/` and shared fixtures live in `tests/conftest.py`.
- Run `pytest -q` before finishing a change unless the task is docs-only or the environment blocks it.

## Git Rules

- Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Never commit `data/raw/`, `data/processed/`, `artifacts/`, or trained model files such as `*.pt` and `*.pkl`.

## Project-Specific Notes

- `--one-click` skips models that are already trained. Use `--force` only when clean retraining is intended.
- `--resume` continues from `checkpoint_last.pt` for interrupted deep-model runs.
- The SHAP reference model is `cnn_bilstm_se` trained on `cicids2017 -> cicids2017` and should be reused unless the user asks to regenerate it.
- `training.selection_metric` defaults to `recall_at_far_1pct` in config YAML files. Do not change it without explicit instruction.
- `--auto-preprocess` and `--auto-feature-selection` are intended to resolve dependencies automatically.
