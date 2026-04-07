# FlowGuard IDS

## Project Overview

Graduation design project: a lightweight network intrusion detection system using **CNN-BiLSTM-SE + SHAP explainability + cross-dataset generalization** on CICIDS2017 and UNSW-NB15.

- **Python**: 3.12
- **PyTorch**: 2.5.1+cu128 (CUDA 12.8, cuDNN 9)
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM) — primary
- **Core deps**: numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.2.0, shap 0.49.1, imbalanced-learn 0.14.1, matplotlib 3.10.8, tqdm 4.67.3, PyYAML 6.0.3, joblib 1.5.3, pytest 9.0.2
- **Package**: `nids/` (installed via `setup.py`)
- **Config**: YAML-driven (`configs/default.yaml`)

## Remote Training Machine (Linux - Primary GPU)

SSH access: `ssh -p 46526 root@connect.westd.seetacloud.com`

- **OS**: Ubuntu 22.04.5 LTS
- **GPU**: NVIDIA GeForce RTX 5090, 32GB VRAM
- **CUDA**: 13.0, Driver 580.76.05
- **Storage**: 3x NVMe SSD RAID5, ~14TB (mounted at `/`)
- **Docker**: 未安装（需先安装）
- **Python**: 未安装（需先安装）

### 远程操作规范

- 项目路径：`/root/autodl-tmp/flowguard-ids`（已克隆）
- 代码同步：`ssh -p 46526 root@connect.westd.seetacloud.com "cd /root/autodl-tmp/flowguard-ids && git pull"`
- **安装 Docker（首次）：**
  ```bash
  ssh -p 46526 root@connect.westd.seetacloud.com
  apt-get update && apt-get install -y docker.io docker-compose
  systemctl start docker && systemctl enable docker
  # 安装 NVIDIA Container Toolkit
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-docker.list
  apt-get update && apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  ```
- 构建镜像：`ssh -p 46526 root@connect.westd.seetacloud.com "cd /root/autodl-tmp/flowguard-ids && docker build -t flowguard-ids:latest ."`
- 运行容器：`docker run -d --gpus all --name flowguard -v /root/autodl-tmp/flowguard-ids:/workspace flowguard-ids bash -c 'while true; do sleep 3600; done'`
- 进入容器：`docker exec -it flowguard bash`
- 长时间训练：`docker exec flowguard bash -c 'cd /workspace/autoresearch && python train.py'`
- **训练/测试前，必须先检查代码是否最新**

### 传输数据（从 Windows）

在 Windows 上打包后传输：
```bash
# Windows 上
docker run --rm -v E:\flowguard-ids:/data alpine tar -cvf - -C /data data/raw data/processed | ssh -p 46526 root@connect.westd.seetacloud.com "cd /root/autodl-tmp/flowguard-ids && tar -xvf -"
```

## Remote Training Machine (Windows - 备用)

SSH access: `ssh Lenovo@10.70.72.246`

- **OS**: Windows 11
- **CPU**: Intel Core i7-12700H (12th Gen)
- **RAM**: 32GB
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU, 6GB VRAM
- **CUDA**: 13.2, Driver 595.79
- **Docker**: Docker Desktop 29.3.0（需手动启动 Docker Desktop GUI）
- **WSL**: 仅有 docker-desktop，无 Ubuntu

### Windows 远程操作规范

- 项目完整路径：`E:\flowguard-ids`（含 data/raw、data/processed、artifacts）
- Docker 镜像：`flowguard-ids:latest`（已构建，含 CUDA + 所有依赖）
- 持久容器：`flowguard`（`sleep infinity` 保持运行，重启后需 `docker start flowguard`）
- 进入容器：`docker exec -it flowguard bash`
- **每次 SSH 操作前，先验证目标机器：**
  ```bash
  ~/.claude/scripts/find-windows.sh 10.70.72.246
  ```
  - 输出 IP → 正确，继续操作
  - 输出 FAIL → IP 有误或被其他设备占用，使用 AskUserQuestion 询问用户当前 IP，更新 CLAUDE.md 后重试
- 使用 `ssh Lenovo@<verified-ip> "<command>"` 执行单条命令
- 运行容器统一用：`docker run --rm --gpus all -v E:\flowguard-ids:/workspace flowguard-ids <command>`
- 长时间任务（训练）用 tmux：`docker run -d --gpus all -v E:\flowguard-ids:/workspace --name <name> flowguard-ids tmux new-session -d -s train '<cmd>'`
- docker build 必须在 Windows 本地终端执行（SSH 无法访问 Windows 凭证管理器）
- 代码同步（挂载目录，无需进容器）：`ssh Lenovo@<ip> "git -C E:\flowguard-ids pull"`
- **训练/测试前，必须先检查代码是否最新**：
  ```bash
  ssh Lenovo@<ip> "git -C E:\flowguard-ids fetch && git -C E:\flowguard-ids status -uno"
  ```
  若显示 `Your branch is behind`，先 pull 再操作
- Docker daemon 未运行时，提示用户手动打开 Docker Desktop

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

## Development Workflow

- **代码编辑**：本地 Mac
- **测试**：优先本地 conda 环境；本地无法满足的（CUDA、GPU训练、完整数据集）用 Linux GPU 服务器
- **训练**：Linux GPU 服务器 (RTX 5090)

**执行 GPU 服务器任务的标准流程：**
1. 本地完成代码修改，先跑本地可运行的测试（`pytest -q`）
2. `git push`
3. SSH 到 GPU 服务器 `ssh -p 46526 root@connect.westd.seetacloud.com` 执行 `git pull`
4. 在 Docker 容器中执行训练

Raw data（CICIDS2017、UNSW-NB15）需从 Windows 机器传输或重新下载。

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
- `selection_metric` in config (default: `recall_at_far_1pct`) drives model checkpoint selection — this prioritizes models that detect attacks reliably while keeping false alarm rate low. Other options: `recall_at_far_5pct`, `best_f1`, `pr_auc`, `avg_attack_recall`.
- `--auto-preprocess` and `--auto-feature-selection` flags handle dependency resolution automatically.