# Chapter 3 — Research Methodology

## 3.1 Background

This chapter describes the methodology used by the present study to reproduce the Contrastive Learning using Augmented Negatives (CLAN) framework of Wilkie et al. (2025), to compare it against seven self-supervised learning (SSL) baselines, and to conduct a structured ablation of its key design choices. Section 3.2 states the notation and problem formulation. Section 3.3 specifies the encoder architecture (ContrastiveMLP). Section 3.4 derives the CLAN loss function and relates it to the alignment–uniformity framework of Wang and Isola (2020). Section 3.5 describes the augmentation family. Section 3.6 defines the centroid-based anomaly score and the few-shot fine-tune protocol. Section 3.7 summarises optimisation and inference. Section 3.8 details the dataset (Lycos2017) and the preprocessing pipeline. Section 3.9 specifies the seven baseline SSL methods. Section 3.10 fixes the evaluation protocol. Section 3.11 enumerates the six ablation axes. Section 3.12 lists implementation and reproducibility details.

## 3.2 Notation and Problem Formulation

Let $x \in \mathbb{R}^d$ denote a flow feature vector, where $d = 72$ for Lycos2017 after the seven metadata columns have been dropped (see §3.8). Let $y \in \{0, 1, \dots, C\}$ denote its label, with $y = 0$ reserved for benign traffic and $y > 0$ for one of $C$ attack classes. Given a corpus $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, the partition $\mathcal{D}_B = \{(x_i, 0) : y_i = 0\}$ contains only benign flows, and $\mathcal{D}_A = \mathcal{D} \setminus \mathcal{D}_B$ contains only attack flows.

The present study decomposes the NIDS problem into two sub-problems:

1. **Anomaly detection.** Using only $\mathcal{D}_B$ at training time, the study learns an encoder $f_\theta : \mathbb{R}^d \to \mathbb{R}^{d'}$ and a scoring function $s : \mathbb{R}^{d'} \to \mathbb{R}$ such that $s(f_\theta(x))$ is higher for attack flows than for benign flows. At deployment, $s$ is compared against a threshold.
2. **Few-shot multiclass attack classification.** Given the pretrained $f_\theta$ and a very small labelled subset of $K$ samples per class, the study fits a lightweight classification head $g_\phi : \mathbb{R}^{d'} \to \mathbb{R}^{C+1}$ on top of $f_\theta$ and reports macro-F1 on a held-out test set.

## 3.3 Encoder: ContrastiveMLP

Wilkie et al. (2025) deliberately adopt a lightweight multi-layer perceptron rather than a deeper architecture for two reasons: (a) NIDS flow features are tabular and do not benefit from the inductive biases of convolutional or recurrent networks once the raw packets have been aggregated by CICFlowMeter, and (b) inference latency is a practical constraint in production deployments. The present study adopts their encoder unchanged (see `nids/models/contrastive_mlp.py`), which is a four-layer residual MLP with ReLU activations and an optional linear projection head.

Let $h_0 = x$ and let $h_\ell = \mathrm{DenseBlock}_\ell(h_{\ell-1})$ for $\ell \in \{1, 2, 3, 4\}$, where

$$
\mathrm{DenseBlock}_\ell(h) = \underbrace{\sigma\bigl(W_\ell h + b_\ell\bigr)}_{\text{linear + ReLU}} + \underbrace{R_\ell(h)}_{\text{residual}}.
$$

Here $R_\ell$ is the identity when the input and output dimensions match and a linear resizing otherwise, following the residual design of He et al. (2016). Stacking four DenseBlocks with hidden width 1024 yields the CLAN encoder. A projection head $P : \mathbb{R}^{1024} \to \mathbb{R}^{64}$ then produces the embedding $f_\theta(x) = P(h_4)$. Design alternatives explored in the ablation (§3.11) include encoder depth $\in \{2, 3, 4, 6\}$ and the presence or absence of an explicit L2 normalisation step at the output of $P$.

## 3.4 Objective: CLAN Loss

The CLAN loss of Wilkie et al. (2025) occupies a novel position in the self-supervised learning taxonomy reviewed in Chapter 2. Whereas SimCLR (Chen et al., 2020) pulls augmented pairs together, BYOL (Grill et al., 2020) dispenses with negative samples entirely, and Barlow Twins (Zbontar et al., 2021) replaces negatives with a redundancy-reduction term, CLAN takes a third option: it retains an explicit repulsive term, but the quantity it repels is not another in-batch sample — it is the *augmented* version of the anchor.

Given a batch $\mathcal{B} = \{x_i\}_{i=1}^{B}$ of benign samples and their augmented counterparts $\mathcal{B}^{aug} = \{x_i^{aug}\}_{i=1}^{B}$ (produced by the augmentation module of §3.5), let $z_i = f_\theta(x_i)$ and $z_i^{aug} = f_\theta(x_i^{aug})$. Define the cosine distance

$$
D(u, v) = \frac{1 - \dfrac{\langle u, v \rangle}{\|u\|_2 \|v\|_2}}{2} \in [0, 1].
$$

The CLAN loss comprises two terms balanced by a scalar $\alpha \in [0, 1]$:

**Intra-class (alignment).** Same-distribution samples are pulled together:

$$
\mathcal{L}_{\text{intra}} = \frac{1}{|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}} D(z_i, z_j), \qquad \mathcal{P} = \{(i, j) : i \neq j,\ D(z_i, z_j) > \epsilon\}.
$$

**Inter-class (hinge repulsion).** Augmented counterparts are pushed away up to a margin $m$:

$$
\mathcal{L}_{\text{inter}} = \frac{1}{|\mathcal{N}|} \sum_{(i, j) \in \mathcal{N}} \bigl[m - D(z_i, z_j^{aug})\bigr]_+, \qquad \mathcal{N} = \{(i, j) : [m - D(z_i, z_j^{aug})]_+ > \epsilon\}.
$$

The overall objective is

$$
\mathcal{L}_{\text{CLAN}} = \alpha\, \mathcal{L}_{\text{intra}} + (1 - \alpha)\, \mathcal{L}_{\text{inter}}.
$$

Three remarks connect the objective to the related work. First, the alignment–uniformity framework of Wang and Isola (2020) proves that minimising InfoNCE is equivalent to optimising a combination of alignment (similarity of positive pairs) and uniformity (even distribution on the hypersphere); the two terms of the CLAN loss are the direct NIDS-specific analogues, with $\mathcal{L}_{\text{intra}}$ as alignment and $\mathcal{L}_{\text{inter}}$ as a *targeted* uniformity term that excludes exactly the regions into which augmentations map — the regions the model should treat as anomalous at test time. Second, the practice of treating augmented samples as hard negatives is a domain-specific instance of the MoCHi proposal of Kalantidis et al. (2020); the NIDS-specific twist is that practitioners can design augmentations to resemble the anomaly distribution, making the negatives' "hardness" controllable. Third, in contrast to the supervised-contrastive ConFlow of Liu et al. (2023) and the benign-only SSCL-IDS of Golchin et al. (2024), the CLAN objective never uses attack labels during pretraining, matching the practical constraint in which attack labels are scarce or unreliable (Engelen et al., 2021).

The implementation is given in `nids/training/losses/clan.py`; its numerical correctness, alpha-bound validation, and functional-module equivalence are unit tested in `tests/test_clan_loss.py`.

## 3.5 Augmentation Family

The augmented view $x_i^{aug}$ is produced by one of five stateless functions, selected via the configuration field `augmentation.name`:

- **UniformResample** (paper default). For each feature $k$ selected by a Bernoulli($p_f$) mask, the value is replaced by a sample from $\mathcal{U}(-m_v, m_v) + \mu_v$.
- **GaussianResample.** The masked positions are replaced by samples from $\mathcal{N}(\mu_v, \sigma_v^2)$.
- **Jitter.** Gaussian noise from $\mathcal{N}(\mu_v, \sigma_v^2)$ is *added* to the masked positions rather than replacing them.
- **ZeroOutNoise.** Masked positions are set to zero.
- **FeatureShuffle.** Each masked position $k$ is filled with the value from a permuted feature $\pi(k)$.

All augmentations execute under `torch.no_grad()` to prevent gradient leakage. Upstream CLAN uses UniformResample with $p_f = 0.1$, $p_s = 1.0$, $m_v = 1.7$, $\mu_v = 0.0$; the present study treats the augmentation family, its strength $p_f$, and the parameter $m_v$ as ablation axes (§3.11).

## 3.6 Anomaly Scoring and Few-Shot Fine-Tuning

**Anomaly detection score.** Following Wilkie et al. (2025), the benign centroid

$$
\mu = \frac{1}{|\mathcal{D}_B|} \sum_{x \in \mathcal{D}_B} f_\theta(x)
$$

is computed once on the training split and cached. At test time the score is $s(x) = -\cos(\mu, f_\theta(x))$. A higher score indicates a larger angular deviation from the benign manifold. The evaluation reports per-class one-vs-benign AUROC (§3.10) rather than a single threshold, preserving operating-point independence.

**Few-shot fine-tuning.** Given a pretrained encoder $f_\theta$ and a labelled subset $\mathcal{D}^{(K)}$, the present study attaches a linear head $g_\phi(z) = W z + b$ with $W \in \mathbb{R}^{(C+1) \times d'}$. Following the upstream implementation in `finetune_clan.py`, both $f_\theta$ and $g_\phi$ are trained jointly with cross-entropy loss for 100 epochs at $\text{lr} = 10^{-3}$, weight decay $10^{-6}$, batch size 64, and an optional label smoothing of 0.0 or 0.1. Test metrics are macro-F1, macro-recall, macro-precision, and accuracy; §3.10 justifies the choice.

## 3.7 Training, Optimisation, and Inference

**Optimiser.** AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay 0 during pretraining and $10^{-6}$ during fine-tuning, matching the upstream settings.

**Learning-rate schedule.** Warmup-cosine annealing (Loshchilov & Hutter, 2017), as implemented in `nids/training/schedules.py`: linear warmup from $10^{-6}$ to $10^{-4}$ over the first 10% of training steps, followed by cosine decay back to $10^{-6}$ over the remainder. The `WarmupCosineSchedule` class matches the one used by Wilkie et al.

**Batching.** The pretraining batch size is 8192 (upstream default). A `WeightedRandomSampler` draws samples with weights proportional to inverse class frequency, preventing minority-class collapse in the fine-tuning phase. See `nids/data/loaders.py::tabular_dl`.

**Seeding.** The utility `nids.utils.reproducibility.seed_everything(seed)` sets NumPy, PyTorch (CPU and CUDA), Python's `random` module, and the `PYTHONHASHSEED` environment variable, and enables `torch.backends.cudnn.deterministic=True` with `benchmark=False`. Three pretraining seeds, $\{42, 43, 44\}$, are used for mean-and-standard-deviation reporting; the data split seed is fixed at 39 058 032 so that every seed operates on the same partition.

**Inference.** A single forward pass through $f_\theta$ followed by one dot-product with the cached centroid yields the anomaly score. The inference cost is $O(d' + |\mathcal{D}_B| / |\mathcal{B}|)$, substantially lighter than the BYOL, SimSiam, and VICReg baselines, which require either a memory bank or an auxiliary predictor during evaluation (Wilkie et al., 2025).

## 3.8 Dataset: Lycos2017

### 3.8.1 Provenance

Lycos2017 was released by Rosay et al. (2021) as the corrected version of CICIDS2017. It was produced by re-running a replacement feature extractor (*LycoSTand*) on the original CICIDS2017 packet captures and re-labelling each flow according to the corrected ground truth documented by Engelen et al. (2021) and Rosay et al. (2022). The present study uses the preprocessed single-CSV bundle distributed at `https://lycos-ids.univ-lemans.fr`, cached locally at `data/raw/lycos.csv`.

### 3.8.2 Schema

The CSV contains 79 columns per row. The first seven are metadata that the encoder must never see (`flow_id`, `src_addr`, `src_port`, `dst_addr`, `dst_port`, `ip_prot`, `timestamp`); these are dropped upfront by the `drop_cols` list in `configs/default.yaml`. The remaining 72 columns are CICFlowMeter-style flow statistics, all numeric. A single `label` column carries a string ground truth, taking the value `benign` or one of 12 attack labels matching the CLAN paper's taxonomy (Botnet, DDoS, DoS Golden Eye, DoS Hulk, DoS Slow HTTP Test, DoS Slow Loris, FTP Patator, Portscan, SSH Patator, Web Brute Force, Web XSS, Heartbleed, Web SQL Injection).

### 3.8.3 Split

The present study uses a stratified 50/50 train/test split with `split_seed = 39 058 032`, matching the configuration of Wilkie et al. (2025, `eval_clan.py`). A validation carve-out is disabled by default (`val_ratio = 0.0`); ablations that require validation (§3.11) enable it at 10%. Attack classes with fewer than 100 rows in the development set are held out as a separate *zero-day* split that never reaches the pretraining loader, matching the CLAN paper's evaluation recipe. Finally, when `anomaly_detection = True` (the pretraining regime), every non-benign row is removed from the training split *after* the stratified split, reducing the training set to benign traffic only.

### 3.8.4 Preprocessing

Any residual categorical column that leaks through `drop_cols` triggers a loud `ValueError` rather than a silent one-hot encoding; this catches schema drift early. `NaN` and `±inf` values are clamped to zero. Features are then standardised to zero mean and unit variance using statistics fit on the benign training split only, preventing leakage from attack statistics. The implementation lives in `nids/data/lycos.py::get_data`.

## 3.9 Seven SSL Baselines

All comparison methods run under a shared ContrastiveMLP encoder, a shared augmentation module, a shared DataLoader, a shared set of random seeds, and a shared evaluation protocol. Only the loss function differs. Each baseline is implemented in `nids/training/losses/<name>.py` and selected via the YAML field `loss.name`. Table 3.1 summarises the baselines.

Table 3.1: Seven SSL baselines compared against CLAN under the same encoder, augmentation, and evaluation protocol.

| Method | Reference | Key idea | Positives | Negatives |
|---|---|---|---|---|
| CLAN | Wilkie et al. (2025) | Augmented view is a *negative*; centroid-based anomaly score | Other benign samples | Augmented view |
| SimCLR | Chen et al. (2020) | InfoNCE with in-batch negatives | Augmented view | Other in-batch samples |
| Barlow Twins | Zbontar et al. (2021) | Decorrelation objective (no negatives) | Augmented view | N/A |
| BYOL | Grill et al. (2020) | Momentum target + predictor (no negatives) | Augmented view via EMA target | N/A |
| VICReg | Bardes et al. (2022) | Variance + invariance + covariance terms | Augmented view | N/A |
| SimSiam | Chen and He (2021) | Siamese stop-gradient (no negatives) | Augmented view | N/A |
| ConFlow | Liu et al. (2023) | Supervised contrastive + cross-entropy | Same-class samples | Different-class samples |
| SSCL-IDS | Golchin et al. (2024) | Benign-only SimCLR variant | Augmented view | Other benign in batch |

All baselines use the same encoder width (1024 × 4) and the same 64-dimensional projection head in order to hold the parameter count constant.

## 3.10 Evaluation Protocol

### 3.10.1 Anomaly Detection Evaluation

After pretraining $f_\theta$, the benign centroid $\mu$ is computed on the training split. The test split, merged with the zero-day holdout, is embedded and the score $s(x) = -\cos(\mu, f_\theta(x))$ is computed for each sample. For each attack class $c$, the one-vs-benign AUROC is

$$
\mathrm{AUROC}_c = \Pr\bigl(s(x_a) > s(x_b)\bigr) \quad \text{for } x_a \sim \mathcal{D}_A^c, x_b \sim \mathcal{D}_B.
$$

The primary headline metric is **Mean AUROC** across the 12 attack classes. A per-class breakdown is additionally reported to expose qualitative patterns.

### 3.10.2 Few-Shot Multiclass Evaluation

The study sweeps $K \in \{8, 16, 32, 64, 128, 256, 512, 1024\}$. For each $K$:

1. A `num_benign = K`, `num_mal = K` balanced subset is drawn from the training split (see `nids/data/utils.py::sample_data`).
2. Features are renormalised using this subset's mean and standard deviation, matching the upstream `finetune_clan.py`.
3. The encoder and linear head are fine-tuned for 100 epochs at $\text{lr} = 10^{-3}$, $\text{wd} = 10^{-6}$, $\text{batch size} = 64$.
4. Evaluation is performed on the full held-out test split excluding the fine-tune subset.

The primary metric is **macro-F1**. Following Engelen et al. (2021) and Lanvin et al. (2023), macro-F1 is preferred over accuracy or weighted-F1 because CICIDS2017 and Lycos2017's worst label errors concentrate in precisely the small classes to which macro-F1 is sensitive. Macro-recall, macro-precision, and accuracy are reported as secondary metrics.

### 3.10.3 Reporting

Every reported number is the mean and standard deviation across three pretraining seeds, $\{42, 43, 44\}$. The data split seed is held fixed so that each method pretrains on the same partition. Few-shot subsampling uses its own independent seed of 42 to keep the $K$-sample subsets stable across methods.

## 3.11 Ablation Studies

The present study holds all other knobs at the CLAN defaults and varies the following six axes.

1. **Margin.** $m \in \{0.1, 0.25, 0.5, 1.0, 2.0\}$ (default 0.5). This axis tests whether a softer or harder hinge changes the AUROC / macro-F1 trade-off.
2. **Augmentation family.** $\{\text{UniformResample}, \text{GaussianResample}, \text{Jitter}, \text{ZeroOutNoise}, \text{FeatureShuffle}\}$. This axis tests whether the observed CLAN gain is robust to the augmentation distribution or is tied to the specific family used by Wilkie et al.
3. **Augmentation strength.** $p_f \in \{0.05, 0.1, 0.2, 0.4, 0.8\}$ at fixed UniformResample. This axis tests the *sweet-spot* view of Tian et al. (2020): augmentations that are too weak make the pretext task trivial, and augmentations that are too strong destroy semantic content.
4. **Encoder depth.** Number of DenseBlocks $\in \{2, 3, 4, 6\}$ at fixed width 1024. This axis tests whether the four-layer default is compute-justified.
5. **L2 normalisation of embeddings.** On or off at the output of $P$. This axis tests whether the implicit normalisation inside the cosine distance is sufficient.
6. **Few-shot sample count.** Already part of the headline evaluation. Reports the 8 → 1024 curve on both CLAN and the best-performing baseline.

Each ablation cell is run across the three seeds. The compute budget per cell on an NVIDIA RTX 3060 Laptop GPU (6 GB VRAM) is approximately 35 minutes of pretraining plus 6 minutes of fine-tuning sweep.

## 3.12 Implementation Details

### 3.12.1 Hardware and Software

Code is developed locally on macOS with numerically-pure unit tests executed via `pytest -q`. Training and evaluation run on Windows 11 with an NVIDIA RTX 3060 Laptop GPU (6 GB VRAM), CUDA 12.4, PyTorch 2.5+, and Python 3.10+. The package is installed in editable mode via `pip install -e .`; the full dependency list is maintained in `requirements.txt`.

### 3.12.2 Configuration

Every tunable knob is a field of `configs/default.yaml`, organised into seven sections: `data`, `model`, `loss`, `augmentation`, `training`, `finetune`, and `runtime`. The command-line entry points (`scripts/train.py`, `scripts/eval.py`, `scripts/finetune.py`) read this YAML via `nids.config.load_config` and optionally override `runtime.device` through a flag. No hyperparameter is hardcoded in Python source.

### 3.12.3 Artefact Layout

Each run writes to `artifacts/<loss.name>/<timestamp>_seed<S>/`, producing the following files: the checkpoint `best_model.pt.tar`, the resolved configuration `resolved_config.yaml`, the evaluation report `eval_report.json`, the fine-tune reports `finetune_report_shots<K>.json`, and a training-curves plot `training_curves.png`. Raw data (`data/raw/`), preprocessed caches (`data/processed/`), and artefacts (`artifacts/`) are gitignored.

### 3.12.4 Reproducibility

Determinism is enforced at three levels: random seeding (see §3.7), dependency pinning (lower bounds in `requirements.txt` and exact versions on the Windows training rig), and code attribution (every ported module cites the upstream CLAN source). Missing upstream modules — specifically `data/load_data.py`, `data/loaders.py`, and `data/utils.py`, which are not checked into the CLAN repository — are flagged as re-implementations in their module docstrings. Unit tests in `tests/test_metrics.py`, `tests/test_contrastive_mlp.py`, `tests/test_clan_loss.py`, and `tests/test_augmentations.py` guard the numerical correctness of the metric, encoder, loss, and augmentation implementations.
