```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# CHAPTER 3

# RESEARCH METHODOLOGY

## 3.1 Background

This chapter describes the methodology used by the present study to reproduce the Contrastive Learning using Augmented Negatives (CLAN) framework of Wilkie et al. (2025) on Lycos2017 (Rosay et al., 2021) and to run the same CLAN pipeline on the original CICIDS2017 release (Sharafaldin et al., 2018) as a controlled noisy-label audit. Section 3.2 fixes notation and formalises the problem. Section 3.3 specifies the encoder architecture (ContrastiveMLP). Section 3.4 derives the CLAN loss function and relates it to the alignment–uniformity framework of Wang and Isola (2020). Section 3.5 describes the augmentation family. Section 3.6 defines the centroid-based anomaly score and the few-shot fine-tune protocol. Section 3.7 summarises optimisation and inference. Section 3.8 details the two datasets and their preprocessing pipelines. Section 3.9 specifies the dual-dataset evaluation design that this thesis introduces. Section 3.10 fixes the evaluation protocol. Section 3.11 documents paper-versus-code discrepancies discovered during the port, the minimal ablations carried out within the available compute budget, and honestly declared scope reductions relative to the upstream study. Section 3.12 lists implementation and reproducibility details.

## 3.2 Notation and Problem Formulation

Let $x \in \mathbb{R}^d$ denote a flow feature vector, where $d$ is determined at runtime from the loaded dataset after the metadata columns and any all-zero columns have been dropped. For Lycos2017 (Rosay et al., 2021) the upstream CLAN convention sets $d = 72$; for the original CICIDS2017 release (Sharafaldin et al., 2018) the same preprocessing pipeline yields a slightly different value because the two datasets differ in extractor (LycoSTand versus CICFlowMeter v3) and hence in their zero-column footprint. The encoder consumes whichever $d$ the loader produces; §3.12.3 describes the runtime-dispatch mechanism.

Let $y \in \{0, 1, \dots, C\}$ denote the label, with $y = 0$ reserved for benign traffic and $y > 0$ for one of $C$ attack classes. Given a corpus $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, the partition $\mathcal{D}_B = \{(x_i, 0) : y_i = 0\}$ contains only benign flows, and $\mathcal{D}_A = \mathcal{D} \setminus \mathcal{D}_B$ contains only attack flows.

The present study decomposes the NIDS problem into two sub-problems:

1. **Anomaly detection.** Using only $\mathcal{D}_B$ at training time, the study learns an encoder $f_\theta : \mathbb{R}^d \to \mathbb{R}^{d'}$ and a scoring function $s : \mathbb{R}^{d'} \to \mathbb{R}$ such that $s(f_\theta(x))$ is higher for attack flows than for benign flows. At deployment, $s$ is compared against a threshold.
2. **Few-shot multiclass attack classification.** Given the pretrained $f_\theta$ and a very small labelled subset of $K$ samples per class, the study fits a lightweight classification head $g_\phi : \mathbb{R}^{d'} \to \mathbb{R}^{C+1}$ on top of $f_\theta$ and reports macro-F1 on a held-out test set.

Both sub-problems are evaluated on each of the two datasets independently, using identical hyperparameters, identical encoder weights architecture, identical augmentation, and identical evaluation protocol. The difference in reported numbers therefore isolates the effect of dataset label quality from every other confound.

## 3.3 Encoder: ContrastiveMLP

Wilkie et al. (2025) deliberately adopt a lightweight multi-layer perceptron rather than a deeper architecture for two reasons: (a) NIDS flow features are tabular and do not benefit from the inductive biases of convolutional or recurrent networks once the raw packets have been aggregated by the upstream extractor, and (b) inference latency is a practical constraint in production deployments. The present study adopts their encoder unchanged (see `nids/models/contrastive_mlp.py`), which is a four-layer residual MLP with ReLU activations and a linear projection head.

Let $h_0 = x$ and let $h_\ell = \mathrm{DenseBlock}_\ell(h_{\ell-1})$ for $\ell \in \{1, 2, 3, 4\}$, where

$$
\mathrm{DenseBlock}_\ell(h) = \underbrace{\sigma\bigl(W_\ell h + b_\ell\bigr)}_{\text{linear + ReLU}} + \underbrace{R_\ell(h)}_{\text{residual}}.
$$

Here $R_\ell$ is the identity when the input and output dimensions match and a linear resizing otherwise, following the residual design of He et al. (2016). Stacking four DenseBlocks with hidden width 1024 yields the CLAN encoder. A projection head $P : \mathbb{R}^{1024} \to \mathbb{R}^{64}$ then produces the embedding $f_\theta(x) = P(h_4)$.

The input dimension $d$ is passed into `create_model(cfg, input_dim=splits.input_dim)` at runtime; the configuration file's `model.input_dim` field is treated only as an advisory value with an explicit logger warning emitted when the two disagree. This is the minimum hardening needed to make the encoder portable across datasets without silent shape-mismatch failures.

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

The augmented view $x_i^{aug}$ is produced by the upstream CLAN default `UniformResample` transform. The codebase keeps the other augmentation modules for tests and future ablations, but the final thesis configuration intentionally exposes only the two parameters used by the default transform (`augmentation.max_val` and `augmentation.p_feature`):

- **UniformResample** (paper default). For each feature $k$ selected by a Bernoulli($p_f$) mask, the value is replaced by a sample from $\mathcal{U}(-m_v, m_v) + \mu_v$.
- **GaussianResample.** The masked positions are replaced by samples from $\mathcal{N}(\mu_v, \sigma_v^2)$.
- **Jitter.** Gaussian noise from $\mathcal{N}(\mu_v, \sigma_v^2)$ is *added* to the masked positions rather than replacing them.
- **ZeroOutNoise.** Masked positions are set to zero.
- **FeatureShuffle.** Each masked position $k$ is filled with the value from a permuted feature $\pi(k)$.

All augmentations execute under `torch.no_grad()` to prevent gradient leakage. Both datasets use `UniformResample` with $p_f = 0.1$, $p_s = 1.0$, $m_v = 1.7$, $\mu_v = 0.0$ — the upstream CLAN default — so that the comparison in Chapter 4 isolates the dataset rather than the augmentation.

## 3.6 Anomaly Scoring and Few-Shot Fine-Tuning

**Anomaly detection score.** Following Wilkie et al. (2025), the benign centroid

$$
\mu = \frac{1}{|\mathcal{D}_B|} \sum_{x \in \mathcal{D}_B} f_\theta(x)
$$

is computed once on the training split and cached. At test time the score is $s(x) = -\cos(\mu, f_\theta(x))$. A higher score indicates a larger angular deviation from the benign manifold. The evaluation reports per-class one-vs-benign AUROC (§3.10) rather than a single threshold, preserving operating-point independence.

**Few-shot fine-tuning.** Given a pretrained encoder $f_\theta$ and a labelled subset $\mathcal{D}^{(K)}$, the present study attaches a linear head $g_\phi(z) = W z + b$ with $W \in \mathbb{R}^{(C+1) \times d'}$. Both $f_\theta$ and $g_\phi$ are trained jointly with cross-entropy loss for 100 epochs at $\text{lr} = 10^{-3}$, weight decay $10^{-6}$, batch size 64, and label smoothing 0.0. The paper-versus-code discrepancy on this learning rate — Wilkie et al. (2025, §V-A) report $10^{-6}$ whereas the upstream code default is $10^{-3}$ — is discussed in §3.11.1; this thesis adopts the code value and interprets the paper figure as a typographical error. Test metrics are macro-F1, macro-recall, macro-precision, and accuracy; §3.10 justifies the choice of macro-F1 as the primary headline metric.

## 3.7 Training, Optimisation, and Inference

**Optimiser.** AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay 0 during pretraining and $10^{-6}$ during fine-tuning, matching the upstream settings.

**Learning-rate schedule.** Warmup-cosine annealing (Loshchilov & Hutter, 2017), as implemented in `nids/training/schedules.py`: linear warmup from $10^{-6}$ to $10^{-4}$ over the first 10% of training steps, followed by cosine decay back to $10^{-6}$ over the remainder.

**Batching and mixed precision.** The upstream pretraining batch size is 8192. This thesis lowers it to 2048 to fit within the 6 GB VRAM of the RTX 3060 Laptop GPU on which all experiments run, and enables automatic mixed precision (`torch.cuda.amp.autocast` + `GradScaler`) to recover throughput; the resulting wall-clock is approximately 30–60 minutes per pretraining run on 3060. The same batch size is used on both datasets. A `WeightedRandomSampler` draws samples with weights proportional to inverse class frequency during fine-tuning, preventing minority-class collapse. The implementation lives in `nids/data/loaders.py::tabular_dl`.

**Seeding.** The utility `nids.utils.reproducibility.seed_everything(seed)` sets NumPy, PyTorch (CPU and CUDA), Python's `random` module, and the `PYTHONHASHSEED` environment variable, and enables `torch.backends.cudnn.deterministic = True` with `benchmark = False`. Three pretraining seeds $\{42, 43, 44\}$ are used for mean-and-standard-deviation reporting; the data split seed is fixed at 39 058 032 so that every seed operates on the same partition; fine-tune sample seeds $\{0, 1, \dots, 9\}$ are used for the ten-run averaging described in §3.10.

**Inference.** A single forward pass through $f_\theta$ followed by one dot-product with the cached centroid yields the anomaly score. The inference cost is $O(d')$ per query, substantially lighter than the memory-bank-based baselines of Chapter 2 (Wilkie et al., 2025).

## 3.8 Datasets: Lycos2017 and CICIDS2017

### 3.8.1 Lycos2017 (clean corpus)

Lycos2017 was released by Rosay et al. (2021, 2022) as the corrected version of CICIDS2017. It was produced by re-running a replacement feature extractor (*LycoSTand*) on the original CICIDS2017 packet captures and re-labelling each flow according to the corrected ground truth documented by Engelen et al. (2021). The present study uses the preprocessed single-CSV bundle distributed at `https://lycos-ids.univ-lemans.fr`, cached locally at `data/raw/lycos.csv`. The schema contains seven metadata columns (`flow_id`, `src_addr`, `src_port`, `dst_addr`, `dst_port`, `ip_prot`, `timestamp`) dropped before training, a `label` column, and the remaining CICFlowMeter-style flow statistics. The implementation in `nids/data/lycos.py` standardises features to zero mean and unit variance using statistics fit on the benign training split only, preventing leakage from attack statistics.

### 3.8.2 CICIDS2017 (noisy control corpus)

The CICIDS2017 corpus as distributed by the Canadian Institute for Cybersecurity (Sharafaldin et al., 2018) is used without any of the corrections published by Engelen et al. (2021), Rosay et al. (2022), Lanvin et al. (2023), or Liu et al. (2022). The loader in `nids/data/cicids.py` reads the eight day-split archives directly from `data/raw/lycos-ids2017/cicids2017/csv_files/`, normalises the well-known leading-space column names (e.g. `" Label"` becomes `label`), canonicalises the en-dash and mojibake variants of the Web Attack label values, replaces Inf with zero, and standardises features on the benign training split. Labels are otherwise preserved exactly as distributed. The four documented error categories — malformed TCP state machines (Engelen et al., 2021), time-window label leakage (Engelen et al., 2021), duplicated flows from dual termination paths (Rosay et al., 2022), and class-level rank flips (Lanvin et al., 2023) — are therefore inherited into the training and evaluation splits. This is deliberate: the purpose of running CLAN on CICIDS2017 is to measure how much those errors shift the headline metrics, not to silently repair them.

One additional CICIDS2017 defect surfaced during the port that is not cleanly catalogued in the prior-literature audits: the `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.zip` archive contains 288 602 rows whose label field is literally an empty string (neither a valid class name nor a documented placeholder). These rows are dropped by the loader with a logged warning — dropping rows with *missing* labels does not constitute cleaning of *wrong* labels and therefore does not weaken the noisy-label control argument. Rows with well-defined but erroneous labels (the classes of errors that Engelen et al., 2021 and Rosay et al., 2022 document) are retained unchanged. After this NaN-label drop, the CICIDS2017 corpus as consumed by the pipeline contains approximately 2.83 M flows spanning fourteen non-benign classes, of which three (Heartbleed, Infiltration, Web Attack — SQL Injection) fall below the `sample_threshold = 100` cut-off and are routed to the zero-day holdout exactly as on Lycos2017.

### 3.8.3 Shared Preprocessing Kernel

Both loaders delegate to `nids.data._core.prepare_splits`, which applies: (a) column drop by normalised name, (b) all-zero column drop (a step matching the upstream `get_data` in the CLAN reference code), (c) NaN/Inf clamping to zero, (d) stratified 50/50 train/test split with `split_seed = 39 058 032`, (e) zero-day carve-out for attack classes with fewer than 100 samples, (f) benign-only filter under `anomaly_detection=True`, and (g) training-split-only feature standardisation. The shared kernel design ensures that any difference in the Chapter 4 tables is attributable to the corpus rather than to the preprocessing.

### 3.8.4 Class Alignment Between Corpora

CICIDS2017 uses upper-case label values with whitespace and en-dashes (e.g. `Web Attack – Brute Force`) whereas Lycos2017 uses lower-case snake-case (`web_attack_brute_force`). The CICIDS2017 loader canonicalises to the Lycos2017 naming so that the Chapter 4 per-class tables align row by row; the three Web-Attack classes and the two FTP/SSH Patator classes map bijectively. The CICIDS2017 distribution contains two additional rare classes (Heartbleed, Infiltration) that receive zero-day treatment because both fall below the `sample_threshold = 100` cutoff.

## 3.9 Dual-Dataset Evaluation Design

The design of this study is the simplest controlled audit that can answer the research questions of §1.3: hold everything constant except the dataset. Concretely, the same `configs/lycos.yaml` and `configs/cicids.yaml` files share every field except `data.dataset`, `data.csv_path`, and `data.drop_cols` (which differ only because the two extractors produce different metadata column names). Pretraining, evaluation, and fine-tune sweep run under identical seeds, identical batch size, identical optimiser state, identical learning-rate schedule, identical augmentation, and identical model architecture.

The study therefore produces a four-way cross-tabulation: for each of the three pretraining seeds and each of the two datasets, one pretrained encoder, one centroid-based AUROC evaluation, and one full fine-tune sweep over eight shot counts with ten fine-tune seeds each. The total count is $2 \times 3 = 6$ pretraining runs and $2 \times 3 \times 8 \times 10 = 480$ fine-tune runs. The total wall-clock on RTX 3060 is estimated at approximately 17 hours.

This design trades breadth (no SSL-family comparison) for depth on a single scientific question: *how much does CLAN's headline number move when the underlying dataset shifts from clean to noisy, holding everything else constant?* The rationale for this trade is made explicit in §3.11.3.

## 3.10 Evaluation Protocol

### 3.10.1 Anomaly Detection Evaluation

After pretraining $f_\theta$, the benign centroid $\mu$ is computed on the training split. The test split, merged with the zero-day holdout, is embedded and the score $s(x) = -\cos(\mu, f_\theta(x))$ is computed for each sample. For each attack class $c$, the one-vs-benign AUROC is

$$
\mathrm{AUROC}_c = \Pr\bigl(s(x_a) > s(x_b)\bigr) \quad \text{for } x_a \sim \mathcal{D}_A^c, x_b \sim \mathcal{D}_B.
$$

The primary headline metric is **Mean AUROC** across the attack classes. A per-class breakdown is additionally reported to expose qualitative patterns and to power the ranking-stability analysis required by RQ3.

### 3.10.2 Few-Shot Multiclass Evaluation

The study sweeps $K \in \{8, 16, 32, 64, 128, 256, 512, 1024\}$. For each $K$:

1. A balanced subset with `num_benign = K` and `num_mal = K` is drawn from the training split (see `nids/data/utils.py::sample_data`).
2. The encoder and linear head are fine-tuned jointly for 100 epochs at $\text{lr} = 10^{-3}$, $\text{wd} = 10^{-6}$, $\text{batch size} = 64$.
3. Evaluation is performed on the full held-out test split excluding the fine-tune subset.

The primary metric is **macro-F1**. Following Engelen et al. (2021) and Lanvin et al. (2023), macro-F1 is preferred over accuracy or weighted-F1 because CICIDS2017's and Lycos2017's worst label errors concentrate in precisely the small classes to which macro-F1 is sensitive. Macro-recall, macro-precision, and accuracy are reported as secondary metrics.

Per the CLAN paper (Wilkie et al., 2025, §V-A), every few-shot macro-F1 number reported in Chapter 4 is the mean and standard deviation across **ten independent sample seeds**, where each seed produces an independently-drawn fine-tune subset. This is implemented by `scripts/finetune_sweep.py`.

### 3.10.3 Ranking Stability Statistic

For RQ3, the per-class AUROC ranking produced on each dataset is compared using Spearman's rank correlation $\rho$ and Kendall's $\tau$. A $\rho$ close to 1 indicates that CLAN's *ordering* of attack classes by difficulty is stable between the clean and noisy corpora; a $\rho$ substantially below 1 indicates that the label regime materially changes which attacks the model finds easy or hard.

### 3.10.4 Reporting

Every AUROC number in Chapter 4 is the mean and standard deviation across three pretraining seeds $\{42, 43, 44\}$. Every macro-F1 number is the mean and standard deviation across three pretraining seeds × ten fine-tune sample seeds ($n = 30$). The data split seed is held fixed at 39 058 032 so that each method pretrains on the same partition of each corpus.

## 3.11 Methodology Transparency

### 3.11.1 Paper-versus-Code Discrepancies Uncovered

During the port, two discrepancies between the CLAN paper (Wilkie et al., 2025) and the upstream Apache-2.0 reference implementation were identified and are documented here as *reproducibility findings* in the sense of MLRC (Reproducibility in Machine Learning Challenge).

First, the upstream repository does not include a `data/` subpackage despite every entry-point script importing from it (for example, `train_clan.py` line 10 imports `from data.load_data import get_data`, `from data.loaders import tabular_dl`, and `from data.utils import sample_data`). The three missing modules were reverse-engineered from their call-site signatures and from Rosay et al.'s (2022) documented preprocessing pipeline; they are re-implemented in `nids/data/lycos.py`, `nids/data/loaders.py`, and `nids/data/utils.py` with module docstrings that flag them explicitly as reconstructions.

Second, the paper (§V-A) states that fine-tuning uses a learning rate of $10^{-6}$, whereas the upstream code (`finetune_clan.py` line 42) sets the argparse default to $10^{-3}$ — a discrepancy of three orders of magnitude. Using $10^{-6}$ over 100 epochs yields essentially frozen weights, inconsistent with the paper's reported 8-shot macro-F1 of 0.496. This thesis therefore adopts the code value and interprets the paper figure as a typographical error. This interpretation is further supported by the fact that every other hyperparameter in the same paragraph (100 epochs, batch size 64) is consistent between paper and code; the learning rate is the sole outlier.

Both findings are logged in the code (`scripts/finetune.py` module docstring and the fine-tune defaults in `nids/config.py` / `configs/default.yaml`) so that a downstream reader can locate the provenance without re-reading the thesis.

### 3.11.2 Ablations Within the Available Compute Budget

The full ablation design that would mirror the upstream CLAN paper's 200-iteration random search over five hyperparameters is infeasible on a single RTX 3060. The thesis therefore restricts itself to one targeted ablation that can be run as a by-product of the dual-dataset protocol: the few-shot shot-count sweep $K \in \{8, 16, \dots, 1024\}$ is the natural shared axis between the two datasets and is reported in Chapter 4 as the primary ablation. All other CLAN hyperparameters (margin $m$, augmentation family, augmentation strength $p_f$, encoder depth) are held fixed at the paper defaults so that Chapter 4's interpretation of the Lycos2017-versus-CICIDS2017 shift is not confounded by simultaneous hyperparameter variation.

### 3.11.3 Honest Scope Declaration

Two scope reductions are made explicit here to forestall the most predictable examiner objections.

First, this thesis does not re-run the 200-iteration random-search + five-fold cross-validation protocol that Wilkie et al. (2025, §V-A) employ to choose hyperparameters. That protocol produces 1 000 complete pretraining runs per SSL method. On a single RTX 3060 this would require approximately 500 GPU-hours per method, which is infeasible. Instead this thesis adopts the hyperparameter values published in the upstream code and records the shared values in `nids/config.py` / `configs/default.yaml`; the Lycos2017 and CICIDS2017 YAML profiles differ only in dataset fields. This is a legitimate reproducibility shortcut — the authors' own code is the authoritative source for their hyperparameters — but it means that any suboptimal number reported on CICIDS2017 cannot be disentangled from the hypothesis "CICIDS2017 needs different hyperparameters than Lycos2017."

Second, this thesis does not compare CLAN to the seven SSL baselines listed in Wilkie et al. (2025, Tables I–III). The CLAN paper already provides that comparison on Lycos2017; re-running it on CICIDS2017 would require implementing and validating seven additional loss functions (SimCLR, Barlow Twins, BYOL, VICReg, SimSiam, ConFlow, SSCL-IDS), which sits outside the achievable scope of a single-student undergraduate project on commodity hardware. The present thesis therefore confines itself to the *single-method dual-dataset* audit that no one has yet published, on the grounds that depth on one new question is more valuable than breadth on a question that is already answered. Extending this audit to the seven SSL baselines is listed as future work in Chapter 5.

## 3.12 Implementation Details

### 3.12.1 Hardware and Software

Code is developed locally on macOS with numerically-pure unit tests executed via `pytest -q`. Training and evaluation run on Windows 11 with an NVIDIA RTX 3060 Laptop GPU (6 GB VRAM), CUDA 12.4, PyTorch 2.5+, and Python 3.10+. The package is installed in editable mode via `pip install -e .`; the full dependency list is maintained in `requirements.txt`.

### 3.12.2 Configuration

The YAML configuration is deliberately small after scope reduction. The dataset profiles `configs/lycos.yaml` and `configs/cicids.yaml` mostly specify the dataset name, raw-data path, and metadata columns to drop; shared CLAN defaults live in the frozen dataclasses in `nids.config` and in `configs/default.yaml`. The command-line entry points (`scripts/train.py`, `scripts/eval.py`, `scripts/finetune_sweep.py`) read the YAML via `nids.config.load_config`; training and evaluation normally require only `--config` and, when running multiple seeds, `--seed` or `--pretrain-seed`.

### 3.12.3 Artefact Layout

Each run writes to `artifacts/<dataset>/clan/seed<S>/`, producing the following files: the checkpoint `clan.pt.tar`, the resolved configuration `resolved_config.yaml`, the evaluation report `eval_report.json`, the fine-tune per-run JSON `finetune_shots<K>_seed<S>.json`, and the sweep summary CSV `finetune_summary.csv`. Raw data (`data/raw/`), preprocessed caches (`data/processed/`), and artefacts (`artifacts/`) are gitignored.

### 3.12.4 Reproducibility

Determinism is enforced at three levels: random seeding (see §3.7), dependency pinning (lower bounds in `requirements.txt` and exact versions on the Windows training rig), and code attribution (every ported module cites the upstream CLAN source at https://github.com/jackwilkie/CLAN, Apache-2.0). Unit tests in `tests/test_metrics.py`, `tests/test_lycos_loader.py`, `tests/test_cicids_loader.py`, `tests/test_contrastive_mlp.py`, `tests/test_clan_loss.py`, `tests/test_augmentations.py`, and `tests/test_distance.py` guard the numerical correctness of the metric, data-loader, encoder, loss, augmentation, and distance implementations. The full pytest suite completes in under one second on Apple Silicon and in under three seconds on the Windows training rig.
