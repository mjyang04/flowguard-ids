# CHAPTER 1

# INTRODUCTION

## 1.1 Motivation

Network intrusion detection systems (NIDS) sit on the critical path of essentially every modern enterprise, cloud tenant, and IoT deployment. The operational reality, however, keeps diverging from the textbook picture. Alsaedi et al. (2020) estimate that IoT traffic alone exceeds 75 ZB per year; Neto et al. (2023) report that the CICIoT2023 testbed records more than 48 M flows in a week from just 105 devices. Against this volume, the community's strongest baselines — from early autoencoders (Javaid et al., 2016; Shone et al., 2018) through CNN-LSTM hybrids (Hwang et al., 2019; Najar et al., 2025) to state-of-the-art Transformers (Manocchio et al., 2024; Han et al., 2023) — remain *supervised* classifiers, dependent on labelled attack traffic that is expensive to produce and poorly transferable between networks.

Three separate pressures have reshaped the research agenda over the last five years:

1. **Label scarcity.** Supervised NIDS need annotated attack flows that are either exceedingly rare in production (zero-day scenarios) or require a controlled testbed, which in turn biases the benchmark distribution. Golchin et al. (2024) demonstrate that this mismatch alone can swing AUROC by more than 20 percentage points when a supervised model trained on CICIDS2017 is applied to a neighbouring dataset.
2. **Dataset-integrity crisis.** The most cited benchmark of the deep-learning era, CICIDS2017 (Sharafaldin et al., 2018), has been shown by Engelen et al. (2021, WTMC), Rosay et al. (2021, 2022), Lanvin et al. (2023, CRiSIS) and Liu et al. (2022, IEEE CNS) to contain systematic labelling errors that can flip the ranking of competing methods. Any reproduction or comparison built on the original CICIDS2017 therefore inherits a known evaluation-noise floor.
3. **Contrastive self-supervised breakthroughs.** In the broader representation-learning literature, the SimCLR / MoCo / BYOL / SimSiam / Barlow Twins / VICReg family (Chen et al., 2020; He et al., 2020; Grill et al., 2020; Chen & He, 2021; Zbontar et al., 2021; Bardes et al., 2022) has shown that high-quality embeddings can be learned without labels. Their NIDS translations — Anomal-E (Caville et al., 2022), ConFlow (Liu et al., 2023), SSCL-IDS (Golchin et al., 2024), CLAN (Wilkie et al., 2025) — demonstrated that self-supervised pretraining on *benign-only* traffic can match or exceed supervised performance on several benchmarks.

These three pressures converge on a narrower research question than the one usually posed. The dominant benchmarking habit in the NIDS literature is to publish new methods on CICIDS2017 without re-examining whether the benchmark itself is trustworthy, despite a multi-year chain of integrity audits (Engelen et al., 2021; Rosay et al., 2021, 2022; Lanvin et al., 2023; Liu et al., 2022). Method rankings obtained on a label-noisy dataset may be artefacts of the noise rather than of the methods. This matters acutely for self-supervised NIDS: Wilkie et al. (2025) evaluate CLAN exclusively on Lycos2017 — Rosay et al.'s (2022) relabelled, feature-repaired descendant of CICIDS2017 — and report a strong mean AUROC of 0.959. Whether this headline number, or the method itself, survives exposure to the original noisy CICIDS2017 corpus has never been measured.

This thesis addresses that question directly.

## 1.2 Problem Statement

Let $\mathcal{X} = \{x_i\}_{i=1}^{N}$ be a corpus of network flows, each represented as a $d$-dimensional vector of statistical / header-level features. Let $\mathcal{X}_B \subset \mathcal{X}$ denote the *benign* subset and $\mathcal{X}_A = \mathcal{X} \setminus \mathcal{X}_B$ the attack subset. The practical self-supervised NIDS problem has two sub-problems:

1. **Anomaly detection.** Given only $\mathcal{X}_B$ at training time, learn an encoder $f_\theta : \mathbb{R}^d \to \mathbb{R}^{d'}$ and a scoring function $s : \mathbb{R}^{d'} \to \mathbb{R}$ such that $s(f_\theta(x))$ is *higher* for $x \in \mathcal{X}_A$ than for $x \in \mathcal{X}_B$. Deployment compares $s$ against a threshold.
2. **Few-shot multiclass attack classification.** Given the pretrained $f_\theta$ and a very small number of labelled samples per attack class ($K \in \{8, 16, 32, \dots, 1024\}$), fit a lightweight classification head $g_\phi : \mathbb{R}^{d'} \to \mathbb{R}^{|\mathcal{Y}|}$ on top of $f_\theta$ and report its macro-F1 on a held-out test set.

CLAN's proposal (Wilkie et al., 2025) is a specific answer: pretrain $f_\theta$ using a contrastive loss in which the positives are *other* benign samples in the batch and the negatives are *augmented* versions of the same samples, and use the cosine similarity to the centroid of $f_\theta(\mathcal{X}_B)$ as $s$. The headline claim — a mean AUROC of 0.959 and an 8-shot macro-F1 of 0.496 — rests entirely on evaluation on Lycos2017. Because the dataset choice is load-bearing for those numbers, the scientific value of the method depends on how stable those numbers are when the underlying corpus is swapped.

## 1.3 Research Questions

Rather than add yet another row to the CLAN-versus-other-SSL comparison table on a single dataset, this thesis studies **the single-method stability of CLAN across a clean (Lycos2017) and a noisy (CICIDS2017) corpus**, using identical hyperparameters, identical data pipelines, and identical evaluation protocol.

**RQ1.** Can CLAN's headline Lycos2017 numbers (mean AUROC ≈ 0.959, 8-shot macro-F1 ≈ 0.496 — Wilkie et al., 2025) be recovered under independent reproduction using the upstream Apache-2.0 code, the same training schedule, and three random seeds? Faithful reproduction is defined as a 95 % confidence interval around the paper's reported value.

**RQ2.** When CLAN is trained and evaluated, with identical pipeline and hyperparameters, on the original CICIDS2017 (Sharafaldin et al., 2018) — preserving the labelling errors documented by Engelen et al. (2021), Rosay et al. (2022), Lanvin et al. (2023), and Liu et al. (2022) — how much does the mean AUROC shift relative to Lycos2017? Does the shift fall within the 9–17 F1 point range that Engelen et al. (2021) reported for supervised classifiers on the same dataset pair?

**RQ3.** Does the *ranking of per-class AUROC* produced by CLAN remain stable between the two datasets, or does label noise re-order attack classes (for example, by collapsing the separation between *botnet* and *benign* when benign windows contain unlabelled C2 traffic as Engelen et al., 2021 documented)?

**RQ4.** Does the **few-shot fine-tune curve** — macro-F1 at $K \in \{8, 16, \dots, 1024\}$ labelled samples per class, averaged over ten fine-tune seeds — degrade on CICIDS2017 in a way consistent with the hypothesis that labelling errors depress recoverable class structure at low $K$?

## 1.4 Contributions

1. **Independent CLAN reproduction on Lycos2017.** To the best of our knowledge, this is the first externally-published reproduction of the full CLAN pipeline. The reproduction is faithful to the upstream Apache-2.0 implementation (Wilkie et al., 2025, https://github.com/jackwilkie/CLAN), with lightweight refactors for YAML-driven configuration and type safety.
2. **Reproducibility findings.** During the port, two paper-versus-code discrepancies were identified and documented: (a) a missing `data/` subpackage in the public repository that had to be reverse-engineered from call sites; (b) a three-order-of-magnitude discrepancy between the fine-tune learning rate stated in the paper (10⁻⁶) and the value in the reference code (10⁻³), with the latter being required to reproduce the reported few-shot macro-F1 numbers. These findings are discussed in Chapter 4.
3. **First controlled CLAN evaluation across Lycos2017 and CICIDS2017.** Using an identical encoder, identical augmentation pipeline, identical evaluation protocol, and three matched seeds, CLAN is run on both datasets. This is the first study to quantify the effect of the label-quality gap — previously characterised only for supervised classifiers (Engelen et al., 2021; Rosay et al., 2022; Lanvin et al., 2023) — on a self-supervised NIDS headline metric.
4. **Per-class stability analysis.** The per-attack AUROC and the ranking of attack classes between the two datasets are compared directly, providing evidence for or against the hypothesis that the noise in CICIDS2017 reshapes which attacks a self-supervised model finds easy or hard.
5. **Open, reproducible pipeline.** The full implementation — YAML configurations for both datasets, data loaders (including a CICIDS2017 loader that honours the raw CIC distribution without silent patching), model, training loop with automatic mixed precision for 6 GB VRAM, evaluation, and unit tests — is released under Apache-2.0. Seeds $\{42, 43, 44\}$ are used throughout; artefacts are versioned under `artifacts/<dataset>/clan/seed<S>/`.

## 1.5 Thesis Organisation

- **Chapter 2 — Literature Review** traces the relevant waves of NIDS research, the broader SSL / contrastive-learning literature that CLAN inherits from, the subset of that literature that has been adapted to NIDS, and — central to this thesis — the multi-year dataset-integrity debate that produced Lycos2017.
- **Chapter 3 — Research Methodology** gives a formal description of CLAN, the controlled dual-dataset evaluation protocol, the fine-tune averaging procedure, and the set of reproducibility safeguards adopted here.
- **Chapter 4 — Results and Discussion** reports the Lycos2017 reproduction, the CICIDS2017 control, the per-class stability analysis, and the few-shot curves, then interprets the magnitude and direction of the observed shifts against the prior literature on label noise in intrusion detection benchmarks.
- **Chapter 5 — Conclusion** summarises the contributions, acknowledges the specific limitations of the single-method scope adopted here, and outlines three directions for future work — extending the dual-dataset protocol to the seven SSL baselines that CLAN's original paper lists, implementing robust-to-noise variants of the CLAN objective, and running the same test on contemporary corpora such as CICIoT2023.
- **References** are listed at the end of this report; every citation in the thesis body resolves there with an author-year entry.
