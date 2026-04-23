# Chapter 1 — Introduction

## 1.1 Motivation

Network intrusion detection systems (NIDS) sit on the critical path of essentially every modern enterprise, cloud tenant, and IoT deployment. The operational reality, however, keeps diverging from the textbook picture. Alsaedi et al. (2020) estimate that IoT traffic alone exceeds 75 ZB per year; Neto et al. (2023) report that the CICIoT2023 testbed records more than 48 M flows in a week from just 105 devices. Against this volume, the community's strongest baselines — from early autoencoders (Javaid et al., 2016; Shone et al., 2018) through CNN-LSTM hybrids (Hwang et al., 2019; Najar et al., 2025) to state-of-the-art Transformers (Manocchio et al., 2024; Han et al., 2023) — remain *supervised* classifiers, dependent on labelled attack traffic that is expensive to produce and poorly transferable between networks.

Three separate pressures have reshaped the research agenda over the last five years:

1. **Label scarcity.** Supervised NIDS need annotated attack flows that are either exceedingly rare in production (zero-day scenarios) or require a controlled testbed, which in turn biases the benchmark distribution. Golchin et al. (2024) demonstrate that this mismatch alone can swing AUROC by more than 20 percentage points when a supervised model trained on CICIDS2017 is applied to a neighbouring dataset.
2. **Dataset-integrity crisis.** The most cited benchmark of the deep-learning era, CICIDS2017 (Sharafaldin et al., 2018), has been shown by Engelen et al. (2021, WTMC), Rosay et al. (2021, 2022), Lanvin et al. (2023, CRiSIS) and Liu et al. (2022, IEEE CNS) to contain systematic labelling errors that can flip the ranking of competing methods. Any reproduction or comparison built on the original CICIDS2017 therefore inherits a known evaluation-noise floor.
3. **Contrastive self-supervised breakthroughs.** In the broader representation-learning literature, the SimCLR / MoCo / BYOL / SimSiam / Barlow Twins / VICReg family (Chen et al., 2020; He et al., 2020; Grill et al., 2020; Chen & He, 2021; Zbontar et al., 2021; Bardes et al., 2022) has shown that high-quality embeddings can be learned without labels. Their NIDS translations — Anomal-E (Caville et al., 2022), ConFlow (Liu et al., 2023), SSCL-IDS (Golchin et al., 2024), CLAN (Wilkie et al., 2025) — demonstrated that self-supervised pretraining on *benign-only* traffic can match or exceed supervised performance on several benchmarks.

These three pressures converge on a specific research opportunity. **CLAN (Wilkie et al., 2025, IEEE CSR)** introduces a paradigm flip within contrastive NIDS: rather than treating augmented views as positives — the default since SimCLR — it treats them as *hard negatives*, driving benign samples toward a shared centroid while pushing distorted counterparts outwards. Wilkie et al. report a +0.031 mean AUROC gain over the next best method and a +0.056 gain over SSCL-IDS on the relabelled Lycos2017 dataset (Rosay et al., 2021). Yet no externally-published study has independently verified these numbers, and CLAN's sensitivity to its own design choices — the margin `m`, the augmentation family, the encoder depth, the few-shot schedule — has not been systematically ablated.

This thesis addresses those two gaps.

## 1.2 Problem Statement

Let $\mathcal{X} = \{x_i\}_{i=1}^{N}$ be a corpus of network flows, each represented as a $d$-dimensional vector of statistical / header-level features. Let $\mathcal{X}_B \subset \mathcal{X}$ denote the *benign* subset and $\mathcal{X}_A = \mathcal{X} \setminus \mathcal{X}_B$ the attack subset. The practical NIDS problem has two sub-problems:

1. **Anomaly detection.** Given only $\mathcal{X}_B$ at training time, learn an encoder $f_\theta : \mathbb{R}^d \to \mathbb{R}^{d'}$ and a scoring function $s : \mathbb{R}^{d'} \to \mathbb{R}$ such that $s(f_\theta(x))$ is *higher* for $x \in \mathcal{X}_A$ than for $x \in \mathcal{X}_B$. Deployment compares $s$ against a threshold.
2. **Few-shot multiclass attack classification.** Given the pretrained $f_\theta$ and a very small number of labelled samples per attack class ($K \in \{8, 16, 32, \dots, 1024\}$), fit a lightweight classification head $g_\phi : \mathbb{R}^{d'} \to \mathbb{R}^{|\mathcal{Y}|}$ on top of $f_\theta$ and report its macro-F1 on a held-out test set.

CLAN's proposal (Wilkie et al., 2025) is a specific answer: pretrain $f_\theta$ using a contrastive loss in which the positives are *other* benign samples in the batch and the negatives are *augmented* versions of the same samples, and use the cosine similarity to the centroid of $f_\theta(\mathcal{X}_B)$ as $s$. The claim this thesis tests is that this answer outperforms seven baseline SSL losses under matched encoder, augmentation, and evaluation protocol.

## 1.3 Research Questions

**RQ1.** Does a faithful reproduction of CLAN on Lycos2017 recover the headline numbers reported by Wilkie et al. (2025) — namely a mean AUROC of approximately 0.959 and an 8-shot macro-F1 of approximately 0.496 — within the noise floor of three random seeds?

**RQ2.** Under a shared encoder and a shared augmentation policy, does CLAN's augmented-as-negative objective strictly outperform each of seven mainstream SSL losses (SimCLR, Barlow Twins, BYOL, VICReg, SimSiam, ConFlow, SSCL-IDS) on mean AUROC and on the few-shot macro-F1 curve?

**RQ3.** How sensitive is CLAN to its key design choices? Specifically, how do (a) margin `m`, (b) augmentation family, (c) augmentation strength (`p_feature`), (d) encoder depth, (e) L2-normalisation of embeddings, and (f) the number of labelled fine-tune samples affect downstream performance?

**RQ4.** Is Lycos2017 — the corrected, re-labelled version of CICIDS2017 released by Rosay et al. (2021) — a more robust evaluation target than the original CICIDS2017, in the sense that method rankings obtained on Lycos2017 are reproducible across seeds and align with the errors documented by Engelen et al. (2021) and Lanvin et al. (2023)?

## 1.4 Contributions

1. **Independent CLAN reproduction.** To the best of our knowledge, this is the first externally-published reproduction of the CLAN pipeline. The reproduction faithfully follows the upstream implementation (Wilkie et al., 2025, Apache-2.0 code at https://github.com/jackwilkie/CLAN), with lightweight refactors for YAML-driven configuration, type safety, and licence-preserving attribution. See Chapter 3 for the algorithmic description and Chapter 4 for reproducibility details.
2. **Controlled seven-baseline SSL comparison.** All comparison methods run under a shared encoder (ContrastiveMLP), shared augmentation library, shared Lycos2017 split, and shared evaluation protocol, isolating the contribution of the loss function itself. This removes a well-documented confound that plagues cross-paper NIDS comparisons (Engelen et al., 2021; Lanvin et al., 2023; Liu et al., 2022).
3. **Structured ablation.** Chapter 4 defines six ablation axes — margin, augmentation family, augmentation strength, encoder depth, L2 normalisation, shot count — and Chapter 5 reports the results. This is the first externally-published study of CLAN's parameter sensitivity.
4. **Dataset-integrity grounded evaluation.** By pretraining and evaluating on Lycos2017 rather than the original CICIDS2017, the thesis sidesteps the label-noise floor documented by Engelen et al. (2021), Rosay et al. (2021, 2022), Lanvin et al. (2023), and Liu et al. (2022). Chapter 4 makes the integrity argument explicit and includes a side-by-side CICIDS2017 audit for context.
5. **Open, reproducible pipeline.** The full implementation — configuration, data pipeline, model, training loop, evaluation, and tests — is released under Apache-2.0 and uses a single YAML file (`configs/default.yaml`) to expose every experimental knob. Seeds `{42, 43, 44}` are used throughout; artifacts and checkpoints are versioned under `artifacts/<loss_name>/<timestamp>_seed<S>/`.

## 1.5 Thesis Organisation

- **Chapter 2 — Related Work** traces the five waves of deep-learning NIDS, the broader SSL / contrastive-learning literature, the subset of that literature that has been adapted to NIDS, and the dataset-integrity debate that motivates Lycos2017.
- **Chapter 3 — Method** gives a formal description of CLAN: the ContrastiveMLP encoder, the CLANLoss objective, the augmentation family, the centroid-based anomaly score, and the fine-tune protocol.
- **Chapter 4 — Experimental Setup** details the Lycos2017 split, the seven baseline SSL losses, the shared evaluation protocol, the six ablation axes, and the reproducibility conventions (seeds, artifacts, licences).
- **Chapter 5 — Results** reports the headline comparison table, the per-class AUROC breakdown, the ablation grids, and the few-shot multiclass curves.
- **Chapter 6 — Discussion** frames the results against the research questions, examines failure modes, and identifies limitations of the Lycos2017 corpus that remain unresolved.
- **Chapter 7 — Conclusion** summarises the contributions and outlines four concrete directions for future work.
- **Bibliography** is maintained in `references.md`; every citation in this thesis resolves there with an author-year key.
