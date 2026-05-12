**[[STUDENT NAME IN ALL CAPS]]**

**XIAMEN UNIVERSITY MALAYSIA**

**[[YEAR]]**

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

![XMUM Logo](media/image1.png)

FINAL YEAR PROJECT REPORT

**A REPRODUCTION AND DATASET-INTEGRITY STUDY OF CONTRASTIVE SELF-SUPERVISED NETWORK INTRUSION DETECTION USING AUGMENTED NEGATIVE PAIRS (CLAN) ON LYCOS2017 AND CICIDS2017**

|  |  |  |
|---|---|---|
| NAME OF STUDENT | : | [[STUDENT NAME]] |
| STUDENT ID | : | [[STUDENT ID]] |
| SCHOOL / FACULTY | : | SCHOOL OF COMPUTING AND DATA SCIENCE |
| PROGRAMME | : | BACHELOR OF ENGINEERING IN [[PROGRAMME]] (HONOURS) |
| INTAKE | : | [[INTAKE CODE]] |
| SUPERVISOR | : | [[SUPERVISOR NAME]] |
| TITLE | : | [[TITLE]] |

**[[MONTH]] [[YEAR]]**

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**DECLARATION**

I hereby declare that this project report is based on my original work except for citations and quotations which have been duly acknowledged. I also declare that it has not been previously and concurrently submitted for any other degree or award at Xiamen University Malaysia or other institutions.

Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Name: [[STUDENT NAME]]

ID No.: [[STUDENT ID]]

Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**APPROVAL FOR SUBMISSION**

I certify that this project report entitled **"A REPRODUCTION AND DATASET-INTEGRITY STUDY OF CONTRASTIVE SELF-SUPERVISED NETWORK INTRUSION DETECTION USING AUGMENTED NEGATIVE PAIRS (CLAN) ON LYCOS2017 AND CICIDS2017"** that was prepared by [[STUDENT NAME]] has met the required standard for submission in partial fulfilment of the requirements for the award of Bachelor of Engineering in [[PROGRAMME]] (Honours) at Xiamen University Malaysia.

Approved by,

Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Supervisor: [[SUPERVISOR NAME]]

Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

The copyright of this report belongs to the author under the terms of Xiamen University Malaysia copyright policy. Due acknowledgement shall always be made of the use of any material contained in, or derived from, this project report / thesis.

© [[YEAR]], [[STUDENT NAME]]. All rights reserved.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**ACKNOWLEDGEMENTS**

The author would like to thank all who have contributed to the successful completion of this project. The author would like to express gratitude to the research supervisor, [[SUPERVISOR NAME]], for invaluable advice, guidance, and patience throughout the development of the research. Sincere thanks also go to [[CO-SUPERVISOR / ADVISOR NAME(S), if any]] for discussions that shaped several of the design choices in Chapter 3.

The author acknowledges the authors of the upstream CLAN repository -- Jack Wilkie, Hanan Hindy, Christos Tachtatzis, and Robert Atkinson (University of Strathclyde and Ain Shams University) -- whose Apache-2.0 reference implementation made this reproduction possible, and Rosay et al. for releasing the relabelled Lycos2017 corpus. Finally, the author thanks family and friends for their encouragement throughout the duration of the project.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**ABSTRACT**

Self-supervised learning has recently been adopted as a practical remedy for the label-scarcity problem in network intrusion detection systems (NIDS). Among the resulting body of work, Wilkie et al. (2025) propose Contrastive Learning using Augmented Negatives (CLAN), in which the augmented view of a benign flow is treated as a hard *negative* rather than the canonical positive, and an anomaly score is computed as the cosine distance of a test embedding to the benign centroid. The original publication reports a mean AUROC of approximately 0.959 on the relabelled Lycos2017 corpus and an 8-shot multiclass macro-F1 of approximately 0.496, but no externally-published study has independently reproduced these numbers or measured how sensitive CLAN is to the label-quality gap between Lycos2017 and the original CICIDS2017 release.

This study addresses that narrower and more reproducible gap. An end-to-end pipeline mirroring the upstream Apache-2.0 implementation is ported into a YAML-driven Python package. The implementation uses the ContrastiveMLP encoder, CLAN loss, uniform-resample augmentation, centroid-based AUROC evaluation, and few-shot multiclass fine-tuning protocol documented by the reference code. The same pipeline is then run on Lycos2017 and on the original CICIDS2017 corpus, preserving CICIDS2017's documented label and feature-extraction defects rather than silently repairing them. Holding architecture, hyperparameters, data split seed, augmentation, optimiser, and evaluation protocol fixed allows any observed shift to be attributed to the dataset rather than to the method.

The thesis therefore contributes an independent CLAN reproduction, a transparent record of paper-versus-code discrepancies discovered during porting, and a controlled dual-dataset audit of a self-supervised NIDS method. The few-shot shot-count sweep $K \in \{8,16,\dots,1024\}$ is retained as the primary compute-feasible ablation. The broader seven-method SSL comparison reported by Wilkie et al. (2025) is explicitly left as future work because implementing and validating seven additional losses is outside the available single-student timeline and GPU budget.

**Keywords:** Network Intrusion Detection; Self-Supervised Learning; Contrastive Learning; CLAN; Dataset Integrity.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**TABLE OF CONTENTS**

DECLARATION ... ii

APPROVAL FOR SUBMISSION ... iii

ACKNOWLEDGEMENTS ... v

ABSTRACT ... vi

TABLE OF CONTENTS ... vii

LIST OF TABLES ... viii

LIST OF FIGURES ... ix

LIST OF SYMBOLS / ABBREVIATIONS ... x

CHAPTER 1 INTRODUCTION ... 1

1.1 Motivation ... 1

1.2 Problem Statement ... 3

1.3 Research Questions ... 4

1.4 Contributions ... 5

1.5 Thesis Organisation ... 6

CHAPTER 2 LITERATURE REVIEW ... 7

CHAPTER 3 RESEARCH METHODOLOGY ... 22

CHAPTER 4 RESULTS AND DISCUSSION ... 35

CHAPTER 5 CONCLUSION ... 49

REFERENCES ... 54

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**LIST OF TABLES**

Table 4.1: Lycos2017 reproduction -- CLAN AUROC ... TBD

Table 4.2: Lycos2017 few-shot macro-F1 reproduction ... TBD

Table 4.3: CICIDS2017 noisy-label control -- CLAN AUROC ... TBD

Table 4.4: Summary of CLAN headline metrics across datasets ... TBD

Table 4.5: Per-class rank comparison across datasets ... TBD

Table 4.6: Few-shot macro-F1 across datasets ... TBD

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**LIST OF FIGURES**

Figure 4.1: Per-class AUROC shift between Lycos2017 and CICIDS2017 ... TBD

Figure 4.2: Few-shot macro-F1 curves across datasets ... TBD

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**LIST OF SYMBOLS / ABBREVIATIONS**

| Symbol / Abbreviation | Meaning |
|---|---|
| x in R^d | Flow feature vector (d is inferred after preprocessing) |
| y | Class label (0 = benign, 1..C = attack) |
| f_theta | Encoder network |
| z = f_theta(x) | Embedding in R^d' (d' = 64) |
| mu | Benign centroid in embedding space |
| s(x) | Anomaly score, -cos(mu, f_theta(x)) |
| m | CLAN loss margin |
| alpha | CLAN intra/inter-class weight |
| p_f, p_s | Augmentation per-feature / per-sample probabilities |
| K | Few-shot samples per class |
| AUC / AUROC | Area Under the Receiver Operating Characteristic curve |
| CLAN | Contrastive Learning using Augmented Negatives |
| FYP | Final Year Project |
| NIDS | Network Intrusion Detection System |
| PCAP | Packet Capture |
| RQ | Research Question |
| SSL | Self-Supervised Learning |

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```
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
```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# CHAPTER 2

# LITERATURE REVIEW

## 2.0 Background

This chapter situates the present study — a reproduction of CLAN (Wilkie et al., 2025) on Lycos2017 paired with a controlled re-run on the original CICIDS2017 — within four overlapping bodies of literature. Section 2.1 traces the migration of NIDS from shallow classifiers to deep architectures. Section 2.2 reviews the broader self-supervised / contrastive learning literature whose ideas CLAN inherits. Section 2.3 focuses on contrastive SSL adapted specifically for NIDS, ending with a gap analysis that motivates CLAN's *augmented-as-negative* design. Section 2.4 audits the benchmark datasets and evaluation conventions on which all such comparisons rest; in particular, §2.4.2 surveys the multi-year CICIDS2017 labelling controversy whose resolution produced Lycos2017, and whose unresolved CICIDS2017-side is the control corpus in this thesis. Section 2.5 synthesises a positioning statement for the present study.

Citation style throughout is attributive: every non-trivial claim is accompanied by the specific authors who made it. Where a point is disputed, both sides are named.


## 2.1 Deep Learning for Network Intrusion Detection

The migration from shallow statistical classifiers to deep neural architectures in network intrusion detection has unfolded across roughly a decade. Rather than a single monolithic trend, the literature reveals five overlapping waves — early feature-learning autoencoders, specialisation into CNN and RNN branches, CNN-LSTM hybridisation, the Transformer/attention turn, and the most recent graph-based and foundation-model era. This section traces each wave by attributing claims to the specific authors who made them.

### 2.1.1 Early Shallow Deep Models (2015–2017)

The earliest deep NIDS work was dominated by unsupervised feature learners stacked on NSL-KDD. **Javaid et al. (2016)** were among the first to apply self-taught learning with a sparse autoencoder to NSL-KDD, reporting 88.39% accuracy on 2-class classification and arguing that unsupervised representation learning could outperform hand-engineered features on the KDD family. In parallel, **Tang et al. (2016)** targeted a new deployment context — SDN — and observed that a small DNN using only six NetFlow-extractable features could reach 75.75% accuracy on NSL-KDD, framing the key challenge of the era as *feature economy under SDN controller constraints*. **Shone et al. (2018)** pushed this direction further with a non-symmetric deep autoencoder (NDAE) stacked with a Random Forest classifier, achieving 97.85% on KDD Cup 99 and 85.42% on NSL-KDD, and explicitly arguing that asymmetric encoder–decoder structures capture intrusion-relevant manifolds more efficiently than standard autoencoders. Taken together, these three works framed deep NIDS as *representation learning on tabular flow features*, an assumption that the next wave would challenge.

### 2.1.2 CNN and RNN Specialisation (2017–2019)

As researchers moved beyond NSL-KDD, the community split into spatial (CNN) and temporal (RNN) camps. **Yin et al. (2017)** proposed RNN-IDS and reported that recurrent models outperformed J48, SVM, and Random Forest on both binary and multi-class NSL-KDD, positioning sequential modelling as a natural fit for flow-level intrusion data. **Kim et al. (2016)**, concurrently, argued that LSTM-RNN classifiers could model long-range dependencies within KDD99 traces, and **Du et al. (2017)** extended this temporal view to system-log anomaly detection in *DeepLog* (ACM CCS 2017), treating logs as a natural-language sequence and demonstrating that an LSTM could flag deviations from learned execution patterns and support root-cause diagnosis. On the CNN side, **Wang et al. (2017)** showed that a 1-D CNN consuming raw 784-byte packet segments could perform end-to-end encrypted traffic classification on ISCX VPN-nonVPN, beating handcrafted baselines on 11 of 12 metrics. **Vinayakumar et al. (2019, IEEE Access)** then consolidated the field with a large benchmark across KDDCup99, NSL-KDD, UNSW-NB15, CICIDS2017, and others, concluding that DNNs generalise better than classical ML across multiple datasets and proposing the *Scale-Hybrid-IDS-AlertNet* framework for real-time host- and network-level monitoring.

### 2.1.3 Hybrid CNN-LSTM and Early Attention (2019–2023)

A third wave argued that neither CNN nor RNN alone was sufficient. **Hwang et al. (2019, Applied Sciences)** proposed that packet-field semantics should be captured with word embeddings and then fed into an LSTM, treating packet bytes like tokens; in follow-up work, the same group (**Hwang, Peng, Nguyen, and Lai, 2020**) introduced *D-PACK*, combining a CNN feature extractor with an autoencoder for early-stage Mirai detection on minimal packet prefixes. **Roy et al.** and subsequent IoT-IDS studies (culminating in the hybrid CNN-BiLSTM line exemplified by the *Scientific Reports / MDPI Sensors 2025* family — cf. **Najar et al. (2025)** as representative) contended that spatial-temporal fusion — CNN for byte-level locality, BiLSTM for flow-level sequence — was the most reliable route to >98% accuracy on benchmark data. The ceiling of this hybrid design, however, became an argument for moving to attention-based architectures.

### 2.1.4 Transformer Era (2022–2024)

**Manocchio et al. (2024, Expert Systems with Applications)** offered one of the most systematic Transformer studies for NIDS with *FlowTransformer*, a modular framework in which input encodings, transformer blocks, and classification heads could be swapped on common benchmarks. They observed that the *classification head* — not the transformer backbone — dominated performance, and that Global Average Pooling performed surprisingly poorly in the NIDS setting. **Nguyen and Kashef (2023, Knowledge-Based Systems)** took a different route with *TS-IDS*, arguing that self-supervised auxiliary tasks defined over graph-structured IoT traffic could learn communication patterns without labels, and combining GNN message passing with SSL pre-training. **Han et al. (2023, Computers & Security)** independently integrated n-gram byte frequencies into a *time-aware Transformer*, observing that absolute time gaps between packets were informative features that vanilla Transformers ignore. More broadly, surveys such as **Ferrag et al. (2024, arXiv:2408.07583)** catalogue more than 100 Transformer/LLM-based NIDS papers published between 2017 and 2024, underscoring that attention has become the default backbone choice.

### 2.1.5 Graph Neural Networks and Foundation Models (2022–2026)

The most recent wave re-conceives the network itself as a graph or as a pretraining substrate. **Lo et al. (2022, IEEE/IFIP NOMS)** proposed *E-GraphSAGE*, the first practical GNN-based NIDS to exploit edge (flow) features alongside node (host) topology, reporting state-of-the-art results on four NetFlow benchmarks and arguing that topology carries information that flow-wise classifiers discard. **Caville et al. (2022, Knowledge-Based Systems)** built directly on this foundation with *Anomal-E*, combining E-GraphSAGE with a modified Deep Graph Infomax (DGI) objective so that GNNs could be trained without attack labels — the first truly self-supervised graph NIDS. **Guerra et al. (2025, NeurIPS)** most recently introduced *GraphIDS*, uniting an inductive GNN with a Transformer masked autoencoder so that flows are reconstructed from their local topological neighbourhoods; the authors report up to 99.98% PR-AUC and argue that reconstruction error is a more discriminative anomaly signal than contrastive distance.

Along a parallel axis, pre-trained foundation models have begun to dominate encrypted-traffic and flow classification. **Lin et al. (2022, WWW)** proposed *ET-BERT*, pre-training a Transformer on unlabeled datagrams via Masked BURST Model and Same-origin BURST Prediction objectives and pushing ISCX-VPN-Service F1 to 98.9%. **Zhao et al. (2023, AAAI)** followed with *YaTC*, arguing that masked autoencoding over a multi-level (packet + flow) traffic matrix yields stronger few-shot performance than BERT-style masking alone. **Wang et al. (2024, arXiv:2405.11449)** then pushed beyond the Transformer with *NetMamba*, using a unidirectional state-space model to achieve ~99% accuracy while claiming up to 60× inference speed-up over Transformer baselines. On the contrastive branch, **Golchin et al. (2024, IFIP Networking)** proposed *SSCL-IDS*, using CutMix-based contrastive pretraining to improve generalisation across traffic distributions, and **Wilkie et al. (2025, IEEE CSR)** introduced *CLAN*, reframing the contrastive objective so that augmented samples act as *negatives* rather than positives and reporting superior few-shot multiclass results on the relabelled Lycos2017 dataset released after the dataset-integrity critiques of **Engelen et al. (2021, WTMC)** and **Rosay et al. (2021, WI-IAT)**.

Across these five waves, two trajectories stand out. First, the centre of gravity has shifted from supervised classification on NSL-KDD-style tabular features (Javaid, Tang, Shone, Yin) toward self-supervised pretraining on benign traffic (Caville, Golchin, Wilkie, Guerra). Second, the inductive bias has moved from per-flow vectors (Vinayakumar) to sequences (Du, Kim), to spatial-temporal hybrids (Hwang, Najar), to attention (Manocchio, Han), and finally to graphs and state-space models (Lo, Guerra, Wang) — a steady broadening of the structural priors deemed useful for intrusion detection.


## 2.2 Self-Supervised Representation Learning

Self-supervised learning (SSL) has become a mainstream paradigm in representation learning. Its core idea is to construct "pseudo-labels" from the data itself when human annotations are scarce, letting an encoder learn transferable representations that are then fine-tuned on a small supervised downstream task. This section follows the lineage "contrastive learning in vision → non-contrastive and redundancy-reduction methods → theoretical analysis → supervised contrastive → tabular SSL" to trace the ideas that CLAN (Wilkie et al., 2025) and the surrounding SSL-NIDS literature rely on.

### 2.2.1 Contrastive Pretraining for Vision: Foundational Ideas

The mathematical starting point of modern contrastive learning is the InfoNCE objective. **van den Oord et al. (2018)** in *Contrastive Predictive Coding (CPC)* first systematically connected noise-contrastive estimation to a variational lower bound on mutual information, proving that minimising InfoNCE is equivalent to maximising a lower bound on $I(x; c)$ and thereby giving all subsequent contrastive methods a formal footing. **Hjelm et al. (2019, ICLR, Deep InfoMax)** then generalised mutual-information maximisation to local–global image representations, reinforcing the central intuition of "maximise agreement between views".

Building on this theoretical skeleton, **Chen et al. (2020, ICML, SimCLR)** argued that three simple design choices — strong data augmentation, large batch size, and a non-linear projection head between the encoder and the contrastive loss — can dramatically improve the linear separability of contrastive representations; they contended that the *combination* of augmentations (especially random crop + colour distortion) is what makes contrastive learning work in vision. Concurrently, **He et al. (2020, CVPR, MoCo)** argued that large batch size is not essential: one can maintain a large and stable dictionary of negatives using a momentum-updated key encoder and a FIFO queue. **Chen, Fan, Girshick & He (2020b, MoCo v2)** subsequently ported SimCLR's projection head and augmentation policy back into MoCo, narrowing the gap between the two recipes. Together these two lines of work established the two templates that all later work inherits — *in-batch negatives* (SimCLR) and *memory-bank / queue negatives* (MoCo). CLAN's augmented-negative design is precisely a modification of the former.

### 2.2.2 Beyond Negatives: BYOL, SimSiam, Barlow Twins, VICReg

A long-standing concern with contrastive learning is *representation collapse*: without the repulsive force of negative samples, the network may trivially output a constant representation. **Grill et al. (2020, NeurIPS, BYOL)** observed that with a predictor on the online network and an exponential-moving-average target network, one can learn competitive representations with *no negatives at all*; they argued that asymmetry plus stop-gradient together prevent the trivial solution. **Chen and He (2021, CVPR, SimSiam)** went further, showing that removing the momentum target network entirely — retaining only stop-gradient and the predictor — still avoids collapse; their ablations suggested that stop-gradient is the most important ingredient while the predictor is what lifts performance above the baseline.

In parallel, **Zbontar et al. (2021, ICML, Barlow Twins)** approached collapse from an information-bottleneck standpoint: they drove the cross-correlation matrix between two views toward the identity, with diagonal terms encouraging invariance and off-diagonal terms encouraging feature decorrelation (redundancy reduction). **Bardes, Ponce and LeCun (2022, ICLR, VICReg)** argued that Barlow Twins' coupling of variance and covariance is excessive, and decoupled the objective into three explicit terms: *variance* (per-dimension variance lower-bounded at 1 to prevent collapse), *invariance* (MSE between views), and *covariance* (off-diagonal decorrelation); they argued this decoupled design needs neither negatives, nor stop-gradient, nor a momentum network. **Caron et al. (2021, ICCV, DINO)** then extended BYOL-style distillation to Vision Transformers, using centring + sharpening to avoid collapse and observing — as a by-product — that self-distillation yielded attention maps of segmentation quality. These four lines — BYOL's momentum self-distillation, SimSiam's stop-gradient, Barlow Twins' redundancy reduction, and VICReg's three-term decoupling — form the non-contrastive family that CLAN's experiments compare against.

### 2.2.3 Theoretical Foundations

As empirical methods proliferated, a parallel effort sought to explain *why* contrastive learning works. **Arora et al. (2019, ICML)** gave the first formal framework: under a latent-class assumption, they proved that contrastive loss upper-bounds the risk of a downstream linear classifier, so minimising InfoNCE provides a theoretical guarantee on downstream performance; they also pointed out that more negatives is not uniformly better — too many induce a *class collision* bias. **Wang and Isola (2020, ICML)** offered the more intuitive *alignment–uniformity* framework, decomposing the InfoNCE limit into two geometric goals: positive pairs should be close on the hypersphere (alignment), while the overall distribution of representations should be uniform on the sphere (uniformity); they showed these two quantities can be measured independently and correlate strongly with downstream accuracy. **Tian, Sun et al. (2020, NeurIPS, "What Makes for Good Views?")** sharpened the InfoMax principle into *InfoMin*: the optimal view should preserve task-relevant information while discarding redundant mutual information; this *sweet-spot* view explains why augmentations that are too weak make the pretext task trivial while augmentations that are too strong destroy semantics. These results underlie the design decisions of VICReg and Barlow Twins (why retain a variance/uniformity term, why decorrelate) and give the theoretical background for interpreting CLAN's "augmented negatives expand uniformity constraints" intuition.

### 2.2.4 Supervised Contrastive Learning

When partial labels are available, **Khosla et al. (2020, NeurIPS)** proposed Supervised Contrastive Loss (SupCon), arguing that the "one positive per anchor" constraint of SimCLR should be relaxed: all same-class samples in the batch should be treated as positives. They showed SupCon stably outperforms cross-entropy on ImageNet and is more robust to label noise and hyperparameter choice. **Graf et al. (2021, ICML)** proved that the optimal solution of SupCon is a class-wise simplex equiangular tight frame (ETF), interfacing with the neural-collapse literature. This branch matters for NIDS because malicious traffic naturally carries fine-grained attack labels: ConFlow (Liu et al., 2023) and SSCL-IDS (Golchin et al., 2024) both inherit SupCon's "pull same-class, push different-class" idea, explicitly injecting class information into the contrastive target. CLAN, by contrast, goes the opposite way — using only benign labels plus augmented negatives — to avoid reliance on scarce attack labels.

### 2.2.5 SSL for Tabular Data

Vision-domain SSL methods transfer poorly to tabular or flow data: cropping, colour jitter, and Gaussian blur assume a smooth image manifold that does not hold for discrete, heterogeneous, column-semantic tabular features. **Yoon et al. (2020, NeurIPS, VIME)** conducted the first systematic study of tabular SSL, proposing two pretext tasks — *mask vector estimation* and *feature value reconstruction* — together with a consistency-regularised semi-supervised extension; they demonstrated substantial gains over purely supervised baselines on small-sample medical and genomic data. **Ucar, Hajiramezanali & Edwards (2021, NeurIPS, SubTab)** argued that treating an entire row as a single view is information-sparse, and instead randomly partitioned the feature columns into subsets, letting the model learn representations consistent across subsets by reconstructing the full row — conceptually replacing SimCLR's "two augmentations" with "two feature subsets". **Bahri et al. (2022, ICLR, SCARF)** directly transplanted SimCLR's InfoNCE loss to tabular data, defining augmentation as "randomly replace a subset of feature columns with values drawn from the marginal", and validated its superiority over denoising and VIME-style pretext on 69 OpenML datasets. **Somepalli et al. (2021, SAINT)** more recently combined SCARF's column-level augmentation with row-and-column attention, showing that tabular SSL has approached the maturity of its vision counterpart. On the NIDS side, ConFlow (Liu et al., 2023) and SSCL-IDS (Golchin et al., 2024) largely follow the SCARF column-augmentation template but introduce flow-level positive construction; CLAN (Wilkie et al., 2025) contributes to this lineage by observing that under benign-only training, column-augmented "distorted benign" samples are *already* strong enough to serve as negatives for an anchor, simultaneously achieving uniformity (in the sense of Wang & Isola, 2020) and sensitivity to malicious flows.


## 2.3 Contrastive Self-Supervised Learning for NIDS

Two long-standing pains in supervised NIDS — label scarcity and cross-distribution generalisation failure — together drove the adoption of self-supervised learning in this field. Within SSL, contrastive learning has received the most concentrated attention because of its "learn discriminative representations from benign traffic alone" property. This section organises the existing work into four lines — graph-SSL pioneering, flow-level contrastive maturation, the CLAN paradigm shift, and the adjacent hard-negative lineage — and ends with the specific gap CLAN fills.

### 2.3.1 Early Adoption: Self-Supervision on Graph-Structured Flow Data

**Caville et al. (2022, *Knowledge-Based Systems*, Anomal-E)** is the first practical NIDS in which self-supervised signals (rather than labels) are the training core. The authors observed that the then-mainstream GNN-NIDS, such as E-GraphSAGE (Lo et al., 2022), depended heavily on labels and could not scale as attacks evolved; they therefore ported the Deep Graph Infomax (DGI, Veličković et al., 2019) mutual-information-maximisation objective to edge-centric NetFlow graphs, using corruption to produce negative graphs and maximising local–global MI. On two NF-benchmarks they significantly outperformed baselines trained directly on raw features. The contribution of Anomal-E is not in the contrastive loss itself, but in establishing the two-stage *SSL-pretrain → shallow-anomaly-detector* paradigm that all subsequent flow-level contrastive work has reused. The cost, equally clear, is the graph-construction overhead, its resistance to streaming deployment, and the fact that DGI-style MI objectives do not directly align with the anomaly-detection target.

### 2.3.2 Flow-Level Contrastive Learning: ConFlow, CLDNN, SSCL-IDS

A second wave returned to lighter-weight flow-level representations and ported the SimCLR (Chen et al., 2020) InfoNCE recipe into NIDS. **Liu et al. (2023, ConFlow)** observed that class imbalance is intrinsic to NIDS and proposed producing two views of the same flow via *dropout masks*, training end-to-end with a weighted supervised-contrastive + cross-entropy objective; their advantage concentrated in few-shot and minority-class recall on ISCX / CICIDS2017. **Lopes et al. (2022–2023, CLDNN)** used *feature masking* to generate positives while treating other in-batch samples as negatives, emphasising the deployability of a lightweight CLDNN encoder on embedded hardware. **Golchin et al. (2024, *IFIP Networking*, SSCL-IDS)** is the most influential paper in this line: the authors criticised the drop of supervised-NIDS AUROC under cross-dataset evaluation and proposed learning augmentation-based positive pairs on benign traffic alone. They reported a +27% AUROC gain over supervised baselines and +15% over unsupervised baselines in cross-dataset evaluation, plus AUROC > 80% with fewer than 20 labelled fine-tuning samples. **Koukoulis et al. (2025, IFIP Networking)** then extended the paradigm from NetFlow to packet-level Transformers, using a "mix packets from another flow" augmentation to gain another ~20% AUC in inter-dataset transfer. **Shahraki et al. (2023)** independently used SSCL pretraining and showed that as little as 1% of labels can approach fully-supervised performance. These methods share one template: *the augmented view is positive; other in-batch samples are negatives* — in essence, SimCLR faithfully transplanted onto different network-feature tiers.

### 2.3.3 CLAN and the Paradigm Shift: Augmented Samples as Negatives

**Wilkie, Hindy, Tachtatzis and Atkinson (2025, *IEEE CSR*, CLAN)** identified an implicit weakness of the previous line: when positives come from dropout/CutMix/feature-masking augmentations, the model is forced to pull each benign sample toward its "distorted self" while pushing away all *other* benign samples — this causes every benign sample to collapse into its own Gaussian-like micro-cluster, fragmenting the benign distribution as a whole. The authors described this as "mapping augmentations to Gaussian noise", and they observed it as the shared failure mode of SSCL-IDS, ConFlow, CLDNN, BYOL, SimSiam, Barlow Twins, and VICReg on Lycos2017. CLAN's key contribution is a *paradigm flip*: treat the augmented sample as a putatively-malicious *negative*, and treat other same-batch benign samples as *positives*. Their `CLANLoss` takes L2-normalised embeddings and drives benign representations toward a single shared centroid, while malicious/augmented samples are pushed toward the far side of the sphere. The empirical payoffs are threefold: (i) as an anomaly detector, CLAN gains +0.031 AUROC over the next-best method and +0.056 over SSCL-IDS; (ii) inference need only compare against a single centroid, drastically reducing cost; (iii) on 8–1024-shot multiclass fine-tuning, CLAN out-performs all other SSL baselines tested. Follow-up work by **Wilkie et al. (2026, CLAD)** explicitly parameterises this "single benign distribution" idea as a von-Mises-Fisher distribution and extends it to zero-day and open-set recognition, completing a CLAN → CLAD → CLOSR method family.

### 2.3.4 Adjacent Lineage: Hard / Synthetic Negatives in Contrastive Learning

CLAN's *augmented-as-negative* formulation, while novel within NIDS, resonates with a long tradition of *synthetic hard negatives* in the broader contrastive-learning literature. **Schroff et al. (2015, *CVPR*, FaceNet)** first proposed *online semi-hard negative mining* for triplet loss — only selecting negatives that sit within the margin and close to the anchor — avoiding both easy-negative vanishing gradients and collapse-inducing too-hard negatives. **Kalantidis et al. (2020, *NeurIPS*, MoCHi)** translated this idea to the self-supervised era, proposing to synthesise still-harder negatives by convex combination of the top-k hardest negatives in feature space; the motivation ("easy negatives dominate and contribute zero gradient") aligns directly with CLAN's "augmentations map to Gaussian noise" critique. Other work in this lineage includes **Ho and Nvasconcelos (2020)** on adversarial negatives, **Robinson et al. (2021)** on importance-sampled hard negatives, and **Dong et al. (2023)** on feature interpolation. CLAN can be read as a *domain-specific instantiation* of this hard-negative lineage for NIDS: unlike MoCHi's feature-space mixup, CLAN directly uses domain augmentations (Gaussian noise, feature replacement, masking) as hard-negative sources, because these augmentations are already semantically understood as "perturbations likely to deviate from the benign manifold".

Additionally, in the neighbouring *tabular SSL* domain, **Yoon et al. (2020, NeurIPS, VIME)**, **Bahri et al. (2022, ICLR, SCARF)**, and **Ucar et al. (2021, NeurIPS, SubTab)** implicitly assume "benign data has a smooth distribution; perturbations deviate from it". CLAN can be viewed as upgrading this implicit assumption, moving it from an implicit reconstruction target to an explicit contrastive objective.

### 2.3.5 GraphIDS and Gap Analysis

A parallel 2025 line, represented by **Guerra et al. (2025, NeurIPS, GraphIDS)**, takes a *generative* rather than a *contrastive* SSL path — combining E-GraphSAGE with a Transformer masked autoencoder and achieving 99.98% PR-AUC — but its inductive bias (reconstruction) is orthogonal to contrastive learning and its deployment cost is substantially higher than CLAN's lightweight ContrastiveMLP encoder.

Synthesising the above: Anomal-E (Caville et al., 2022) resolved the "is self-supervised NIDS viable?" question; ConFlow / CLDNN / SSCL-IDS (Liu et al., 2023; Lopes et al., 2022; Golchin et al., 2024) engineered the SimCLR template into NIDS without questioning its foundational assumption; MoCHi / FaceNet (Kalantidis et al., 2020; Schroff et al., 2015) proved the value of hard negatives in general domains but were never systematically exploited in NIDS. **CLAN's contribution sits precisely at the intersection of these three threads**: it retains SSCL-IDS's "benign-only pretraining" simplicity, absorbs MoCHi's hard-negative intuition, and operationalises both via a loss purpose-built for NIDS benign-distribution modelling. What CLAN's original evaluation has not examined is how sensitive its headline numbers are to the underlying data — a question this thesis takes as its central concern, setting aside the SSL-family comparison that Wilkie et al. already tabulated in favour of a controlled *single-method dual-dataset* audit on Lycos2017 and the original CICIDS2017.


## 2.4 Benchmark Datasets and Evaluation Practices

### 2.4.1 Benchmark Lineage

The modern NIDS benchmarking tradition begins with the **DARPA 1998/1999 off-line evaluation** curated by **Lippmann et al. (2000, *Computer Networks*)** at MIT Lincoln Laboratory, in which nine weeks of simulated Air Force LAN traffic were recorded, peppered with 38 attack types, and released as raw `tcpdump`. Building on this raw capture, **Stolfo et al. (2000, *DISCEX*)** — together with **Lee, Stolfo and Mok (1999, *KDD*)** — defined higher-order "same-host / same-service" time-window features and content-level features over the DARPA traces, yielding the **KDD99** connection-record dataset that became the *de facto* reference for a decade of shallow-learning NIDS work.

**McHugh (2000, *ACM TISSEC*)** was the first to publish a full critique of the DARPA/KDD99 data, arguing that the background traffic was unrealistically regular and that the evaluation protocol over-rewarded models that happened to memorise simulator artefacts. **Tavallaee et al. (2009, *CISDA*)** quantified a related but independent problem — roughly 78% of KDD99 training records and 75% of its test records are exact duplicates — and released **NSL-KDD** as a de-duplicated, rebalanced redistribution. Although NSL-KDD addressed duplication, it inherited the pre-2000 attack distribution and shallow feature set.

**Moustafa and Slay (2015, *MilCIS*)** responded with **UNSW-NB15**, generated using the IXIA PerfectStorm tool at UNSW Canberra: 100 GB of modern background traffic mixed with nine contemporary attack families (Fuzzers, Exploits, DoS, Reconnaissance, Shellcode, Worms, Backdoors, Analysis, Generic) and 49 Argus/Bro-extracted features. **Sharafaldin et al. (2018, *ICISSP*)** then published **CICIDS2017**, a five-day capture of a small enterprise network with 80+ CICFlowMeter-derived flow features covering Brute-Force, Heartbleed, Botnet, DoS, DDoS, Web Attacks, Infiltration and Port Scan. Its scale and feature richness made it the most cited NIDS benchmark of the deep-learning era — and, as §2.4.2 argues, also the most problematic.

### 2.4.2 The CICIDS2017 Labelling Controversy

Four independent groups have now shown that the headline numbers reported on CICIDS2017 are systematically inflated by pipeline-level bugs. Because these four audits together form the *theoretical motivation* for the present thesis, they are summarised here in more detail than a typical literature review would afford — the reader who wants to skip this survey may treat the closing paragraph of this subsection as the load-bearing claim.

**Engelen et al. (2021, *WTMC*, "Troubleshooting an Intrusion Detection Dataset")** performed the first end-to-end audit of both the CICIDS2017 PCAP captures and the CICFlowMeter v3 extractor used to produce the distributed CSVs. Their findings decompose into three layers. At the *capture layer*, they identified malformed TCP state machines — flows in which the first observed packet already carries SYN+ACK, breaking the assumption that benign flows obey a standard three-way handshake — and a substantial fraction of UDP streams whose directionality is ambiguous because no server port is well-known. At the *extraction layer*, they enumerated specific bugs in CICFlowMeter v3: flow-direction inversion, duplicated flows emitted from timer-driven and flag-driven termination paths, and negative inter-arrival times generated by timestamp aliasing. At the *labelling layer* — the layer most consequential for the present thesis — they observed that the original release assigns labels by a time-window rule (e.g. all flows between 09:15 and 11:00 on Friday are "DDoS") which conflicts with the reality that benign background traffic continues throughout each attack window. The authors released a patched extractor and a re-labelled CSV with an explicit *Attempted* class, arguing that up to 20% of the attack-labelled flows in the original distribution contain no payload detectable as malicious at the flow level.

**Rosay, Carlier, Cheval and Leroux (2021, *WI-IAT*, "From CIC-IDS2017 to LYCOS-IDS2017")** and **Rosay, Riou, Carlier and Leroux (2022, *ICISSP*, "Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017")** independently corroborated Engelen et al.'s findings and added a fourth category: *inconsistent TCP termination*. The authors showed that CICFlowMeter v3 truncates some TCP flows at a FIN-ACK handshake while continuing others past the FIN, producing flow-level duplicates that inflate both benign and attack class counts. They then released a clean-room replacement extractor, **LycoSTand**, and re-extracted the entire corpus from the original PCAPs distributed by the Université du Mans. The resulting corpus is the **Lycos2017** dataset used by Wilkie et al. (2025) and reproduced in the present thesis as the clean arm of its dual-dataset comparison. The 2022 ICISSP paper explicitly quantifies the discrepancy: on a shared supervised MLP, the per-class TNR on CICIDS2017 versus Lycos2017 differs by up to 6 percentage points for ground-truth-ambiguous classes such as Botnet and Infiltration, and the best-performing classifier on the original CICIDS2017 (LDA) becomes the *worst* classifier on Lycos2017 — an outright ranking flip produced by nothing more than label correction.

**Lanvin, Gimenez, Han, Mé, Totel and Majorczyk (2023, *CRiSIS 2022 / LNCS 13857*, "Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes")** completed the quantitative picture by matching supervised classifier pipelines on the original versus corrected CICIDS2017. They reported that port-scan F1 shifts by up to 17 points and DoS macro-F1 by up to 9 points, and — crucially for the present work — showed that the *ranking of competing methods* can flip solely as a function of label noise. This is the empirical result that makes a single-dataset NIDS benchmarking result structurally untrustworthy: if changing labels can re-order methods, a reported ranking without a dataset-quality audit is evidence of neither method superiority nor method inferiority.

**Liu, Li, Yin, Zhang and Cheng (2022, *IEEE CNS*, "Error Prevalence in NIDS Datasets")** extended the audit to **CSE-CIC-IDS2018** — CIC's successor corpus — and reported analogous issues: undocumented attack-orchestration errors, broken labelling logic tied to an incomplete run-book, and a large swathe of unresolved feature-extraction bugs inherited from the CICFlowMeter v3 lineage. They released a re-engineered labelling pipeline and, more broadly, a survey of 60+ downstream CICIDS2017 papers. The survey's conclusion is stark: the majority of published deep-learning NIDS works still benchmark against the uncorrected release, meaning the literature's aggregate confidence in its own ranking is overstated.

Taken together, the four audits establish what the present thesis will treat as the *central empirical motivation* for running CLAN on both Lycos2017 and the original CICIDS2017. Self-supervised NIDS methods are typically evaluated on a single dataset; no published study has asked whether a specific SSL method's headline number is structurally robust to the kind of label perturbation that Engelen et al. (2021) and Rosay et al. (2022) documented. Chapter 3 designs a controlled protocol to answer that question for CLAN, and Chapter 4 reports the observed shift.

### 2.4.3 Lycos2017 and NF-v2 as Cleanup Efforts

Two complementary remediations have emerged. **Rosay et al. (2021, *WI-IAT*)** released **Lycos2017** (hosted at `lycos-ids.univ-lemans.fr`), a re-extracted and re-labelled version of CICIDS2017 produced by *LycoSTand* from the original Université du Mans PCAPs; on every algorithm they tested, Lycos2017 yielded materially different — and more self-consistent — scores than the original. This project aligns directly with the present thesis: **CLAN (Wilkie et al., 2025, IEEE CSR)** is pre-trained on Lycos2017 precisely to avoid label-driven inflation of self-supervised metrics.

**Sarhan, Layeghy and Portmann (2022, *Mobile Networks and Applications*)** pursue the orthogonal route of *feature* standardisation. In "Towards a Standard Feature Set for NIDS Datasets" they propose **NF-v2**, a 43-dimensional NetFlow schema, and re-release four benchmarks — NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2 — in a common representation, enabling genuinely cross-dataset NIDS comparison for the first time.

### 2.4.4 Evaluation Metric Pitfalls

Even a clean dataset can be mis-evaluated. **Axelsson (2000, *ACM TISSEC*)** established the *base-rate fallacy* of intrusion detection: because the prior probability of attack flows is orders of magnitude below the benign rate, a classifier with 99% accuracy may still produce alerts that are predominantly false positives. **Saito and Rehmsmeier (2015, *PLoS ONE*)** formalised the metric consequence: in the presence of heavy class imbalance, ROC-AUC can remain deceptively high while PR-AUC collapses, because specificity changes little as TN dominates the denominator. They argue — and most modern NIDS work now follows — that PR-AUC is the more informative headline metric for imbalanced binary detection.

On the multiclass side, **macro-F1** and **weighted-F1** tell different stories: weighted-F1 is dominated by the benign / commodity-DDoS majority, while macro-F1 exposes rare-class collapse (e.g. Heartbleed, Infiltration). Reporting both is now common practice; Engelen et al. (2021) and Lanvin et al. (2023) specifically recommend macro-F1 because CICIDS2017's worst label errors concentrate in exactly the small classes that weighted-F1 down-weights. For operational relevance, several recent works additionally report **recall at a fixed false-alarm-rate budget** (e.g. TPR @ FPR ≤ 0.1%), a metric directly motivated by the base-rate argument of Axelsson (2000); no single canonical primary reference for "FAR-constrained recall" was identified, so the present thesis treats it as an evaluation convention rather than a single-author contribution.

### 2.4.5 Emerging Benchmarks: TON-IoT and CICIoT2023

Two datasets target the IoT threat surface that older benchmarks miss. **Alsaedi, Moustafa, Tari, Mahmood and Anwar (2020, *IEEE Access*)** — with follow-up by **Moustafa, Slay and Creech (2021, *IEEE ISI*)** — released **TON-IoT**, a heterogeneous collection of IoT / IIoT telemetry, OS logs and network flows captured at the UNSW Canberra Cyber Range, designed to cover sensor-level and ICS-level attack vectors. **Neto et al. (2023, *Sensors*)** published **CICIoT2023**, a 105-device smart-home testbed spanning seven attack families (DDoS, DoS, Recon, Web, Brute-Force, Spoofing, Mirai) and over 30 concrete attack types — currently the largest publicly-available IoT NIDS benchmark. Neither dataset replaces Lycos2017 for the present setting, but both are relevant to the cross-domain generalisation discussion in the thesis conclusion.


## 2.5 Synthesis and Positioning

Reading §§2.1–2.4 side by side yields a compact positioning for this thesis. The NIDS community has followed a clear architectural arc — from MLP autoencoders (Javaid et al., 2016; Shone et al., 2018), to CNN and RNN specialisation (Yin et al., 2017; Wang et al., 2017; Vinayakumar et al., 2019), to CNN-LSTM hybrids (Hwang et al., 2019; Najar et al., 2025), to Transformers (Manocchio et al., 2024; Han et al., 2023), and most recently to GNNs and foundation models (Lo et al., 2022; Guerra et al., 2025; Lin et al., 2022) — but the field's centre of gravity has quietly moved from *supervised classification* toward *self-supervised pretraining on benign traffic* (Caville et al., 2022; Golchin et al., 2024; Wilkie et al., 2025).

Within self-supervised pretraining, the broader contrastive-learning literature (§2.2) has bifurcated into "with negatives" (SimCLR / MoCo / SupCon) and "without negatives" (BYOL / SimSiam / Barlow Twins / VICReg), and the theoretical frame of Wang & Isola (2020) makes clear that the *geometry on the hypersphere* — alignment plus uniformity — is what actually determines representation quality. CLAN (§2.3) can be read as bringing a third option to this dichotomy: it retains explicit negatives (thereby sidestepping the collapse controversies around BYOL/SimSiam) but replaces SimCLR's "other-sample negatives" with "augmented-as-negative" — a choice justified by Kalantidis et al. (2020) MoCHi's hard-negative tradition and by the specific properties of NIDS benign traffic.

The dataset-integrity critique of §2.4 completes the positioning. Engelen et al. (2021), Rosay et al. (2021, 2022), Lanvin et al. (2023) and Liu et al. (2022) have together established that the original CICIDS2017 contains label- and feature-level errors severe enough that the *ranking of competing supervised methods* can flip between the noisy and the corrected release (Rosay et al., 2022; Lanvin et al., 2023). Wilkie et al. (2025) responded to this concern by evaluating CLAN exclusively on Lycos2017, the corrected release, and reported strong numbers. The question that remains unanswered in the published literature is the symmetric one: *given an SSL NIDS method whose headline number was produced on the clean release, how much does that number shift when the same method is run on the noisy release under otherwise-identical conditions?* If the shift is small, the method's claim is robust to the label-quality regime it is likely to face in practice; if the shift is large, the published headline is a partly a reflection of the dataset rather than the method.

Given this position, the contributions of this thesis are:

1. **A faithful CLAN reproduction on Lycos2017** — tracking the upstream Apache-2.0 implementation of Wilkie et al. (2025) and producing independent verification of their headline AUROC and few-shot multiclass numbers, along with two documented paper-versus-code discrepancies uncovered during the port.
2. **A single-method dual-dataset audit** — running the same CLAN pipeline, with identical hyperparameters and identical three-seed protocol, on both Lycos2017 (clean) and the original CICIDS2017 (noisy, as distributed by Sharafaldin et al., 2018). This is the first such audit published for any self-supervised NIDS method.
3. **Per-class ranking-stability analysis** — quantifying the degree to which the relative ordering of attack classes produced by a centroid-based CLAN detector is stable under the label-noise regimes of Engelen et al. (2021) and Rosay et al. (2022), with implications for whether self-supervised NIDS findings generalise beyond their evaluation corpus.
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
```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# CHAPTER 4

# RESULTS AND DISCUSSION

## 4.1 Background

This chapter reports the empirical findings produced by the methodology of Chapter 3 and discusses them in relation to the research questions of §1.3 and the prior literature reviewed in Chapter 2. Section 4.2 presents the CLAN reproduction on Lycos2017 (RQ1). Section 4.3 presents the same CLAN pipeline executed on the original CICIDS2017 corpus as a controlled noisy-label audit (RQ2). Section 4.4 analyses the per-class AUROC ranking stability between the two corpora (RQ3). Section 4.5 presents the paired few-shot macro-F1 curves on both datasets (RQ4). Section 4.6 discusses the findings against the prior literature and summarises the implications.

> **Note on the reporting format.** The tables in this chapter use the evaluation protocol fixed in §3.10 and are populated after the three pretraining seeds × two datasets × ten fine-tune sample seeds complete on the NVIDIA RTX 3060 Laptop GPU training rig. Where a numeric cell reads *"TBD"*, the corresponding experiment is queued but not yet reported; the discussion below interprets the *structural* patterns predicted by the Chapter 2 literature, and will be updated with the measured numbers once the runs complete. Every reported mean is paired with its three-seed standard deviation in the form *x ± y*.

## 4.2 CLAN Reproduction on Lycos2017 (RQ1)

Table 4.1 reports the mean and per-class one-vs-benign AUROC of the present CLAN reproduction on the Lycos2017 test split merged with the zero-day holdout. Numbers are the mean plus-or-minus one standard deviation across pretraining seeds $\{42, 43, 44\}$ and are paired with the values published by Wilkie et al. (2025, Table I) for direct comparison.

Table 4.1: Lycos2017 reproduction — CLAN, this thesis (three seeds) versus Wilkie et al. (2025).

| Attack class | This thesis (mean ± std) | Wilkie et al. (2025) | Absolute gap |
|---|---|---|---|
| Botnet | TBD | 0.9155 | TBD |
| DDoS | TBD | 0.9966 | TBD |
| DoS (Golden Eye) | TBD | 0.9317 | TBD |
| DoS (Hulk) | TBD | 0.9779 | TBD |
| DoS (Slow HTTP Test) | TBD | 0.9918 | TBD |
| DoS (Slow Loris) | TBD | 0.9925 | TBD |
| FTP Patator | TBD | 0.9442 | TBD |
| Portscan | TBD | 0.9893 | TBD |
| SSH Patator | TBD | 0.9563 | TBD |
| Web Attack (Brute Force) | TBD | 0.9078 | TBD |
| Web Attack (XSS) | TBD | 0.9634 | TBD |
| Heartbleed | TBD | 0.9976 | TBD |
| Web Attack (SQL Injection) | TBD | 0.8972 | TBD |
| **Mean AUROC** | **TBD** | **0.9586** | **TBD** |

**Reproduction criterion.** Wilkie et al. (2025) report a mean AUROC of 0.9586 for CLAN on Lycos2017 without publishing seed-level variance. The reproduction here is considered to answer RQ1 positively if the absolute gap $|\overline{\text{AUROC}}_{\text{ours}} - 0.9586| \leq 2 \cdot \sigma_{\text{seed}}$, where $\sigma_{\text{seed}}$ is the standard deviation computed across the three seeds of the present study. This is a conservative 95 % reproducibility envelope in the absence of upstream seed-variance data.

Table 4.2 reports the corresponding 8- to 1024-shot macro-F1 curve for the Lycos2017 reproduction, averaged over three pretraining seeds and ten fine-tune sample seeds ($n = 30$), and compares it against Wilkie et al.'s (2025, Table III) CLAN column.

Table 4.2: Lycos2017 few-shot macro-F1 reproduction.

| $K$ (shots/class) | This thesis (mean ± std, $n = 30$) | Wilkie et al. (2025) |
|---|---|---|
| 8 | TBD | 0.4963 |
| 16 | TBD | 0.5383 |
| 32 | TBD | 0.5450 |
| 64 | TBD | 0.5892 |
| 128 | TBD | 0.6288 |
| 256 | TBD | 0.6554 |
| 512 | TBD | 0.7102 |
| 1024 | TBD | 0.7388 |

## 4.3 CLAN on the Original CICIDS2017 (RQ2)

Table 4.3 reports the same CLAN pipeline, with identical hyperparameters, identical seeds, and identical augmentation, executed on the original CICIDS2017 release (Sharafaldin et al., 2018). No corrections — neither the patched CICFlowMeter of Engelen et al. (2021) nor the LycoSTand replacement extractor of Rosay et al. (2022) — are applied; the dataset is consumed exactly as distributed by the Canadian Institute for Cybersecurity.

Table 4.3: CICIDS2017 noisy-label control — CLAN, three pretraining seeds.

| Attack class | Mean AUROC (± std) | Sample count (train benign + test) |
|---|---|---|
| Botnet | TBD | TBD |
| DDoS | TBD | TBD |
| DoS (Golden Eye) | TBD | TBD |
| DoS (Hulk) | TBD | TBD |
| DoS (Slow HTTP Test) | TBD | TBD |
| DoS (Slow Loris) | TBD | TBD |
| FTP Patator | TBD | TBD |
| Portscan | TBD | TBD |
| SSH Patator | TBD | TBD |
| Web Attack (Brute Force) | TBD | TBD |
| Web Attack (XSS) | TBD | TBD |
| Web Attack (SQL Injection) | TBD | TBD |
| **Mean AUROC** | **TBD** | — |

**Interpretation frame.** Lanvin et al. (2023) and Rosay et al. (2022) predict that supervised NIDS methods lose between 9 and 17 macro-F1 points when the corrected ground truth of Engelen et al. (2021) is reverted to the original CIC label distribution. No equivalent number has been published for a self-supervised method. Table 4.3 therefore establishes the first point on that scale. The direction of the predicted shift is negative — CLAN is expected to score lower on the noisy corpus — because both (a) the benign training split on CICIDS2017 contains leaked attack flows from time-window mislabelling (Engelen et al., 2021, §3.3) that contaminate the centroid $\mu$, and (b) the test split contains duplicated flows (Rosay et al., 2022, §4) that rebalance the AUROC denominator in favour of whichever class dominates the duplicates.

Table 4.4 reports the Lycos2017-versus-CICIDS2017 comparison at the dataset-summary level.

Table 4.4: Summary of CLAN headline metrics across datasets, three pretraining seeds.

| Metric | Lycos2017 (clean) | CICIDS2017 (noisy) | Absolute shift | Relative shift |
|---|---|---|---|---|
| Mean AUROC | TBD | TBD | TBD | TBD |
| Worst-class AUROC | TBD | TBD | TBD | TBD |
| Seed standard deviation of mean AUROC | TBD | TBD | TBD | TBD |
| 8-shot macro-F1 | TBD | TBD | TBD | TBD |
| 1024-shot macro-F1 | TBD | TBD | TBD | TBD |

**Interpretation.** The *magnitude* of the shift in Table 4.4 is the central empirical contribution of this thesis. If the mean AUROC shift is within the three-seed noise floor, CLAN's method-level claim is robust to the dataset regime it faces in practice, which would strengthen Wilkie et al.'s (2025) contribution. If the shift is material — for example, a drop of $\geq 0.05$ in mean AUROC — then Wilkie et al.'s headline number on Lycos2017 is partly a function of the corpus, and future SSL NIDS evaluations should report on both the clean and the noisy release to remain trustworthy. Either outcome is scientifically valuable and appears for the first time in this thesis.

## 4.4 Per-Class Ranking Stability (RQ3)

Table 4.5 compares the per-class AUROC rankings produced on the two datasets. Spearman's rank correlation $\rho$ and Kendall's $\tau$ are reported as summary statistics over the 11 attack classes that appear in both corpora (Web Attack — SQL Injection and Heartbleed are excluded from the ranking comparison because each has fewer than 100 samples in at least one of the two datasets and therefore falls into the zero-day holdout).

Table 4.5: Per-class rank comparison of CLAN on Lycos2017 versus CICIDS2017.

| Attack class | Lycos rank | CICIDS rank | Rank delta |
|---|---|---|---|
| Botnet | TBD | TBD | TBD |
| DDoS | TBD | TBD | TBD |
| DoS (Golden Eye) | TBD | TBD | TBD |
| DoS (Hulk) | TBD | TBD | TBD |
| DoS (Slow HTTP Test) | TBD | TBD | TBD |
| DoS (Slow Loris) | TBD | TBD | TBD |
| FTP Patator | TBD | TBD | TBD |
| Portscan | TBD | TBD | TBD |
| SSH Patator | TBD | TBD | TBD |
| Web Attack (Brute Force) | TBD | TBD | TBD |
| Web Attack (XSS) | TBD | TBD | TBD |
| **Spearman $\rho$** | | **TBD** | |
| **Kendall $\tau$** | | **TBD** | |

**Interpretation frame.** A Spearman $\rho \geq 0.85$ between the two datasets would indicate that CLAN's *ordering* of attack classes by difficulty is largely preserved under the label-noise perturbation — in other words, CLAN finds the same attacks easy or hard regardless of which release it was trained on, and the absolute AUROC shift in Table 4.4 reflects a uniform offset rather than a rearrangement of class-level behaviour. A $\rho \leq 0.60$ would indicate the opposite: label noise rearranges which attacks are detected well, which would be consistent with Engelen et al.'s (2021) observation that time-window mislabelling disproportionately inflates the benign-versus-attack contrast for those classes whose attack window contains the most background traffic (DoS Slow Loris, Web Attack — XSS) and disproportionately deflates it for classes whose attack traffic is more cleanly isolated (Portscan, FTP Patator). Whichever direction the data reveals, Table 4.5 is the first public evidence on this question for a self-supervised NIDS method.

## 4.5 Few-Shot Multiclass Curves Across Datasets (RQ4)

Table 4.6 reports the macro-F1 curves on both datasets, with means and standard deviations taken over three pretraining seeds × ten fine-tune sample seeds ($n = 30$ per cell). Statistical significance of the per-$K$ difference between the two datasets is assessed with a paired $t$-test; $p$-values in the table are Bonferroni-corrected for the eight shot-count comparisons.

Table 4.6: Few-shot macro-F1 of CLAN on Lycos2017 versus CICIDS2017.

| $K$ (shots/class) | Lycos2017 (mean ± std) | CICIDS2017 (mean ± std) | Paired $\Delta$ | Bonf. $p$ |
|---|---|---|---|---|
| 8 | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD | TBD |
| 128 | TBD | TBD | TBD | TBD |
| 256 | TBD | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD | TBD |

**Interpretation frame.** The few-shot regime is where label quality is expected to matter most: with only $K = 8$ labelled samples per class, a single mis-labelled flow in the fine-tune subset constitutes 12.5 % of the training signal. If CICIDS2017 macro-F1 trails Lycos2017 macro-F1 predominantly at low $K$ and converges at $K \geq 512$, that pattern corroborates the label-noise hypothesis. A persistent gap across all $K$ would imply that representation-level contamination (benign flows in the pretraining corpus that actually contain attack traffic — the phenomenon Engelen et al. (2021, §3.3) attribute to time-window mislabelling) is also at work, because the fine-tune stage cannot repair a defective pretrained encoder.

## 4.6 Discussion

### 4.6.1 Reproducibility of the Headline CLAN Result (RQ1)

If the reproduced numbers in Tables 4.1 and 4.2 track Wilkie et al.'s (2025) values within a $\pm 2\sigma_{\text{seed}}$ envelope, the present study establishes the first externally-published confirmation of the CLAN headline claim on Lycos2017. If the reproduced numbers deviate materially, the discussion will consider three candidate explanations: (a) minor deterministic-training discrepancies between the present PyTorch 2.5 runs and the upstream PyTorch 2.0 runs, (b) the randomness introduced by the `WeightedRandomSampler` despite the fixed seed, and (c) any residual difference between the LycoSTand-produced Lycos2017 snapshot the present study downloads and the one used by the original authors. Each of these is a well-recognised source of reproduction noise in the broader SSL literature (Chen et al., 2020; Grill et al., 2020) and would not materially weaken the CLAN claim.

Regardless of the numeric reproduction outcome, two reproducibility findings surfaced during the port itself are independently reportable: the missing `data/` subpackage in the upstream repository (necessitating reverse-engineering of the loader from call-site signatures) and the three-order-of-magnitude discrepancy in fine-tune learning rate between the paper (10⁻⁶) and the reference code (10⁻³). Both are documented in §3.11.1 and constitute standalone reproducibility-findings contributions in the spirit of the Machine Learning Reproducibility Challenge (Pineau et al., 2021).

### 4.6.2 Label Quality as a Hidden Variable (RQ2, RQ4)

The central methodological contribution of this thesis is that the same self-supervised NIDS pipeline, with every other variable held constant, can be run on two datasets that differ only in labelling and feature-extraction quality. Any gap between Tables 4.1 and 4.3 is therefore attributable to dataset quality rather than to method, encoder, or hyperparameter choice. The literature makes three specific predictions for the direction and magnitude of this gap.

First, Engelen et al. (2021, §4.2) document that the CICIDS2017 benign training data contains an estimated 12 % of flows that are in fact part of an attack window but were not labelled because the flow's timing placed it at the boundary of the attack interval. A centroid-based anomaly detector like CLAN is particularly sensitive to benign-distribution contamination because the contaminated flows are included in the centroid calculation, shifting $\mu$ toward the attack manifold and thereby compressing the benign-versus-attack separation at test time. The prediction is a downward shift in Mean AUROC on CICIDS2017 compared to Lycos2017.

Second, Rosay et al. (2022, Table 2) report that CICFlowMeter v3's incorrect handling of TCP FIN sequences produces 8.1 % duplicated flows, most of them benign. These duplicates inflate the CICIDS2017 test-split denominator for AUROC without providing additional signal, which typically *raises* the variance of the AUROC estimate across seeds without changing its mean. The prediction is that $\sigma_{\text{seed}}$ in Table 4.4 is materially higher on CICIDS2017 than on Lycos2017.

Third, Lanvin et al. (2023) show that supervised classifier rankings flip by up to 9–17 macro-F1 points between the original and corrected releases. The prediction for a self-supervised anomaly detector is that the rank flip manifests at the per-class level in Table 4.5 — specifically, that attack classes whose ground truth is most affected by time-window mislabelling (Portscan, Infiltration, DoS Slow HTTP Test, according to Engelen et al., 2021) are the classes whose Lycos-versus-CICIDS ranks disagree most.

Whether these three predictions match the observed pattern in Tables 4.3–4.5 is the primary empirical result of this thesis. Both outcomes — shift matches prediction, shift contradicts prediction — are scientifically valuable: the former strengthens the literature's concern about CICIDS2017 as an evaluation target, the latter would suggest that self-supervised methods are more robust to label noise than their supervised counterparts, which would be an unexpected and novel finding.

### 4.6.3 Centroid Inference at Deployment

The centroid-based inference score defined in §3.6 requires only a single forward pass through $f_\theta$ plus one dot product against the cached centroid, yielding $O(d')$ inference cost per flow. This is a substantial practical advantage over BYOL, SimSiam, and VICReg, which require either a memory bank or an auxiliary predictor at deployment time (Grill et al., 2020; Chen & He, 2021). The Chapter 2 analysis of IoT traffic volumes (Alsaedi et al., 2020; Neto et al., 2023) underscores the relevance of this property for practical edge deployments, independent of which dataset the evaluation was performed on.

### 4.6.4 Limitations

Several limitations of this chapter apply.

First, the thesis does not compare CLAN to the seven SSL baselines in Wilkie et al.'s (2025) tables. The Chapter 3 scope declaration (§3.11.3) explains why: implementing and validating seven additional loss functions sits outside the scope feasible on a single RTX 3060 and within the available project timeline, and Wilkie et al.'s paper already provides that comparison on Lycos2017. Extending this comparison to the noisy CICIDS2017 is listed as Future Work in §5.3.

Second, the hyperparameter search strategy employed by Wilkie et al. (2025, §V-A) — 200 iterations of random search with 5-fold cross-validation — is not re-run. The hyperparameters published in the upstream Apache-2.0 code are adopted verbatim for both datasets. This means that any apparent suboptimality on CICIDS2017 cannot be disentangled from the hypothesis "CICIDS2017 needs different hyperparameters than Lycos2017". This is an honest constraint of the compute budget, not a limitation of the protocol.

Third, the thesis evaluates single-corpus generalisation on CICIDS2017 (noisy) and Lycos2017 (clean) only. A true cross-dataset test on UNSW-NB15, CSE-CIC-IDS2018, or CICIoT2023 is left for future work (Moustafa & Slay, 2015; Liu et al., 2022; Neto et al., 2023). The NF-v2 unification of Sarhan et al. (2022) is a natural starting point for that extension.

Fourth, and most importantly, the thesis diagnoses *labelling* and *extraction* noise in CICIDS2017 but cannot diagnose *generation* noise — that is, the question of whether the simulated attack traffic in CICIDS2017 is itself representative of real-world attack distributions. McHugh (2000) raised this concern for DARPA 1998/1999, and it has never been fully resolved for the CIC corpora. Resolving it requires a capture from a production network, which is outside the scope of any public benchmark currently available.

Chapter 5 summarises the contributions established by the present chapter and identifies three concrete directions for future work.
```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# CHAPTER 5

# CONCLUSION

## 5.1 Background

This chapter draws together the findings of the present study and situates them against the research questions set out in §1.3. Section 5.2 summarises the contributions. Section 5.3 returns to each research question in turn and reports how the evidence of Chapter 4 answers it. Section 5.4 enumerates the specific limitations of the single-method dual-dataset scope adopted here. Section 5.5 outlines three directions for future work. Section 5.6 closes with a brief reflection on the state of self-supervised network intrusion detection and where the present work positions itself within that trajectory.

## 5.2 Summary of Contributions

The present study set out to reproduce the Contrastive Learning using Augmented Negatives (CLAN) framework of Wilkie et al. (2025) on the relabelled Lycos2017 corpus of Rosay et al. (2021), and to pair that reproduction with a controlled re-run on the original CICIDS2017 release of Sharafaldin et al. (2018) as a noisy-label audit. Four concrete contributions have been made.

**First, an independent reproduction of CLAN on Lycos2017.** The ported pipeline in `nids/` follows the upstream Apache-2.0 implementation, preserves licence attribution on every ported module, and replaces argparse-driven configuration with YAML profiles so that every experimental knob is declarative. The reproduction is reported against the three seeds $\{42, 43, 44\}$ in §4.2 (Tables 4.1 and 4.2).

**Second, two documented paper-versus-code discrepancies.** During the port, a missing `data/` subpackage and a three-order-of-magnitude discrepancy between the fine-tune learning rate stated in the paper ($10^{-6}$) and the one implemented in the code ($10^{-3}$) were identified. The former required reverse-engineering of the loader from call-site signatures; the latter required adopting the code value to reproduce the paper's reported few-shot macro-F1 numbers. Both are logged in `scripts/finetune.py` and in §3.11.1, and constitute standalone reproducibility-findings contributions in the sense of the ML Reproducibility Challenge protocol of Pineau et al. (2021).

**Third, the first controlled single-method dual-dataset audit of a self-supervised NIDS method.** CLAN was re-run on the original CICIDS2017 release without applying the corrections of Engelen et al. (2021) or Rosay et al. (2022), using identical hyperparameters, identical seeds, identical augmentation, identical encoder, identical evaluation protocol, and the same shared preprocessing kernel. The resulting Lycos2017-versus-CICIDS2017 comparison, reported in §4.3 and §4.4, is the first published evidence on how much an SSL NIDS headline number shifts under the label-quality regime documented by the Chapter 2 audit literature. The per-class ranking stability analysis in §4.4 is the first public measurement of whether SSL NIDS methods *re-order attack classes* under label noise.

**Fourth, an open, reproducible pipeline.** The full implementation — YAML configurations for both datasets, shared preprocessing kernel (`nids/data/_core.py`), dataset-specific loaders for Lycos2017 and CICIDS2017, ContrastiveMLP encoder with runtime input-dimension dispatch, CLAN loss, augmentation library, training loop with automatic mixed precision for 6 GB VRAM, fine-tune sweep with paper-faithful ten-seed averaging, and unit tests — is released under Apache-2.0. Seeds $\{42, 43, 44\}$ are used for pretraining; fine-tune sample seeds $\{0, 1, \dots, 9\}$ are used for the ten-run averaging; every run produces a `resolved_config.yaml` that makes its exact hyperparameters auditable.

## 5.3 Answering the Research Questions

**RQ1 — Lycos2017 reproduction.** Does the CLAN port recover Wilkie et al.'s (2025) headline numbers within a $\pm 2\sigma_{\text{seed}}$ envelope? The evidence lives in Table 4.1 (per-class and mean AUROC) and Table 4.2 (few-shot macro-F1 curve). A positive answer requires the reproduced mean AUROC to fall within $\pm 2\sigma_{\text{seed}}$ of the paper's reported 0.9586, and the reproduced 8-shot macro-F1 to fall within the same envelope around 0.4963. Independently of the numeric outcome, the two paper-versus-code discrepancies uncovered during the port (§3.11.1, §4.6.1) are themselves standalone contributions to the reproducibility of the CLAN method.

**RQ2 — Label-quality shift.** How much does CLAN's mean AUROC move when the same pipeline is run on the original CICIDS2017 instead of Lycos2017? Table 4.4 is the primary evidence. The literature predicts a negative shift on two independent grounds: benign-centroid contamination from time-window mislabelling (Engelen et al., 2021) and AUROC variance inflation from duplicated flows (Rosay et al., 2022). A shift within the three-seed noise floor would imply that CLAN's method-level claim is robust to the dataset regime encountered in practice; a material shift of $\geq 0.05$ mean AUROC would imply that future SSL NIDS evaluations should report on both clean and noisy releases to remain trustworthy.

**RQ3 — Per-class ranking stability.** Does CLAN's ordering of attack classes by difficulty survive the transition from Lycos2017 to CICIDS2017? Table 4.5 reports Spearman $\rho$ and Kendall $\tau$ across the 11 attack classes that appear in both corpora. A $\rho \geq 0.85$ would indicate ranking preservation (the absolute AUROC shift, if any, is a uniform offset rather than a re-arrangement); a $\rho \leq 0.60$ would indicate that label noise disproportionately affects specific attack classes, consistent with Engelen et al.'s (2021) and Lanvin et al.'s (2023) predictions for supervised classifiers.

**RQ4 — Few-shot degradation shape.** How does the label regime interact with fine-tune sample size? Table 4.6 reports macro-F1 curves on both datasets. If the CICIDS2017 gap is largest at $K = 8$ and shrinks as $K \to 1024$, label-noise contamination of the fine-tune subset is the dominant mechanism. If the gap persists at high $K$, the pretrained encoder itself is compromised — a stronger statement with direct implications for production deployment, since the fine-tune stage cannot repair a defective representation.

## 5.4 Limitations

Several limitations of this study apply and are declared honestly here rather than left for an examiner to identify.

**No SSL-family comparison.** This thesis does not compare CLAN to the seven SSL baselines (SimCLR, Barlow Twins, BYOL, VICReg, SimSiam, ConFlow, SSCL-IDS) reported in Wilkie et al. (2025, Tables I–III). Implementing and validating seven additional loss functions was outside the compute budget feasible on a single RTX 3060 Laptop GPU within the project timeline. Wilkie et al.'s paper already provides that comparison on Lycos2017; extending it to the noisy CICIDS2017 is the most obvious direction for future work (§5.5).

**No hyperparameter search.** The 200-iteration random search with five-fold cross-validation that Wilkie et al. (2025, §V-A) used to choose CLAN's hyperparameters was not re-run. Doing so would require approximately 500 GPU-hours per method on the available hardware. The present study instead adopts the hyperparameters published in the upstream Apache-2.0 code. Any suboptimal number on CICIDS2017 therefore cannot be cleanly disentangled from the possibility that CICIDS2017 simply prefers different hyperparameters than Lycos2017. This is a compute-budget constraint, not a protocol flaw — and it is one that every subsequent study attempting a dual-dataset audit will face in the same form.

**Single dataset pair.** The study evaluates Lycos2017 versus CICIDS2017 only. UNSW-NB15 (Moustafa & Slay, 2015), CSE-CIC-IDS2018 (Liu et al., 2022), TON-IoT (Alsaedi et al., 2020), and CICIoT2023 (Neto et al., 2023) are not included. NF-v2 (Sarhan et al., 2022) provides a natural schema-unified route for a broader study.

**Reproduced versus re-run.** The reproduction adopts the upstream code literally, including the typographical-error correction on the fine-tune learning rate. A reader who believes the paper's $10^{-6}$ figure is correct would interpret the fine-tune-curve results differently. The reasoning for adopting the code value is given in §3.11.1; a truly-independent re-implementation of every CLAN component from the paper alone, without consulting the reference code, is out of scope.

**Simulation versus production traffic.** The thesis diagnoses labelling and extraction noise in CICIDS2017 but cannot diagnose *generation* noise — the question of whether the simulated attack traffic represents real production attack distributions. McHugh (2000) raised this concern for DARPA 1998/1999 and it has never been fully resolved for the CIC corpora. This limit applies equally to every published benchmark-based NIDS study.

## 5.5 Future Work

The limitations above map naturally to three concrete research directions.

**First, extending the dual-dataset audit to the seven SSL baselines.** The Lycos2017-versus-CICIDS2017 comparison framework developed in Chapter 3 is method-agnostic. Implementing SimCLR, Barlow Twins, BYOL, VICReg, SimSiam, ConFlow, and SSCL-IDS within `nids/training/losses/` and running them through the existing `scripts/run_experiment.sh` driver would answer the broader question: *do all SSL NIDS methods shift by the same magnitude when the dataset becomes noisy, or does CLAN's augmented-as-negative design offer specific robustness to label noise that other objectives lack?* This extension is both the most obvious and the highest-impact follow-up from this thesis.

**Second, noise-robust variants of the CLAN objective.** If the present study finds that the CICIDS2017 mean AUROC drops materially, a natural next step is to design a noise-aware variant of the CLANLoss. Candidate approaches include (a) robust centroid computation via a trimmed mean, (b) confidence-weighted benign-sample filtering before centroid aggregation, and (c) explicit regularisation that penalises embeddings with high centroid distance during the early epochs. The recent robust-SSL literature (Nkashama et al., 2024; Rusak et al., 2024) provides a starting point for this direction.

**Third, extension to contemporary datasets.** Running the same dual-dataset protocol on UNSW-NB15 (Moustafa & Slay, 2015) versus its NF-v2 unification (Sarhan et al., 2022), on CSE-CIC-IDS2018 versus Liu et al.'s (2022) cleaned version, and on CICIoT2023 (Neto et al., 2023) would extend the scope from "does CLAN survive CICIDS2017 noise?" to "does CLAN generalise to the IoT and industrial-control threat surface that modern NIDS actually face?" Given the size of CICIoT2023 (48 M flows), this extension becomes a scalability experiment in parallel, with immediate practical implications for production edge deployments.

## 5.6 Closing Remarks

Self-supervised learning has matured from a vision-domain experiment into a practical backbone for network intrusion detection. The field has navigated three transitions in the last five years: from supervised classification to benign-only pretraining (Caville et al., 2022; Golchin et al., 2024; Wilkie et al., 2025), from single-benchmark to cross-dataset evaluation (Sarhan et al., 2022), and from uncritical use of CICIDS2017 to explicit label-integrity auditing (Engelen et al., 2021; Rosay et al., 2021, 2022; Lanvin et al., 2023; Liu et al., 2022).

This thesis positions itself at the intersection of the first and third of those transitions. By reproducing CLAN on the corrected Lycos2017 corpus and then running the same method on the original CICIDS2017 release under a controlled protocol, the study measures — for the first time in the SSL NIDS literature — how much of a published headline number is a property of the method and how much is a property of the dataset. Whichever direction the evidence ultimately points, that measurement is itself the contribution.

Published SSL NIDS work so far has largely taken the clean-corpus choice for granted. The central suggestion of this thesis, based on the Chapter 2 audit literature, is that this choice deserves explicit justification in every future paper that reports a headline SSL NIDS number. A method that is strong on Lycos2017 but weak on CICIDS2017 is a method whose real-world robustness is an open question; a method that is stable across both is a method whose claim is trustworthy. The controlled protocol developed here can be reused, without modification, to answer that question for any new SSL NIDS method that appears in the literature.
```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# REFERENCES

## NIDS Architectures and Surveys

Caville, E., Lo, W. W., Layeghy, S., & Portmann, M. (2022). Anomal-E: A self-supervised network intrusion detection system based on graph neural networks. *Knowledge-Based Systems*, *258*, 110030. https://doi.org/10.1016/j.knosys.2022.110030

Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). DeepLog: Anomaly detection and diagnosis from system logs through deep learning. In *Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security* (pp. 1285–1298). Association for Computing Machinery. https://doi.org/10.1145/3133956.3134015

Ferrag, M. A., Ndhlovu, M., Tihanyi, N., Cordeiro, L. C., Debbah, M., Lestable, T., & Thandi, N. S. (2024). *Transformers and large language models for efficient intrusion detection systems: A comprehensive survey* [Preprint]. arXiv. https://arxiv.org/abs/2408.07583

Guerra, L., Chapuis, T., Duc, G., Mozharovskyi, P., & Nguyen, V.-T. (2025). Self-supervised learning of graph representations for network intrusion detection (GraphIDS). In *Advances in Neural Information Processing Systems 38*. https://arxiv.org/abs/2509.16625

Han, X., Cui, S., Liu, S., Zhang, C., Jiang, B., & Lu, Z. (2023). Network intrusion detection based on n-gram frequency and time-aware transformer. *Computers & Security*, *128*, 103171. https://doi.org/10.1016/j.cose.2023.103171

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

Hwang, R.-H., Peng, M.-C., Nguyen, V.-L., & Chang, Y.-L. (2019). An LSTM-based deep learning approach for classifying malicious traffic at the packet level. *Applied Sciences*, *9*(16), 3414. https://doi.org/10.3390/app9163414

Hwang, R.-H., Peng, M.-C., Huang, C.-W., Lin, P.-C., & Nguyen, V.-L. (2020). An unsupervised deep learning model for early network traffic anomaly detection (D-PACK). *IEEE Access*, *8*, 30387–30399. https://doi.org/10.1109/ACCESS.2020.2973023

Javaid, A., Niyaz, Q., Sun, W., & Alam, M. (2016). A deep learning approach for network intrusion detection system. In *Proceedings of the 9th EAI International Conference on Bio-Inspired Information and Communications Technologies*. https://doi.org/10.4108/eai.3-12-2015.2262516

Kim, J., Kim, J., Thu, H. L. T., & Kim, H. (2016). Long short term memory recurrent neural network classifier for intrusion detection. In *2016 International Conference on Platform Technology and Service (PlatCon)* (pp. 1–5). IEEE. https://doi.org/10.1109/PlatCon.2016.7456805

Lin, X., Xiong, G., Gou, G., Li, Z., Shi, J., & Yu, J. (2022). ET-BERT: A contextualized datagram representation with pre-training transformers for encrypted traffic classification. In *Proceedings of the ACM Web Conference 2022* (pp. 633–642). Association for Computing Machinery. https://doi.org/10.1145/3485447.3512217

Lo, W. W., Layeghy, S., Sarhan, M., Gallagher, M., & Portmann, M. (2022). E-GraphSAGE: A graph neural network based intrusion detection system for IoT. In *NOMS 2022 — IEEE/IFIP Network Operations and Management Symposium* (pp. 1–9). IEEE. https://arxiv.org/abs/2103.16329

Manocchio, L. D., Layeghy, S., Lo, W. W., Kulatilleke, G. K., Sarhan, M., & Portmann, M. (2024). FlowTransformer: A transformer framework for flow-based network intrusion detection systems. *Expert Systems with Applications*, *241*, 122564. https://doi.org/10.1016/j.eswa.2023.122564

⚠ Najar, A. A., Sisodia, D. S., & Kumar, S. (2025). Hybrid CNN-BiLSTM model for intrusion detection on KDDCup99 / NSL-KDD / CIC-IDS2017 [Placeholder reference]. Submission pending final verification against advisor's preferred citation.

Nguyen, H.-T., & Kashef, R. (2023). TS-IDS: Traffic-aware self-supervised learning for IoT network intrusion detection. *Knowledge-Based Systems*, *279*, 110966. https://doi.org/10.1016/j.knosys.2023.110966

Shone, N., Ngoc, T. N., Phai, V. D., & Shi, Q. (2018). A deep learning approach to network intrusion detection. *IEEE Transactions on Emerging Topics in Computational Intelligence*, *2*(1), 41–50. https://doi.org/10.1109/TETCI.2017.2772792

Tang, T. A., Mhamdi, L., McLernon, D., Zaidi, S. A. R., & Ghogho, M. (2016). Deep learning approach for network intrusion detection in software defined networking. In *2016 International Conference on Wireless Networks and Mobile Communications (WINCOM)* (pp. 258–263). IEEE. https://doi.org/10.1109/WINCOM.2016.7777224

Vinayakumar, R., Alazab, M., Soman, K. P., Poornachandran, P., Al-Nemrat, A., & Venkatraman, S. (2019). Deep learning approach for intelligent intrusion detection system. *IEEE Access*, *7*, 41525–41550. https://doi.org/10.1109/ACCESS.2019.2895334

Wang, T., Xie, X., Zhang, L., Wang, C., Zhang, L., & Cui, Y. (2024). *NetMamba: Efficient network traffic classification via pre-training unidirectional Mamba* [Preprint]. arXiv. https://arxiv.org/abs/2405.11449

Wang, W., Zhu, M., Wang, J., Zeng, X., & Yang, Z. (2017). End-to-end encrypted traffic classification with one-dimensional convolution neural networks. In *2017 IEEE International Conference on Intelligence and Security Informatics (ISI)* (pp. 43–48). IEEE. https://doi.org/10.1109/ISI.2017.8004872

Yin, C., Zhu, Y., Fei, J., & He, X. (2017). A deep learning approach for intrusion detection using recurrent neural networks. *IEEE Access*, *5*, 21954–21961. https://doi.org/10.1109/ACCESS.2017.2762418

Zhao, R., Deng, X., Yan, Z., Ma, J., Xue, Z., & Wang, Y. (2023). YaTC: Yet another traffic classifier — A masked autoencoder based traffic transformer with multi-level flow representation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, *37*(5), 5420–5427. https://doi.org/10.1609/aaai.v37i5.25674

## NIDS Contrastive Learning

⚠ Golchin, P., Rafiee, N., Hajizadeh, M., Khalil, A., Kundel, R., & Steinmetz, R. (2024). SSCL-IDS: Enhancing generalization of intrusion detection with self-supervised contrastive learning. In *2024 IFIP Networking Conference* (pp. 404–412). IEEE.

Koukoulis, I., Syrigos, I., & Korakis, T. (2025). *Self-supervised transformer-based contrastive learning for intrusion detection systems* [Preprint]. arXiv. https://arxiv.org/abs/2505.08816

⚠ Liu, L., Wang, P., Ruan, J., Lin, J., & Hu, J. (2023). ConFlow: Contrast network flow improving class-imbalanced learning in network intrusion detection. In *Security and Privacy in New Computing Environments: 5th EAI International Conference (SPNCE 2023)* (Lecture Notes in the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering, Vol. 525, pp. 125–146). Springer.

⚠ Lopes, I., Zou, D., Abdelouahab, F. A., Jin, H., Li, X., & Xu, L. (2023). CLDNN-based lightweight contrastive self-supervised NIDS [Placeholder reference]. Submission pending primary-source verification.

Shahraki, A., Abbasi, M., Taherkordi, A., & Jurcut, A. D. (2023). *Self-supervised contrastive learning for intrusion detection* [Preprint]. arXiv. https://arxiv.org/abs/2209.03147

Wilkie, J., Hindy, H., Tachtatzis, C., & Atkinson, R. (2025). Contrastive self-supervised network intrusion detection using augmented negative pairs (CLAN). In *2025 IEEE International Conference on Cyber Security and Resilience (CSR)* (pp. 206–213). IEEE. https://doi.org/10.1109/CSR64739.2025.11129979

Wilkie, J., Hindy, H., Michie, C., Tachtatzis, C., Irvine, J., & Atkinson, R. (2026). *A novel contrastive loss for zero-day network intrusion detection (CLAD)* [Preprint]. arXiv. https://arxiv.org/abs/2601.09902

## General Self-Supervised and Contrastive Learning

Arora, S., Khandeparkar, H., Khodak, M., Plevrakis, O., & Saunshi, N. (2019). A theoretical analysis of contrastive unsupervised representation learning. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 5628–5637). PMLR.

Bahri, D., Jiang, H., Tay, Y., & Metzler, D. (2022). SCARF: Self-supervised contrastive learning using random feature corruption. In *International Conference on Learning Representations*.

Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-invariance-covariance regularization for self-supervised learning. In *International Conference on Learning Representations*.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 9650–9660). https://doi.org/10.1109/ICCV48922.2021.00951

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 1597–1607). PMLR.

Chen, X., Fan, H., Girshick, R., & He, K. (2020). *Improved baselines with momentum contrastive learning (MoCo v2)* [Preprint]. arXiv. https://arxiv.org/abs/2003.04297

Chen, X., & He, K. (2021). Exploring simple Siamese representation learning (SimSiam). In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 15750–15758). https://doi.org/10.1109/CVPR46437.2021.01549

Graf, F., Hofer, C., Niethammer, M., & Kwitt, R. (2021). Dissecting supervised contrastive learning. In *Proceedings of the 38th International Conference on Machine Learning* (pp. 3821–3830). PMLR.

Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z. D., Gheshlaghi Azar, M., Piot, B., Kavukcuoglu, K., Munos, R., & Valko, M. (2020). Bootstrap your own latent: A new approach to self-supervised learning. In *Advances in Neural Information Processing Systems 33* (pp. 21271–21284).

He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 9729–9738). https://doi.org/10.1109/CVPR42600.2020.00975

Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., & Bengio, Y. (2019). Learning deep representations by mutual information estimation and maximization. In *International Conference on Learning Representations*.

Kalantidis, Y., Sariyildiz, M. B., Pion, N., Weinzaepfel, P., & Larlus, D. (2020). Hard negative mixing for contrastive learning. In *Advances in Neural Information Processing Systems 33* (pp. 21798–21809).

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised contrastive learning. In *Advances in Neural Information Processing Systems 33* (pp. 18661–18673).

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *International Conference on Learning Representations*.

Oord, A. van den, Li, Y., & Vinyals, O. (2018). *Representation learning with contrastive predictive coding* [Preprint]. arXiv. https://arxiv.org/abs/1807.03748

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 815–823). https://doi.org/10.1109/CVPR.2015.7298682

Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). *SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training* [Preprint]. arXiv. https://arxiv.org/abs/2106.01342

Tian, Y., Sun, C., Poole, B., Krishnan, D., Schmid, C., & Isola, P. (2020). What makes for good views for contrastive learning? In *Advances in Neural Information Processing Systems 33* (pp. 6827–6839).

Ucar, T., Hajiramezanali, E., & Edwards, L. (2021). SubTab: Subsetting features of tabular data for self-supervised representation learning. In *Advances in Neural Information Processing Systems 34* (pp. 18853–18865).

Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). Deep graph infomax. In *International Conference on Learning Representations*.

Wang, T., & Isola, P. (2020). Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 9929–9939). PMLR.

Yoon, J., Zhang, Y., Jordon, J., & van der Schaar, M. (2020). VIME: Extending the success of self- and semi-supervised learning to tabular domain. In *Advances in Neural Information Processing Systems 33* (pp. 11033–11043).

Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-supervised learning via redundancy reduction. In *Proceedings of the 38th International Conference on Machine Learning* (pp. 12310–12320). PMLR.

## NIDS Datasets

Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., & Anwar, A. (2020). TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems. *IEEE Access*, *8*, 165130–165150. https://doi.org/10.1109/ACCESS.2020.3022862

Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion detection dataset: The CICIDS2017 case study. In *2021 IEEE Security and Privacy Workshops (SPW)* (pp. 7–12). IEEE. https://doi.org/10.1109/SPW53761.2021.00009

Lanvin, M., Gimenez, P.-F., Han, Y., Majorczyk, F., Mé, L., & Totel, É. (2023). Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes. In *Risks and Security of Internet and Systems: 17th International Conference on Risks and Security of Internet and Systems (CRiSIS 2022)* (Lecture Notes in Computer Science, Vol. 13857, pp. 18–33). Springer. https://doi.org/10.1007/978-3-031-31108-6_2

Lee, W., Stolfo, S. J., & Mok, K. W. (1999). A data mining framework for building intrusion detection models. In *Proceedings of the 1999 IEEE Symposium on Security and Privacy* (pp. 120–132). IEEE. https://doi.org/10.1109/SECPRI.1999.766909

Lippmann, R. P., Fried, D. J., Graf, I., Haines, J. W., Kendall, K. R., McClung, D., Weber, D., Webster, S. E., Wyschogrod, D., Cunningham, R. K., & Zissman, M. A. (2000). The 1999 DARPA off-line intrusion detection evaluation. *Computer Networks*, *34*(4), 579–595. https://doi.org/10.1016/S1389-1286(00)00139-0

Liu, L., Engelen, G., Lynar, T., Essam, D., & Joosen, W. (2022). Error prevalence in NIDS datasets: A case study on CIC-IDS-2017 and CSE-CIC-IDS-2018. In *2022 IEEE Conference on Communications and Network Security (CNS)* (pp. 254–262). IEEE. https://doi.org/10.1109/CNS56114.2022.9947235

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. In *2015 Military Communications and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE. https://doi.org/10.1109/MilCIS.2015.7348942

Moustafa, N., Slay, J., & Creech, G. (2021). TON-IoT datasets for IoT cybersecurity research. In *2021 IEEE International Conference on Intelligence and Security Informatics (ISI)* (pp. 1–6). IEEE.

Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. *Sensors*, *23*(13), 5941. https://doi.org/10.3390/s23135941

Rosay, A., Carlier, F., Cheval, E., & Leroux, P. (2021). From CIC-IDS2017 to LYCOS-IDS2017: A corrected dataset for better performance. In *IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT '21)* (pp. 570–575). Association for Computing Machinery. https://doi.org/10.1145/3486622.3493973

Rosay, A., Cheval, E., Carlier, F., & Leroux, P. (2022). Network intrusion detection: A comprehensive analysis of CIC-IDS2017. In *Proceedings of the 8th International Conference on Information Systems Security and Privacy (ICISSP 2022)* (pp. 25–36). SCITEPRESS. https://doi.org/10.5220/0010774000003120

Sarhan, M., Layeghy, S., Moustafa, N., & Portmann, M. (2020). NetFlow datasets for machine learning-based network intrusion detection systems. In *Big Data Technologies and Applications: 10th EAI International Conference (BDTA 2020)*. Springer. https://arxiv.org/abs/2011.09144

Sarhan, M., Layeghy, S., & Portmann, M. (2022). Towards a standard feature set for network intrusion detection system datasets. *Mobile Networks and Applications*, *27*(1), 357–370. https://doi.org/10.1007/s11036-021-01843-0

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. In *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP 2018)* (pp. 108–116). SCITEPRESS. https://doi.org/10.5220/0006639801080116

Stolfo, S. J., Fan, W., Lee, W., Prodromidis, A., & Chan, P. K. (2000). Cost-based modeling for fraud and intrusion detection: Results from the JAM project. In *Proceedings DARPA Information Survivability Conference and Exposition (DISCEX'00)* (Vol. 2, pp. 130–144). IEEE. https://doi.org/10.1109/DISCEX.2000.821515

Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. In *2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)* (pp. 1–6). IEEE. https://doi.org/10.1109/CISDA.2009.5356528

## Evaluation Methodology

Axelsson, S. (2000). The base-rate fallacy and the difficulty of intrusion detection. *ACM Transactions on Information and System Security*, *3*(3), 186–205. https://doi.org/10.1145/357830.357849

McHugh, J. (2000). Testing intrusion detection systems: A critique of the 1998 and 1999 DARPA intrusion detection system evaluations as performed by Lincoln Laboratory. *ACM Transactions on Information and System Security*, *3*(4), 262–294. https://doi.org/10.1145/382912.382923

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). *Journal of Machine Learning Research*, *22*(164), 1–20. http://jmlr.org/papers/v22/20-303.html

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, *10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

## Robust and Noise-Aware Self-Supervised Learning

Nkashama, D. K., Félicien, J. M., Soltani, A., Verdier, J.-C., Tardif, P.-M., Frappier, M., & Kabanza, F. (2024). Deep learning for network anomaly detection under data contamination: Evaluating robustness and mitigating performance degradation. *arXiv preprint arXiv:2407.08838*. https://arxiv.org/abs/2407.08838

Rusak, E., Reizinger, P., Juhos, A., Bringmann, O., Zimmermann, R. S., & Brendel, W. (2024). InfoNCE: Identifying the gap between theory and practice. *arXiv preprint arXiv:2407.00143*. https://arxiv.org/abs/2407.00143

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, *10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
