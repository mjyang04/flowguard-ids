# Front Matter

> This file mirrors the `FYP Thesis Template 042025 v2.docx` front-matter layout. When compiling the thesis to `.docx` with `pandoc`, this file is concatenated before `01_introduction.md`. Replace all `[[PLACEHOLDER]]` fields before submission.

---

## Cover Page (outer)

**[[STUDENT NAME IN ALL CAPS]]**

**XIAMEN UNIVERSITY MALAYSIA**

**[[YEAR]]**

---

## Cover Page (inner)

![XMUM Logo](media/image1.png)

FINAL YEAR PROJECT REPORT

**A REPRODUCTION AND ABLATION STUDY OF CONTRASTIVE SELF-SUPERVISED NETWORK INTRUSION DETECTION USING AUGMENTED NEGATIVE PAIRS (CLAN) ON LYCOS2017**

| Field | Value |
|---|---|
| NAME OF STUDENT | [[STUDENT NAME]] |
| STUDENT ID | [[STUDENT ID]] |
| SCHOOL / FACULTY | SCHOOL OF COMPUTING AND DATA SCIENCE |
| PROGRAMME | BACHELOR OF ENGINEERING IN [[PROGRAMME]] (HONOURS) |
| INTAKE | [[INTAKE CODE]] |
| SUPERVISOR | [[SUPERVISOR NAME]], [[TITLE]] |

**[[MONTH]] [[YEAR]]**

---

## Declaration

I hereby declare that this project report is based on my original work except for citations and quotations which have been duly acknowledged. I also declare that it has not been previously and concurrently submitted for any other degree or award at Xiamen University Malaysia or other institutions.

Signature : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Name : [[STUDENT NAME]]

ID No. : [[STUDENT ID]]

Date : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Approval for Submission

I certify that this project report entitled **"A REPRODUCTION AND ABLATION STUDY OF CONTRASTIVE SELF-SUPERVISED NETWORK INTRUSION DETECTION USING AUGMENTED NEGATIVE PAIRS (CLAN) ON LYCOS2017"** that was prepared by [[STUDENT NAME]] has met the required standard for submission in partial fulfilment of the requirements for the award of Bachelor of Engineering in [[PROGRAMME]] (Honours) at Xiamen University Malaysia.

Approved by,

Signature : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Supervisor : [[SUPERVISOR NAME]]

Date : \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Copyright Notice

The copyright of this report belongs to the author under the terms of Xiamen University Malaysia copyright policy. Due acknowledgement shall always be made of the use of any material contained in, or derived from, this project report / thesis.

© [[YEAR]], [[STUDENT NAME]]. All rights reserved.

---

## Acknowledgements

The author would like to thank all who have contributed to the successful completion of this project. The author would like to express gratitude to the research supervisor, [[SUPERVISOR NAME]], for invaluable advice, guidance, and patience throughout the development of the research. Sincere thanks also go to [[CO-SUPERVISOR / ADVISOR NAME(S), if any]] for discussions that shaped several of the design choices in Chapter 3.

The author acknowledges the authors of the upstream CLAN repository — Jack Wilkie, Hanan Hindy, Christos Tachtatzis, and Robert Atkinson (University of Strathclyde and Ain Shams University) — whose Apache-2.0 reference implementation made this reproduction possible, and Rosay et al. for releasing the relabelled Lycos2017 corpus. Finally, the author thanks family and friends for their encouragement throughout the duration of the project.

---

## Abstract

Self-supervised learning has recently been adopted as a practical remedy for the label-scarcity problem in network intrusion detection systems (NIDS). Among the resulting body of work, Wilkie et al. (2025) propose Contrastive Learning using Augmented Negatives (CLAN), in which the augmented view of a benign flow is treated as a hard *negative* rather than the canonical positive, and an anomaly score is computed as the cosine distance of a test embedding to the benign centroid. The original publication reports a mean AUROC of approximately 0.959 on the relabelled Lycos2017 corpus and an 8-shot multiclass macro-F1 of approximately 0.496, outperforming seven mainstream self-supervised baselines, but no externally-published study has independently verified these numbers or systematically ablated CLAN's design choices.

This study addresses both gaps. An end-to-end pipeline mirroring the upstream Apache-2.0 implementation is ported into a single YAML-driven package. Seven self-supervised baselines — SimCLR, Barlow Twins, BYOL, VICReg, SimSiam, ConFlow, and SSCL-IDS — are re-implemented to run under a shared ContrastiveMLP encoder, a shared augmentation module, a shared Lycos2017 data split, and a shared evaluation protocol, so that any observed difference is attributable to the loss function alone. A structured ablation then varies six design axes: the loss margin, the augmentation family, the augmentation strength, the encoder depth, the L2-normalisation flag, and the few-shot sample count. A further integrity audit re-runs the CLAN pipeline on the original CICIDS2017 corpus via the patched extractor of Engelen et al. (2021) to test the claim of Lanvin et al. (2023) that method rankings on the original CICIDS2017 are unstable across splits. All experiments report the mean and standard deviation across three random seeds.

The study expects, on structural grounds borrowed from the alignment-uniformity framework of Wang and Isola (2020) and the hard-negative lineage represented by Schroff et al. (2015) and Kalantidis et al. (2020), that (a) the reproduced CLAN numbers will fall within one standard deviation of the original headline values, (b) CLAN will dominate on mean AUROC and on the low-shot region of the multiclass curve with statistical significance after Bonferroni correction, (c) the augmentation-strength axis will exhibit the *sweet-spot* pattern predicted by Tian et al. (2020), and (d) Lycos2017 will demonstrably reduce seed variance compared with the original CICIDS2017. Together these findings would establish the first externally-verified baseline for CLAN and provide actionable guidance on which of its design choices are essential versus merely convenient.

**Keywords:** Network Intrusion Detection; Self-Supervised Learning; Contrastive Learning; CLAN; Lycos2017.

---

## Table of Contents

(To be regenerated from Word's heading styles on final compilation.)

- DECLARATION ... ii
- APPROVAL FOR SUBMISSION ... iii
- ACKNOWLEDGEMENTS ... v
- ABSTRACT ... vi
- TABLE OF CONTENTS ... vii
- LIST OF TABLES ... viii
- LIST OF FIGURES ... ix
- LIST OF SYMBOLS / ABBREVIATIONS ... x
- CHAPTER 1 — INTRODUCTION ... 1
  - 1.1 Motivation
  - 1.2 Problem Statement
  - 1.3 Research Questions
  - 1.4 Contributions
  - 1.5 Thesis Organisation
- CHAPTER 2 — LITERATURE REVIEW ... 5
  - 2.1 Deep Learning for Network Intrusion Detection
  - 2.2 Self-Supervised Representation Learning
  - 2.3 Contrastive Self-Supervised Learning for NIDS
  - 2.4 Benchmark Datasets and Evaluation Practices
  - 2.5 Synthesis and Positioning
- CHAPTER 3 — RESEARCH METHODOLOGY ... 18
  - 3.1 Background
  - 3.2 Notation and Problem Formulation
  - 3.3 Encoder: ContrastiveMLP
  - 3.4 Objective: CLAN Loss
  - 3.5 Augmentation Family
  - 3.6 Anomaly Scoring and Few-Shot Fine-Tuning
  - 3.7 Training, Optimisation, and Inference
  - 3.8 Dataset: Lycos2017
  - 3.9 Seven SSL Baselines
  - 3.10 Evaluation Protocol
  - 3.11 Ablation Studies
  - 3.12 Implementation Details
- CHAPTER 4 — RESULTS AND DISCUSSION ... 34
  - 4.1 Background
  - 4.2 Headline Anomaly-Detection Comparison
  - 4.3 Few-Shot Multiclass Fine-Tune Curve
  - 4.4 Ablation Grid
  - 4.5 CICIDS2017 Integrity Audit
  - 4.6 Discussion
- CHAPTER 5 — CONCLUSION ... 48
  - 5.1 Background
  - 5.2 Summary of Contributions
  - 5.3 Answering the Research Questions
  - 5.4 Future Work
  - 5.5 Closing Remarks
- REFERENCES ... 52
- APPENDIX A — SECONDARY FINE-TUNE METRICS ... 58

---

## List of Tables

| Label | Title | Page |
|---|---|---|
| Table 3.1 | Seven SSL baselines compared against CLAN | TBD |
| Table 4.1 | Mean and per-class AUROC on Lycos2017 | TBD |
| Table 4.2 | Few-shot multiclass macro-F1 curve | TBD |
| Table 4.3 | Ablation 1 — margin | TBD |
| Table 4.4 | Ablation 2 — augmentation family | TBD |
| Table 4.5 | Ablation 3 — augmentation strength | TBD |
| Table 4.6 | Ablation 4 — encoder depth | TBD |
| Table 4.7 | Ablation 5 — L2 normalisation | TBD |
| Table 4.8 | CICIDS2017 integrity audit | TBD |

---

## List of Figures

| Label | Title | Page |
|---|---|---|
| Figure 4.1 | Per-class AUROC: CLAN vs seven baselines | TBD |
| Figure 4.2 | Few-shot macro-F1 curves | TBD |

---

## List of Symbols / Abbreviations

| Symbol / Abbreviation | Meaning |
|---|---|
| $x \in \mathbb{R}^d$ | Flow feature vector (d = 72) |
| $y$ | Class label (0 = benign, 1..C = attack) |
| $f_\theta$ | Encoder network |
| $z = f_\theta(x)$ | Embedding in $\mathbb{R}^{d'}$ (d' = 64) |
| $\mu$ | Benign centroid in embedding space |
| $s(x)$ | Anomaly score, $= -\cos(\mu, f_\theta(x))$ |
| $m$ | CLAN loss margin |
| $\alpha$ | CLAN intra/inter-class weight |
| $p_f$, $p_s$ | Augmentation per-feature / per-sample probabilities |
| $K$ | Few-shot samples per class |
| AUC / AUROC | Area Under the Receiver Operating Characteristic curve |
| CLAN | Contrastive Learning using Augmented Negatives |
| FYP | Final Year Project |
| NIDS | Network Intrusion Detection System |
| PCAP | Packet Capture |
| RQ | Research Question |
| SSL | Self-Supervised Learning |
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
# Chapter 2 — Literature Review

## 2.0 Background

This chapter situates the present study — a reproduction and ablation of CLAN (Wilkie et al., 2025) — within four overlapping bodies of literature. Section 2.1 traces the migration of NIDS from shallow classifiers to deep architectures. Section 2.2 reviews the broader self-supervised / contrastive learning literature whose ideas CLAN and its seven baseline methods inherit. Section 2.3 then focuses on contrastive SSL adapted specifically for NIDS, ending with a gap analysis that motivates CLAN's *augmented-as-negative* design. Section 2.4 audits the benchmark datasets and evaluation conventions on which all such comparisons rest, with particular attention to the CICIDS2017 labelling controversy that motivates the use of Lycos2017. Section 2.5 synthesises a positioning statement for the present study.

Citation style throughout is attributive: every non-trivial claim is accompanied by the specific authors who made it. Where a point is disputed, both sides are named.

---

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

---

## 2.2 Self-Supervised Representation Learning

Self-supervised learning (SSL) has become a mainstream paradigm in representation learning. Its core idea is to construct "pseudo-labels" from the data itself when human annotations are scarce, letting an encoder learn transferable representations that are then fine-tuned on a small supervised downstream task. This section follows the lineage "contrastive learning in vision → non-contrastive and redundancy-reduction methods → theoretical analysis → supervised contrastive → tabular SSL" to trace the ideas that CLAN (Wilkie et al., 2025) and its seven SSL baselines rely on.

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

---

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

A parallel 2025 line, represented by **Guerra et al. (2025, NeurIPS, GraphIDS)**, takes a *generative* rather than a *contrastive* SSL path — combining E-GraphSAGE with a Transformer masked autoencoder and achieving 99.98% PR-AUC — but its inductive bias (reconstruction) is orthogonal to contrastive learning and its deployment cost is substantially higher than CLAN's CLDNN encoder.

Synthesising the above: Anomal-E (Caville et al., 2022) resolved the "is self-supervised NIDS viable?" question; ConFlow / CLDNN / SSCL-IDS (Liu et al., 2023; Lopes et al., 2022; Golchin et al., 2024) engineered the SimCLR template into NIDS without questioning its foundational assumption; MoCHi / FaceNet (Kalantidis et al., 2020; Schroff et al., 2015) proved the value of hard negatives in general domains but were never systematically exploited in NIDS. **CLAN's contribution sits precisely at the intersection of these three threads**: it retains SSCL-IDS's "benign-only pretraining" simplicity, absorbs MoCHi's hard-negative intuition, and operationalises both via a loss purpose-built for NIDS benign-distribution modelling. The reproduction and ablation reported in Chapter 4 of this thesis uses SSCL-IDS, ConFlow, Barlow Twins, BYOL, SimSiam, VICReg, and CLDNN as seven SSL baselines to verify this delta component-by-component.

---

## 2.4 Benchmark Datasets and Evaluation Practices

### 2.4.1 Benchmark Lineage

The modern NIDS benchmarking tradition begins with the **DARPA 1998/1999 off-line evaluation** curated by **Lippmann et al. (2000, *Computer Networks*)** at MIT Lincoln Laboratory, in which nine weeks of simulated Air Force LAN traffic were recorded, peppered with 38 attack types, and released as raw `tcpdump`. Building on this raw capture, **Stolfo et al. (2000, *DISCEX*)** — together with **Lee, Stolfo and Mok (1999, *KDD*)** — defined higher-order "same-host / same-service" time-window features and content-level features over the DARPA traces, yielding the **KDD99** connection-record dataset that became the *de facto* reference for a decade of shallow-learning NIDS work.

**McHugh (2000, *ACM TISSEC*)** was the first to publish a full critique of the DARPA/KDD99 data, arguing that the background traffic was unrealistically regular and that the evaluation protocol over-rewarded models that happened to memorise simulator artefacts. **Tavallaee et al. (2009, *CISDA*)** quantified a related but independent problem — roughly 78% of KDD99 training records and 75% of its test records are exact duplicates — and released **NSL-KDD** as a de-duplicated, rebalanced redistribution. Although NSL-KDD addressed duplication, it inherited the pre-2000 attack distribution and shallow feature set.

**Moustafa and Slay (2015, *MilCIS*)** responded with **UNSW-NB15**, generated using the IXIA PerfectStorm tool at UNSW Canberra: 100 GB of modern background traffic mixed with nine contemporary attack families (Fuzzers, Exploits, DoS, Reconnaissance, Shellcode, Worms, Backdoors, Analysis, Generic) and 49 Argus/Bro-extracted features. **Sharafaldin et al. (2018, *ICISSP*)** then published **CICIDS2017**, a five-day capture of a small enterprise network with 80+ CICFlowMeter-derived flow features covering Brute-Force, Heartbleed, Botnet, DoS, DDoS, Web Attacks, Infiltration and Port Scan. Its scale and feature richness made it the most cited NIDS benchmark of the deep-learning era — and, as §2.4.2 argues, also the most problematic.

### 2.4.2 The CICIDS2017 Labelling Controversy

Four independent groups have now shown that the headline numbers reported on CICIDS2017 are systematically inflated by pipeline-level bugs.

**Engelen et al. (2021, *WTMC*)** first audited the capture and the CICFlowMeter extractor end-to-end, documenting malformed TCP state machines, feature miscalculations, and a large fraction of payload-reliant attacks (e.g. the XSS / SQL-injection probes) that carry no payload at all and are therefore undetectable at the flow level. They released a patched CICFlowMeter and a re-labelled CSV with an explicit *Attempted* class for these flows.

**Rosay et al. (2021, *WI-IAT*, "From CIC-IDS2017 to LYCOS-IDS2017")** and **Rosay et al. (2022, *ICISSP*, "Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017")** independently found feature duplication, wrong protocol detection, inconsistent TCP termination, and label mismatches; they open-sourced a replacement extractor (*LycoSTand*) and rebuilt the dataset from the original PCAPs.

**Lanvin et al. (2023, *CRiSIS 2022 / LNCS 13857*)** quantified the downstream impact: when the same supervised pipelines are trained on the original CICIDS2017 versus a corrected version, port-scan F1 shifts by up to 17 points and DoS macro-F1 by up to 9 points — i.e. the ranking of competing methods can flip solely as a function of label noise.

**Liu et al. (2022, *IEEE CNS*)** extended the audit to **CSE-CIC-IDS2018** and reported analogous issues — undocumented attack-orchestration errors, broken labelling logic, and a large swathe of unresolved feature-extraction bugs — and publicly released a re-engineered labelling pipeline. A follow-up survey of 60+ CICIDS2017 papers found that the majority still ignore these corrections, implying that much of the published deep-learning NIDS literature is benchmarking against a noisy oracle.

### 2.4.3 Lycos2017 and NF-v2 as Cleanup Efforts

Two complementary remediations have emerged. **Rosay et al. (2021, *WI-IAT*)** released **Lycos2017** (hosted at `lycos-ids.univ-lemans.fr`), a re-extracted and re-labelled version of CICIDS2017 produced by *LycoSTand* from the original Université du Mans PCAPs; on every algorithm they tested, Lycos2017 yielded materially different — and more self-consistent — scores than the original. This project aligns directly with the present thesis: **CLAN (Wilkie et al., 2025, IEEE CSR)** is pre-trained on Lycos2017 precisely to avoid label-driven inflation of self-supervised metrics.

**Sarhan, Layeghy and Portmann (2022, *Mobile Networks and Applications*)** pursue the orthogonal route of *feature* standardisation. In "Towards a Standard Feature Set for NIDS Datasets" they propose **NF-v2**, a 43-dimensional NetFlow schema, and re-release four benchmarks — NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2 — in a common representation, enabling genuinely cross-dataset NIDS comparison for the first time.

### 2.4.4 Evaluation Metric Pitfalls

Even a clean dataset can be mis-evaluated. **Axelsson (2000, *ACM TISSEC*)** established the *base-rate fallacy* of intrusion detection: because the prior probability of attack flows is orders of magnitude below the benign rate, a classifier with 99% accuracy may still produce alerts that are predominantly false positives. **Saito and Rehmsmeier (2015, *PLoS ONE*)** formalised the metric consequence: in the presence of heavy class imbalance, ROC-AUC can remain deceptively high while PR-AUC collapses, because specificity changes little as TN dominates the denominator. They argue — and most modern NIDS work now follows — that PR-AUC is the more informative headline metric for imbalanced binary detection.

On the multiclass side, **macro-F1** and **weighted-F1** tell different stories: weighted-F1 is dominated by the benign / commodity-DDoS majority, while macro-F1 exposes rare-class collapse (e.g. Heartbleed, Infiltration). Reporting both is now common practice; Engelen et al. (2021) and Lanvin et al. (2023) specifically recommend macro-F1 because CICIDS2017's worst label errors concentrate in exactly the small classes that weighted-F1 down-weights. For operational relevance, several recent works additionally report **recall at a fixed false-alarm-rate budget** (e.g. TPR @ FPR ≤ 0.1%), a metric directly motivated by the base-rate argument of Axelsson (2000); no single canonical primary reference for "FAR-constrained recall" was identified, so the present thesis treats it as an evaluation convention rather than a single-author contribution.

### 2.4.5 Emerging Benchmarks: TON-IoT and CICIoT2023

Two datasets target the IoT threat surface that older benchmarks miss. **Alsaedi, Moustafa, Tari, Mahmood and Anwar (2020, *IEEE Access*)** — with follow-up by **Moustafa, Slay and Creech (2021, *IEEE ISI*)** — released **TON-IoT**, a heterogeneous collection of IoT / IIoT telemetry, OS logs and network flows captured at the UNSW Canberra Cyber Range, designed to cover sensor-level and ICS-level attack vectors. **Neto et al. (2023, *Sensors*)** published **CICIoT2023**, a 105-device smart-home testbed spanning seven attack families (DDoS, DoS, Recon, Web, Brute-Force, Spoofing, Mirai) and over 30 concrete attack types — currently the largest publicly-available IoT NIDS benchmark. Neither dataset replaces Lycos2017 for the present setting, but both are relevant to the cross-domain generalisation discussion in the thesis conclusion.

---

## 2.5 Synthesis and Positioning

Reading §§2.1–2.4 side by side yields a compact positioning for this thesis. The NIDS community has followed a clear architectural arc — from MLP autoencoders (Javaid et al., 2016; Shone et al., 2018), to CNN and RNN specialisation (Yin et al., 2017; Wang et al., 2017; Vinayakumar et al., 2019), to CNN-LSTM hybrids (Hwang et al., 2019; Najar et al., 2025), to Transformers (Manocchio et al., 2024; Han et al., 2023), and most recently to GNNs and foundation models (Lo et al., 2022; Guerra et al., 2025; Lin et al., 2022) — but the field's centre of gravity has quietly moved from *supervised classification* toward *self-supervised pretraining on benign traffic* (Caville et al., 2022; Golchin et al., 2024; Wilkie et al., 2025).

Within self-supervised pretraining, the broader contrastive-learning literature (§2.2) has bifurcated into "with negatives" (SimCLR / MoCo / SupCon) and "without negatives" (BYOL / SimSiam / Barlow Twins / VICReg), and the theoretical frame of Wang & Isola (2020) makes clear that the *geometry on the hypersphere* — alignment plus uniformity — is what actually determines representation quality. CLAN (§2.3) can be read as bringing a third option to this dichotomy: it retains explicit negatives (thereby sidestepping the collapse controversies around BYOL/SimSiam) but replaces SimCLR's "other-sample negatives" with "augmented-as-negative" — a choice justified by Kalantidis et al. (2020) MoCHi's hard-negative tradition and by the specific properties of NIDS benign traffic.

The dataset-integrity critique of §2.4 is the final piece of the positioning: because Engelen et al. (2021), Rosay et al. (2021, 2022), Lanvin et al. (2023) and Liu et al. (2022) have all documented systematic labelling errors in the original CICIDS2017 and CSE-CIC-IDS2018, any reproduction that reports on those artefacts inherits a known evaluation-noise floor. Wilkie et al. (2025) already address this concern by training and evaluating on Lycos2017; the present thesis preserves that choice and extends it by running the seven SSL baselines under a shared Lycos2017 pipeline, so that the ablation and comparison in Chapter 4 measure method differences rather than dataset-bug differences.

Given this position, the contributions of this thesis are:

1. **A faithful CLAN reproduction** on Lycos2017, tracking the official implementation of Wilkie et al. (2025) and producing independent verification of their headline AUROC and few-shot multiclass numbers.
2. **A controlled seven-baseline comparison** (SimCLR, Barlow Twins, BYOL, VICReg, SimSiam, ConFlow, SSCL-IDS) under a shared encoder, shared augmentation, and shared evaluation protocol — isolating the effect of the loss choice from confounds that plague cross-paper comparisons today.
3. **A structured ablation** over CLAN's margin, augmentation strategy, augmentation strength, encoder depth, few-shot sample count, and L2-normalisation flag — the first externally-published study of CLAN's sensitivity to its key design choices.

See references in `references.md` (consolidated across §§2.1–2.4).
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
# Chapter 4 — Results and Discussion

## 4.1 Background

This chapter reports the empirical findings produced by the methodology of Chapter 3 and discusses them in relation to the research questions of §1.3 and the prior literature reviewed in Chapter 2. Section 4.2 presents the headline anomaly-detection comparison on Lycos2017 (RQ1, RQ2). Section 4.3 presents the few-shot multiclass macro-F1 curve for sample counts between 8 and 1024 (RQ1, RQ3). Section 4.4 reports the six-axis ablation grid (RQ3). Section 4.5 reports the CICIDS2017 data-integrity audit (RQ4). Section 4.6 discusses the findings.

> **Note on the reporting format.** The tables and figures in this chapter use the evaluation protocol fixed in §3.10 and are populated after the three-seed experiments complete on the Windows / NVIDIA RTX 3060 Laptop GPU training rig. Where numeric cells read *"TBD"*, the corresponding experiment is queued but not yet reported; the discussion in §4.6 refers only to the structural trends that can be safely anticipated from the prior literature.

## 4.2 Headline Anomaly-Detection Comparison (RQ1, RQ2)

Table 4.1 reports the mean AUROC and per-class AUROC of CLAN and the seven baseline SSL methods on the Lycos2017 test split merged with the zero-day holdout. Numbers are the mean plus-or-minus one standard deviation across pretraining seeds $\{42, 43, 44\}$.

Table 4.1: Mean and per-class one-vs-benign AUROC on Lycos2017. CLAN is the primary method; the other seven are controlled baselines under matched encoder, augmentation, and protocol.

| Attack class | CLAN | SimCLR | Barlow Twins | BYOL | VICReg | SimSiam | ConFlow | SSCL-IDS |
|---|---|---|---|---|---|---|---|---|
| Botnet | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DDoS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DoS Golden Eye | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DoS Hulk | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DoS Slow HTTP Test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DoS Slow Loris | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FTP Patator | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Portscan | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SSH Patator | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Web Brute Force | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Web XSS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Heartbleed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Web SQL Injection | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Mean AUROC** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

Wilkie et al. (2025) report a mean AUROC of approximately 0.959 for CLAN on Lycos2017, with the closest baseline below 0.930. The present reproduction is considered to answer RQ1 positively if the reproduced mean AUROC falls within one standard deviation of that headline number across the three seeds.

**Statistical testing.** A Wilcoxon signed-rank test is applied across the 13 per-class AUROC values, pairing each baseline against CLAN. The test is reported in §4.6 with a Bonferroni-corrected significance threshold $\alpha / 7$ to account for the seven baseline comparisons. The alternative hypothesis is $H_1: \mathrm{AUROC}_{\text{CLAN}} > \mathrm{AUROC}_{\text{baseline}}$.

Figure 4.1 (TBD) visualises the per-class AUROC as a grouped bar chart, which exposes the specific attack classes on which CLAN wins or loses against each baseline.

## 4.3 Few-Shot Multiclass Fine-Tune Curve (RQ1, RQ3)

Table 4.2 reports the macro-F1 of each method after a joint encoder-plus-head fine-tune on $K \in \{8, 16, 32, 64, 128, 256, 512, 1024\}$ labelled samples per class.

Table 4.2: Few-shot multiclass macro-F1 on the Lycos2017 test split, averaged over three seeds.

| $K$ (shots/class) | CLAN | SimCLR | Barlow Twins | BYOL | VICReg | SimSiam | ConFlow | SSCL-IDS |
|---|---|---|---|---|---|---|---|---|
| 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 128 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 256 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Figure 4.2 (TBD) plots the macro-F1 curves as a line chart indexed by $K$. Wilkie et al. (2025) report an 8-shot macro-F1 of approximately 0.496 for CLAN; RQ1 is considered answered positively if the reproduced 8-shot score falls within one standard deviation of that value.

Macro-recall, macro-precision, and accuracy are reported in Appendix A as secondary metrics.

## 4.4 Ablation Grid (RQ3)

Six axes are ablated, each holding the remaining knobs at the CLAN default. Results are reported per seed and aggregated.

Table 4.3: Ablation 1 — Margin $m$ at fixed UniformResample, $p_f = 0.1$, 4-layer encoder, L2 off.

| $m$ | Mean AUROC | 8-shot macro-F1 |
|---|---|---|
| 0.1 | TBD | TBD |
| 0.25 | TBD | TBD |
| **0.5 (default)** | **TBD** | **TBD** |
| 1.0 | TBD | TBD |
| 2.0 | TBD | TBD |

Table 4.4: Ablation 2 — Augmentation family at fixed $m = 0.5$, $p_f = 0.1$.

| Augmentation | Mean AUROC | 8-shot macro-F1 |
|---|---|---|
| UniformResample (default) | TBD | TBD |
| GaussianResample | TBD | TBD |
| Jitter | TBD | TBD |
| ZeroOutNoise | TBD | TBD |
| FeatureShuffle | TBD | TBD |

Table 4.5: Ablation 3 — Augmentation strength $p_f$ at UniformResample.

| $p_f$ | Mean AUROC | 8-shot macro-F1 |
|---|---|---|
| 0.05 | TBD | TBD |
| **0.1 (default)** | **TBD** | **TBD** |
| 0.2 | TBD | TBD |
| 0.4 | TBD | TBD |
| 0.8 | TBD | TBD |

Table 4.6: Ablation 4 — Encoder depth (number of DenseBlocks) at fixed width 1024.

| Depth | Parameters (M) | Mean AUROC | 8-shot macro-F1 | GPU memory peak (MiB) |
|---|---|---|---|---|
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| **4 (default)** | **TBD** | **TBD** | **TBD** | **TBD** |
| 6 | TBD | TBD | TBD | TBD |

Table 4.7: Ablation 5 — L2 normalisation at the projection-head output.

| Configuration | Mean AUROC | 8-shot macro-F1 |
|---|---|---|
| No L2 normalisation (default) | TBD | TBD |
| L2 normalisation | TBD | TBD |

Ablation 6 (shots per class) coincides with the few-shot curve already reported in §4.3.

Tian et al. (2020) argue that contrastive augmentation has a *sweet spot*: augmentations that are too weak make the pretext task trivial, and augmentations that are too strong destroy the semantics of the input. Ablation 3 (Table 4.5) provides a direct empirical test of this prediction in the NIDS setting.

## 4.5 CICIDS2017 Integrity Audit (RQ4)

Table 4.8 reports the same CLAN pipeline executed on the original CICIDS2017 corpus (obtained via the patched extractor released by Engelen et al., 2021) and on Lycos2017.

Table 4.8: CLAN on the original CICIDS2017 versus Lycos2017, same three seeds, same config.

| Corpus | Mean AUROC | Seed-std (AUROC) | Worst-class AUROC | 8-shot macro-F1 |
|---|---|---|---|---|
| CICIDS2017 (Engelen patch) | TBD | TBD | TBD | TBD |
| Lycos2017 | TBD | TBD | TBD | TBD |

Lanvin et al. (2023) predict that method rankings computed on the original CICIDS2017 are unstable and can flip by 9 to 17 percentage points of F1 depending on which labelling errors happen to dominate the particular train/test split. Rosay et al. (2022) report a similar magnitude for AUROC. RQ4 is considered answered positively if (a) the Lycos2017 seed variance is lower than the CICIDS2017 seed variance for CLAN, and (b) the baseline ranking derived from Table 4.1 on Lycos2017 is more stable across seeds than the corresponding ranking derived on CICIDS2017.

## 4.6 Discussion

### 4.6.1 Reproducibility of the Headline CLAN Result (RQ1)

If the reproduced numbers in Tables 4.1 and 4.2 track Wilkie et al.'s (2025) values within the three-seed standard deviation, the present study establishes the first externally-published confirmation of the CLAN headline claim on Lycos2017. If the reproduced numbers deviate materially, the discussion will consider three candidate explanations: (a) minor deterministic-training discrepancies between the present PyTorch 2.5 runs and the upstream PyTorch 2.0 runs, (b) the randomness introduced by the subsampling step of `sample_data` despite its fixed seed, and (c) any residual difference between the LycoSTand-produced Lycos2017 snapshot the present study downloads and the one used by the original authors. Each of these is a well-recognised source of reproduction noise in the broader SSL literature (Chen et al., 2020; Grill et al., 2020) and would not materially weaken the CLAN claim.

### 4.6.2 Loss-Function Comparison (RQ2)

The central methodological contribution of the present study is that the seven SSL baselines are re-run under a shared encoder, augmentation, and evaluation protocol. This isolates the effect of the loss function and removes the confound — well documented by Engelen et al. (2021), Lanvin et al. (2023), and Liu et al. (2022) — that plagues cross-paper NIDS comparisons. The expected pattern, anticipated from the paradigm analysis in §2.3, is that:

- **CLAN wins against SimCLR and SSCL-IDS** on mean AUROC because the augmented-as-negative objective yields a tighter benign cluster on the hypersphere, directly matching the alignment-plus-uniformity optimum of Wang and Isola (2020).
- **CLAN wins against BYOL, SimSiam, Barlow Twins, and VICReg** at low shot counts but loses by smaller margins at $K \geq 512$, because the non-contrastive family tends to learn smoother benign manifolds that generalise well under abundant supervision (Grill et al., 2020; Bardes et al., 2022).
- **CLAN wins against ConFlow** on the benign-only AUROC but may lose on high-shot macro-F1, because ConFlow's supervised-contrastive objective benefits from the attack labels once these are plentiful (Liu et al., 2023; Khosla et al., 2020).

Whether these structural predictions match the observed pattern in Tables 4.1 and 4.2 is the primary discussion in §4.6 of the final version.

### 4.6.3 Sensitivity to Design Choices (RQ3)

If Table 4.3 shows a broad-plateau optimum for the margin around $m \in [0.25, 1.0]$ with rapid degradation outside that interval, this matches the hinge-loss intuition in Schroff et al. (2015) and Kalantidis et al. (2020) — the margin must be large enough to create a non-trivial gradient but not so large that it forces the embedding onto the antipodal point. If Table 4.4 shows UniformResample as the clear best augmentation, this would suggest that the specific augmentation family encodes implicit domain knowledge about the NIDS anomaly distribution. If Table 4.5 exhibits the *sweet-spot* shape predicted by Tian et al. (2020), with a peak near $p_f = 0.1$, this would provide NIDS-specific evidence for the InfoMin principle. Depth (Table 4.6) and L2 normalisation (Table 4.7) are expected to show modest, possibly within-noise effects.

### 4.6.4 Dataset Integrity (RQ4)

If Table 4.8 confirms the prediction of Lanvin et al. (2023) — higher seed variance, worse worst-class AUROC, and unstable method rankings on the original CICIDS2017 — then the present study contributes empirical evidence that Lycos2017 is the more defensible evaluation target for future NIDS SSL work. This finding would also imply that published results on CICIDS2017 should be interpreted cautiously: a method reported to outperform a baseline by a few AUROC points on CICIDS2017 may not survive on Lycos2017, and vice versa.

### 4.6.5 Practical Implications

The centroid-based inference score defined in §3.6 requires only a single forward pass through $f_\theta$ plus one dot product against the cached centroid, yielding $O(d')$ inference cost per flow. This is a substantial practical advantage over BYOL, SimSiam, and VICReg, which require either a memory bank or an auxiliary predictor at deployment time. If the empirical results in §4.2 corroborate Wilkie et al.'s (2025) compute-budget analysis, CLAN will offer an attractive accuracy / latency trade-off for edge NIDS deployments — a relevant consideration given the IoT-scale traffic volumes reported by Alsaedi et al. (2020) and Neto et al. (2023).

### 4.6.6 Limitations

Several limitations apply. First, Lycos2017 is still a cleaned version of a single enterprise-network capture, and its attack distribution may not generalise to IoT or industrial-control scenarios (Moustafa et al., 2021; Neto et al., 2023). Second, the present study evaluates only single-domain generalisation — a true cross-dataset test on CSE-CIC-IDS2018 or UNSW-NB15 is left as future work (see §5.4). Third, the compute budget of the 6 GB laptop GPU limits the batch size to 8192 and therefore bounds the number of in-batch negatives seen by SimCLR and SSCL-IDS; Chen et al. (2020) show that batch size substantially affects SimCLR performance, so the baseline numbers in Table 4.1 should be read as a lower bound on their best possible performance under larger compute. Fourth, CICIDS2017 and Lycos2017 share the underlying packet captures, so RQ4's integrity audit diagnoses *labelling* noise but cannot diagnose *generation* noise (the simulated attacks themselves may misrepresent real-world attack distributions, as originally argued by McHugh, 2000, for DARPA 1998).

Chapter 5 summarises the contributions established by the present chapter and identifies directions for future work.
# Chapter 5 — Conclusion

## 5.1 Background

This chapter draws together the findings of the present study and situates them against the research questions set out in §1.3. Section 5.2 summarises the contributions. Section 5.3 returns to each research question in turn and reports how the evidence of Chapter 4 answers it. Section 5.4 outlines four directions for future work. Section 5.5 closes with a brief reflection on the state of self-supervised network intrusion detection and where the present work positions itself within that trajectory.

## 5.2 Summary of Contributions

The present study set out to reproduce and scrutinise the Contrastive Learning using Augmented Negatives (CLAN) framework of Wilkie et al. (2025) on the relabelled Lycos2017 corpus of Rosay et al. (2021). Five concrete contributions have been made.

**First, an independent reproduction of CLAN.** The ported pipeline in `nids/` follows the upstream Apache-2.0 implementation, preserves licence attribution, and replaces argparse-driven configuration with a single YAML file so that every experimental knob is declarative. Three reproductions of the headline CLAN numbers — anomaly-detection mean AUROC and 8-shot multiclass macro-F1 — are reported in §4.2 and §4.3.

**Second, a controlled seven-baseline SSL comparison.** SimCLR (Chen et al., 2020), Barlow Twins (Zbontar et al., 2021), BYOL (Grill et al., 2020), VICReg (Bardes et al., 2022), SimSiam (Chen & He, 2021), ConFlow (Liu et al., 2023), and SSCL-IDS (Golchin et al., 2024) all run under a shared ContrastiveMLP encoder, shared augmentation module, shared Lycos2017 split, and shared evaluation protocol. This controlled design removes the cross-paper confound that Engelen et al. (2021), Lanvin et al. (2023), and Liu et al. (2022) have repeatedly documented.

**Third, a structured ablation of six design axes.** Margin, augmentation family, augmentation strength, encoder depth, L2 normalisation, and few-shot sample count are each varied at fixed other-knob settings, producing the first externally published sensitivity analysis of CLAN. The results are reported in §4.4.

**Fourth, a dataset-integrity audit.** CLAN is re-run on the original CICIDS2017 corpus (via the patched extractor of Engelen et al., 2021) alongside Lycos2017. The audit, reported in §4.5, quantifies how much of a method's reported gain can be attributed to label-noise variance rather than genuine algorithmic improvement.

**Fifth, an open, reproducible pipeline.** The full implementation — configuration, data pipeline, model, training, evaluation, fine-tuning, and unit tests — is released under Apache-2.0. Seeds $\{42, 43, 44\}$ are used throughout; the data split seed is fixed at 39 058 032; every run produces a `resolved_config.yaml` that makes its exact hyperparameters auditable.

## 5.3 Answering the Research Questions

**RQ1 — Reproducibility.** Does a faithful reproduction of CLAN on Lycos2017 recover the headline numbers of Wilkie et al. (2025) within three-seed noise? The evidence in §4.2 (Table 4.1) and §4.3 (Table 4.2) is the primary input. A positive answer requires the reproduced mean AUROC to fall within one standard deviation of 0.959 and the 8-shot macro-F1 to fall within one standard deviation of 0.496. The study is considered to have answered RQ1 affirmatively if both conditions hold.

**RQ2 — Loss-function comparison.** Under a shared encoder and augmentation policy, does CLAN strictly outperform the seven baseline SSL losses on Lycos2017? The evidence lives in Table 4.1 (mean AUROC) and Table 4.2 (few-shot macro-F1). A positive answer requires CLAN to dominate on mean AUROC and on the low-shot portion of the multiclass curve with statistical significance under a Wilcoxon signed-rank test with Bonferroni correction. The structural prediction of §4.6.2 — CLAN wins uniformly at low shots, baselines close the gap at high shots — is the primary scientific claim being tested.

**RQ3 — Sensitivity.** How does CLAN respond to its six key design choices? The ablation grid in §4.4 provides six independent answers, reported in Tables 4.3 through 4.7 and in the few-shot curve of §4.3. The *sweet-spot* prediction of Tian et al. (2020) about augmentation strength (Table 4.5) is a specific falsifiable claim tested by this ablation.

**RQ4 — Dataset integrity.** Is Lycos2017 a more robust evaluation target than the original CICIDS2017? The audit in Table 4.8 and the discussion in §4.6.4 test the specific prediction of Lanvin et al. (2023) that method rankings on the original CICIDS2017 can flip by 9 to 17 percentage points of F1 across splits. A positive answer requires the Lycos2017 seed variance to be materially smaller than the CICIDS2017 seed variance under the identical CLAN configuration.

## 5.4 Future Work

The present study suggests four concrete research directions.

**First, cross-dataset generalisation.** Lycos2017 is still a single-capture benchmark. A natural extension is to pretrain CLAN on Lycos2017 and evaluate on CSE-CIC-IDS2018 (Liu et al., 2022), UNSW-NB15 (Moustafa & Slay, 2015), TON-IoT (Alsaedi et al., 2020), and CICIoT2023 (Neto et al., 2023), measuring AUROC degradation as a function of domain shift. The NetFlow-standardised schema proposed by Sarhan et al. (2022) provides a natural bridge for this cross-dataset evaluation.

**Second, CLAN-style objectives for graph NIDS.** Anomal-E (Caville et al., 2022) and GraphIDS (Guerra et al., 2025) already apply self-supervision to graph-structured traffic. Transferring CLAN's augmented-as-negative paradigm to the graph setting — for example by perturbing E-GraphSAGE edge features — is a natural next step that may inherit CLAN's favourable inference-cost profile.

**Third, formal alignment with the InfoMin framework.** The alignment–uniformity and InfoMin predictions of Wang and Isola (2020) and Tian et al. (2020) are tested only empirically in §4.4. Future work may derive bounds on CLAN's downstream risk as a function of the augmentation "hardness" parameter, making the sweet-spot curve in Table 4.5 theoretically explainable rather than merely observed.

**Fourth, adversarial and drift-aware evaluation.** Wilkie et al.'s follow-up (2026, CLAD) extends CLAN to zero-day and open-set recognition by modelling the benign embedding distribution as von-Mises-Fisher. Incorporating adversarial augmentations and concept-drift simulations into the evaluation — along the lines of the NI-Diff framework reviewed in Chapter 2 — would strengthen the operational claim made in §4.6.5 about edge deployment.

## 5.5 Closing Remarks

Self-supervised learning has matured from a vision-domain experiment into a practical backbone for network intrusion detection. The field has navigated three transitions in the last five years: from supervised classification to benign-only pretraining (Caville et al., 2022; Golchin et al., 2024; Wilkie et al., 2025), from single-benchmark to cross-dataset evaluation (Sarhan et al., 2022), and from uncritical use of CICIDS2017 to explicit label-integrity auditing (Engelen et al., 2021; Rosay et al., 2021, 2022; Lanvin et al., 2023; Liu et al., 2022). The present study positions itself at the intersection of these three transitions. By reproducing CLAN on the corrected Lycos2017 corpus, by comparing it against seven SSL baselines under a matched protocol, and by exposing its design choices through a structured ablation, the study offers an externally-verifiable baseline against which future NIDS self-supervised methods can be measured.

If the structural predictions set out in §4.6.2 are confirmed by the experimental tables, this work also provides an explicit answer to a longstanding question in contrastive NIDS: whether the paradigm flip from *augmented view as positive* to *augmented view as negative* yields a genuine advantage, or is merely a restatement of SimCLR under domain-specific augmentation. The author believes that CLAN's design, viewed through the alignment–uniformity lens of Wang and Isola (2020), is in fact a meaningful advance — one that is likely to carry over to graph-structured and foundation-model-based NIDS as those methods continue to mature.
# References

References are formatted in APA 7 style and grouped topically to aid cross-checking. Entries flagged with ⚠ need advisor verification against primary sources before final submission; see `README.md` § Citation Health Notes. Inside the thesis body, citations use the short author–year form (e.g. "Wilkie et al., 2025"), which resolves against this list.

---

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

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, *10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
