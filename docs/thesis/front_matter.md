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
