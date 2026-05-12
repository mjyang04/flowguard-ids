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
