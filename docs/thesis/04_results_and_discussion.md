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
