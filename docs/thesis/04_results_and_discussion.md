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
