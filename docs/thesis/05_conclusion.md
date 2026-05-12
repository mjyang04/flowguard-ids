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
