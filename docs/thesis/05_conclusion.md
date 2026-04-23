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
