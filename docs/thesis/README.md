# Thesis Drafts — FlowGuard IDS (CLAN Reproduction + Dataset Audit)

Working drafts of the thesis chapters, formatted to match the XMUM FYP Thesis Template (see `FYP Thesis Template 042025 v2.docx` in this folder). Every chapter uses **attributive citations** (`X et al. (year) propose / observe / argue that…`) in **APA 7** style.

## Structure (matches FYP template)

| File | Template section | Status | Word count (approx.) |
|---|---|---|---|
| `front_matter.md` | Cover, Declaration, Approval, Copyright, Acknowledgements, Abstract, ToC, Lists | Placeholder | 900 |
| `01_introduction.md` | Chapter 1 — Introduction | **Complete draft** | 1 300 |
| `02_related_work.md` | Chapter 2 — Literature Review (§§2.1–2.5) | **Complete draft** | 5 250 |
| `03_research_methodology.md` | Chapter 3 — Research Methodology (§§3.1–3.12) | **Complete draft** | 3 400 |
| `04_results_and_discussion.md` | Chapter 4 — Results and Discussion (§§4.1–4.6) | **Lycos2017 populated; CICIDS2017 TBD** | 2 400 |
| `05_conclusion.md` | Chapter 5 — Conclusion (§§5.1–5.5) | **Complete draft** | 1 500 |
| `references.md` | References (APA 7) | **≈ 80 entries, APA 7** | 1 800 |

Current body-text total ≈ 15 600 words — within the FYP template's 10 000 – 30 000 word window.

## Compilation

The thesis is maintained as Markdown for version control and compiled to `.docx` using pandoc with the FYP template as the reference:

```bash
pandoc \
  docs/thesis/front_matter.md \
  docs/thesis/01_introduction.md \
  docs/thesis/02_related_work.md \
  docs/thesis/03_research_methodology.md \
  docs/thesis/04_results_and_discussion.md \
  docs/thesis/05_conclusion.md \
  docs/thesis/references.md \
  --reference-doc="docs/thesis/FYP Thesis Template  042025 v2.docx" \
  -o thesis.docx
```

## Chapter 2 (Literature Review) Structure

- **2.1 Deep Learning for Network Intrusion Detection** — five waves: early autoencoders → CNN / RNN specialisation → CNN-LSTM hybrids → Transformer → GNN & foundation models. ≈ 25 references.
- **2.2 Self-Supervised Representation Learning** — SimCLR / MoCo lineage; BYOL / SimSiam / Barlow Twins / VICReg non-contrastive family; theoretical work (Arora; Wang & Isola); supervised contrastive (Khosla); tabular SSL (VIME / SCARF / SubTab). ≈ 21 references.
- **2.3 Contrastive SSL for NIDS** — Anomal-E → ConFlow / SSCL-IDS / CLDNN-related methods → CLAN paradigm shift → hard-negative lineage (FaceNet, MoCHi) → gap analysis. ≈ 15 references.
- **2.4 Datasets and Evaluation** — DARPA → KDD99 → NSL-KDD → UNSW-NB15 → CICIDS2017 → Lycos2017 / NF-v2; metric pitfalls (Axelsson base-rate, Saito PR-AUC); emerging benchmarks. ≈ 17 references.
- **2.5 Synthesis and Positioning** — places the present study at the intersection of the four sub-fields.

## Current Scope

The final project scope is intentionally narrowed for the available timeline:

- Implement and reproduce **CLAN only**.
- Run the same CLAN pipeline on **Lycos2017** and the original **CICIDS2017**.
- Treat the few-shot shot-count sweep as the compute-feasible ablation.
- Leave the seven SSL baselines (SimCLR, Barlow Twins, BYOL, VICReg, SimSiam,
  ConFlow, SSCL-IDS) as future work, not as completed thesis experiments.

This scope matches the current codebase: only `CLANLoss` is implemented, and
baseline losses are not selectable from config or CLI.

## Citation Health Notes

Entries flagged with ⚠ in `references.md` need verification against the primary source before final submission:

1. **SSCL-IDS authorship** (Golchin et al., 2024, IFIP Networking) — confirm against CLAN's own bibliography at arXiv:2509.06550.
2. **ConFlow** — attributed to Liu et al. (2023, SPNCE LNICST 525) but some agents found Wang et al. (2022) alternatives. Kept the more specific citation.
3. **CLDNN baseline** (Lopes et al., 2022–2023) — placeholder pending primary-source confirmation.
4. **Najar et al. (2025)** hybrid CNN-BiLSTM — placeholder for the 2025 Scientific Reports / MDPI Sensors hybrid CNN-BiLSTM line; confirm the specific paper.
5. **"FAR-constrained recall"** — no canonical primary paper; conceptually credited to Axelsson (2000).

## Template Compliance Checklist

- ✅ English throughout
- ✅ Third-person narration ("the author", "the present study") — no first-person voice except the final paragraph of §5.5 (permitted by the template)
- ✅ Chapter numbering matches template (5 chapters)
- ✅ APA 7 references
- ⏳ Placeholder fields in `front_matter.md` (`[[STUDENT NAME]]`, `[[SUPERVISOR NAME]]`, `[[YEAR]]`, etc.) need filling before submission
- ⏳ CICIDS2017 data tables in Chapter 4 (`TBD` cells) need filling after the Windows/RTX 3060 experiments complete
- ⏳ Figures 4.1 and 4.2 are optional plots derived from the same experiment output
