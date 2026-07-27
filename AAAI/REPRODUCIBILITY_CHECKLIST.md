# AAAI-27 Reproducibility Checklist — answer key (for the submission form)

Paste-ready answers for the AAAI reproducibility checklist. The same content is
rendered as a "Reproducibility Checklist" section at the **end of the technical
appendix** (`supplement.pdf`, last section). Use this file if AAAI-27 instead
asks for the answers in a **designated submission-form field**.

> Placement note: AAAI-27 submission-instructions say the checklist is "uploaded
> separately from the main paper in the designated field"; the AAAI-26/-25
> checklist pages say it goes "at the end of the technical appendix." These are
> not in conflict (both mean: not inside the 7-page main PDF). We satisfy both —
> the checklist is in the supplement appendix **and** available here for a form
> field. Confirm the exact AAAI-27 mechanism on the submission site.

## General
1. Conceptual outline / pseudocode of AI methods introduced — **yes**
2. Delineates opinions/hypotheses/speculation from objective facts — **yes**
3. Well-marked pedagogical references for less-familiar readers — **yes**

## Theoretical contributions?  **No** (claims are empirical by design)
> The velocity-contrast = score-difference relation (§3) is a derivation from
> standard, cited tools; the reverse-process regularization is an explicit
> empirical finding (Conclusion), not a theorem. Theorem/proof items therefore
> N/A. Empirical evidence and its code (reverse-step ablation, fake-score
> negative study, rejected SRSD) are included.

## Datasets?  **Yes** (fixed style-reference set)
4. Motivation for selected data — **yes**
5. Novel data in a data appendix — **partial** (IDs/prompts/seeds/order in manifests; images model-generated, described)
6. Novel data public on publication w/ license — **partial** (intended; license TBD)
7. Existing-literature datasets cited — **NA** (none used)
8. Existing-literature datasets public — **NA**
9. Non-public data described in detail — **yes**

## Computational experiments?  **Yes**
10. Number/range of hyper-parameters tried in development — **yes** (parameter study)
11. Pre-processing code in appendix — **partial**
12. All source code for running/analyzing experiments included — **partial** (released)
13. Source code public on publication w/ license — **yes** (intended)
14. New-method code has comments referencing the paper — **partial**
15. Method for setting seeds described — **yes** (single fixed seed 65)
16. Computing infrastructure specified (hardware/software/versions) — **partial** (single-GPU, bf16, AdamW; versions pinned in released env)
17. Evaluation metrics formally described w/ motivation — **yes** (CSD, CLIP-T, layout-leak)
18. Number of runs per reported result stated — **yes** (fixed-seed; 1 run/cell over n=40 styles)
19. Analysis beyond single-number summaries — **yes** (per-style, SD, 95% bootstrap CIs, 40/40 wins, frontier)
20. Significance judged with statistical tests — **partial** (bootstrap CIs, paired wins, correlation stats; no formal null test; no random-seed claim)
21. All final hyper-parameters listed — **yes** (per-mode steps/iters/η/norm table)

## Honesty flags carried into the answers
- "partial" on data/code public + license = release manifest and license terms
  are **not yet finalized** (a remaining pre-camera-ready task).
- "partial" on statistical significance = fixed-seed (seed 65); we label
  variability as cross-style dispersion, **not** random-seed uncertainty, and
  make no multi-seed claim (seeds 64/66 deliberately out of the current budget).
- Reverse-step ablation is a **qualitative** knock-out (Fig. 4 / supp Fig. 7);
  the quantitative CSD/VLM aggregation for it was not run.
