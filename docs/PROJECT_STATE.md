# Project State

Status date: 2026-07-27.

This is the shortest authoritative description of the project. The canonical
paper prose is `AAAI/main.tex`; this file records the operational and empirical
state around it.

## Paper

**Title:** *Test-Time Score Distillation: Mask-Free Semantic Style Editing with
Reverse-Process Regularization*

**Target:** AAAI 2027, anonymous submission using the local `aaai2027` author
kit.

**Compiled state:** `AAAI/main.pdf` is 9 US-Letter pages (technical content
p1-7, references only on p8-9); `AAAI/supplement.pdf` is 11 US-Letter pages;
`AAAI/checklist.pdf` is 2 US-Letter pages, the official AAAI-2027 28-item
reproducibility checklist compiled as a standalone document
(`AAAI/checklist.tex`), not appended to the supplement. All three compile
without undefined references, missing figures, or overfull boxes; all fonts
are embedded with 0 Type-3 fonts (fixed 2026-07-27 via matplotlib
`pdf.fonttype=42`); `aaai2027.sty`/`.bst` are byte-identical to the official
AuthorKit27; no forbidden packages are used; 39 references are cited (one
uncited entry removed from the 40-entry `.bib`). The figure-package validator
passes 15 active packages.

## Method

Test-Time Score Distillation (TSD) optimizes only the current latent at
inference. At each reverse step it:

1. evaluates one frozen reference-conditioned branch and one frozen
   reference-free branch at the same latent, timestep, text metadata, and image
   token geometry;
2. forms the conditional velocity contrast;
3. rms-normalizes that contrast and constructs a detached clean-state
   pseudo-target at fixed update energy;
4. updates the latent for a small number of inner iterations; and
5. applies the unmodified frozen reverse step.

No model weights, adapters, fake-score networks, or generators are trained.
The central empirical interpretation is that the frozen reverse process acts
as an implicit inference-time regularizer. This is not claimed as an exact
projection theorem or guaranteed distribution matching.

## Two Task Modes

- **Semantic Style Editing:** 25 reverse steps, 5 inner iterations, step size
  0.08, rms normalization. The reference composition is preserved while its
  medium is transferred.
- **Global Style Transfer:** 8 reverse steps, dynamic inner schedule, step size
  0.12, rms normalization. The output forms a more independent composition
  while retaining the reference medium.

Guidance-scale and positive loss-weight multipliers are algebraically absorbed
by rms normalization; the live controls are reverse-step count, inner
iterations, step size, and normalization granularity.

## Established Evidence

- Fixed-seed 40-style benchmark, seed 65: TSD Semantic reaches CSD
  `0.72 +/- 0.08` versus `0.27 +/- 0.16` for the matched native
  composition-preserving edit. TSD Global reaches `0.61 +/- 0.09` versus
  `0.25 +/- 0.16` for the matched native new-composition edit.
- TSD wins CSD on all 40 references in both pairings. The reported standard
  deviations are cross-style dispersion, not multi-seed uncertainty.
- Multi-prompt extension covers five fixed styles and four additional prompts.
- The task-aware Agent benchmark scores six separate axes: medium, texture,
  palette, subject, task locality, and visual quality.
- The complete primary Agent supplies reported means. A secondary Agent covers
  68/80 paired cells and is used only for reliability auditing; persistent
  malformed responses are recorded without imputation.
- On shared cells, primary and secondary Agents agree on the preferred method
  in 68/68 cases. The local wrong-reference control validates reference-specific
  style discrimination only, not the full six-axis rubric.
- A fixed five-style matched-input audit includes FireRed Image Edit and Qwen
  Image Edit. It is a perceptual stress test, not a compute-matched leaderboard.
- SRSD was tested and rejected as a style metric: its correlation is dominated
  by over-stylized, content-collapsed outputs.

## Figures

- `AAAI/figures/tsd_framework/` is the canonical method schematic (Fig. 1):
  defines CA/DM, with panel (a) Decoupled DMD (grad L = CA + DM) and panel (b)
  our TSD (grad L = CA alone).
- `AAAI/figures/main_results/` supplies the canonical two-panel quantitative
  figure (Fig. 2) and the two-prompt dense style grid (Fig. 3).
- `AAAI/figures/reverse_shield/` supplies the canonical reverse-shield
  rho-knockout ablation (Fig. 4).
- The six-axis radar figure has been removed from the paper; it is no longer
  canonical.
- `AAAI/figures/agent_benchmark_pipeline/` is the canonical Agent evaluation
  workflow, still included in the supplement. Its package contains generation
  and correction prompts, intent, research notes, caption, QA, script,
  accepted image, and API provenance.
- Every active figure is self-contained; no loose active PNG is permitted at
  the figure root.

## Not Yet Established

- Full three-seed uncertainty over all 40 styles. Seeds 64 and 66 are
  intentionally out of budget for this submission and are deferred to future
  work, not a submission blocker; four methods, 320 cells total.
- Human preference correlation; human evaluation is intentionally outside the
  current validation scope.
- Multi-backbone generality, exact structured-motif reproduction, or a
  compute-matched external leaderboard.

## Wrap-up State

The paper is final and pushed. Disk usage is back to normal (~3.8 GB used,
plenty free); no validation is queued for the submission. The only remaining
work is the human upload step (main-PDF deadline 2026-07-28; code and
supplement appendix due 2026-07-31). The multi-seed queue
(`scripts/run_remaining_validation.sh`, seeds 64/66) is deferred future work,
not a submission blocker, and is not currently running.

## Locked Boundaries

- The controlled baseline is the same native MMDiT image+text edit path that
  TSD modifies.
- Published external methods belong in Related Work; FireRed/Qwen appear only
  in the explicitly bounded five-style audit.
- TSD is training-free of weights but optimization-based at inference.
- K/V manipulation and exact motif transfer are outside the method by design.
- Agent evaluation is a reproducible perceptual diagnostic, not a replacement
  for human evaluation or automatic ground truth.
