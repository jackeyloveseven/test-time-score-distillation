# Project State

Status date: 2026-07-23.

This is the shortest authoritative description of the project. The canonical
paper prose is `AAAI/main.tex`; this file records the operational and empirical
state around it.

## Paper

**Title:** *Test-Time Score Distillation: Mask-Free Semantic Style Editing with
Reverse-Process Regularization*

**Target:** AAAI 2027, anonymous submission using the local `aaai2027` author
kit.

**Compiled state:** `AAAI/main.pdf` and `AAAI/supplement.pdf` are both eight
US-Letter pages. Both compile without undefined references, missing figures, or
overfull boxes. The figure-package validator passes 15 active packages.

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

- `AAAI/figures/tsd_framework/` is the canonical method schematic.
- `AAAI/figures/agent_multidim_radar/` is the canonical six-axis radar figure.
- `AAAI/figures/agent_benchmark_pipeline/` is the canonical Agent evaluation
  workflow. Its package contains generation and correction prompts, intent,
  research notes, caption, QA, script, accepted image, and API provenance.
- Every active figure is self-contained; no loose active PNG is permitted at
  the figure root.

## Not Yet Established

- Full three-seed uncertainty over all 40 styles. Seeds 64 and 66 remain to be
  generated for four methods, 320 cells total.
- The reverse-shield ablation over strengths `{1, 0.5, 0}` under standard and
  stress settings.
- A public-open-weight replication of the third-judge audit.
- Human preference correlation; human evaluation is intentionally outside the
  current validation scope.
- Multi-backbone generality, exact structured-motif reproduction, or a
  compute-matched external leaderboard.

## Current Runtime State

The resumable queue is implemented by `scripts/run_remaining_validation.sh`.
It is **not currently running**. The last driver log stopped at the GPU
headroom gate, reporting roughly 7 GiB free. Relaunch only after verifying
sufficient free memory; never terminate foreign GPU processes.

The root filesystem is at 100% utilization with roughly 18 MiB free. Disk
cleanup is an additional blocker for new experiment outputs and must preserve
owned results, checkpoints, and foreign data.

## Locked Boundaries

- The controlled baseline is the same native MMDiT image+text edit path that
  TSD modifies.
- Published external methods belong in Related Work; FireRed/Qwen appear only
  in the explicitly bounded five-style audit.
- TSD is training-free of weights but optimization-based at inference.
- K/V manipulation and exact motif transfer are outside the method by design.
- Agent evaluation is a reproducible perceptual diagnostic, not a replacement
  for human evaluation or automatic ground truth.
