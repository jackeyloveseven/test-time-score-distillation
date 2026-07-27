# Paper Synopsis

Status date: 2026-07-14.

The full paper is maintained only in `../AAAI/main.tex`. This Markdown file is a
short synopsis for agents and must not become a second independently edited
paper draft.

## Title

**From Visual Goal to Denoising Flow: Carrier-Anchored Image CFG for
Training-Free Style Transfer**

## Problem

Reference-style transfer must separate what appearance is requested from where
that appearance should act. Attention K/V replacement entangles appearance
with reference token geometry. Adapter methods introduce a training problem.
Raw score optimization exposes a native style direction but leaves spatial
support implicit.

## Method

GoalCarrier evaluates a frozen native edit model twice on the exact same latent
and timestep:

```text
g_t = score_with_reference - score_without_reference
```

It then constructs a detached affinity from current content tokens only:

```text
A_t = softmax(normalize(c_t) normalize(c_t)^T / tau)
T_A(g_t) = g_t + lambda_A (A_t g_t - g_t)
```

The transported direction defines a detached clean-state pseudo-target. The
current latent is optimized through the differentiable base branch. No model
parameters or adapter weights are trained.

## Three Contributions

1. A same-state image-CFG residual that exposes a model-native visual goal.
2. A reference-independent spatial carrier and scheduler-consistent online
   objective.
3. Matched mechanism controls and an auditable reproducibility contract.

## Current Evidence

- Strict representative audit: GoalCarrier CSD `0.689619`.
- Representative historical rows: K/V `0.5614`, score match `0.6770`,
  GoalCarrier `0.6882`.
- Three matched pairs: K/V `0.4027` vs. GoalCarrier `0.5662` mean CSD;
  approximately unchanged CLIP-T; explicit Gram/layout trade-off.
- The carrier intervention improves representative leakage from `0.3018` to
  `0.2780` relative to score-only.
- A negative supplementary CSD-reward diagnostic does not improve the formal
  baseline and is not final-image best-of-N.

## Claim Boundary

The paper supports a useful native goal and a promising content-carrier
mechanism on controlled cases. It does not yet support broad benchmark
dominance, a winning CA+DMD result, human preference gains, or a measured
early-layer style-emergence claim.

## Paper Files

```text
../AAAI/main.tex
../AAAI/supplement.tex
../AAAI/references.bib
../AAAI/reproducibility.tex
../AAAI/SUBMISSION_READINESS.md
```

Current layout: 7 technical pages, 2 reference pages, 1 checklist page, and a
separate 3-page supplement. Compilation is performed on `ubuntu-107`.
