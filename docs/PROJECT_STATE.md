# Project State

Status date: 2026-07-14.

This file is the shortest authoritative description of the project. If an old
plan, result log, or handoff conflicts with this file, use this file, current
source code, and `AAAI/main.tex`.

## Current Thesis

The paper studies reference-based style transfer as a goal--carrier--flow
problem. The native image condition exposes a useful style direction, but that
direction needs content-aligned spatial support before it is executed by the
rectified-flow sampler.

## Main Method

The paper-facing method is `score_carrier`, named **GoalCarrier**:

1. evaluate native with-reference and without-reference branches on the same
   latent, timestep, prompt metadata, and image-token IDs;
2. convert their velocity difference to a score difference after analytically
   removing the shared state term;
3. construct a detached token affinity from the current content state only;
4. transport the native goal through that affinity;
5. optimize the current latent through differentiable base clean-state
   prediction and a detached carrier pseudo-target.

No global network parameters are trained during the main method.

## Current Paper

Title:

> From Visual Goal to Denoising Flow: Carrier-Anchored Image CFG for
> Training-Free Style Transfer

The working PDF is seven technical pages, two reference pages, and one AAAI
reproducibility-checklist page. The supplement is three pages. Paper compilation
uses result images available on `ubuntu-107`; the Mac is documentation-only.

The prose is conference-level, but broad empirical readiness is not complete.
The working preamble is also not the official AAAI author-kit style.

## What Is Validated

- Score/velocity conversion and detach boundaries are implemented and tested.
- The strict representative GoalCarrier rerun reproduces CSD `0.689619`.
- On the representative historical comparison, K/V, score match, and
  GoalCarrier obtain CSD `0.5614`, `0.6770`, and `0.6882`.
- On three matched pairs, GoalCarrier improves mean CSD from `0.4027` to
  `0.5662` over K/V and keeps CLIP-T approximately unchanged.
- The three-pair result does not dominate Gram distance or layout leakage.
- AdamW is the default latent optimizer; zero decay is the formal setting.
- A real-image VAE code can condition an optional field student, but held-out
  style evidence is nearly tied with style-ID conditioning.
- Local noisy-state CSD reward, including GRPO and a same-state local oracle,
  does not beat the formal GoalCarrier CSD baseline.
- Regression verification on 2026-07-14: `53 tests, OK`.

## What Is Not Established

- broad superiority over many styles, prompts, and seeds;
- human preference gains;
- a complete runtime/memory comparison;
- a winning persistent CA+DMD result;
- a particular transformer layer where style first appears;
- universal generalization of the optional image-code student;
- equivalence between intermediate-image reward and final-image best-of-N.

## Locked Decisions

- Main method: `score_carrier` / GoalCarrier.
- `ca_only`: online ablation, not the paper headline.
- `ca_dmd`: extension requiring a real persistent fake-score backend.
- `kv_match`: baseline, never the native teacher.
- Adapter/student: optional amortization only.
- Native branches must be same-state.
- Carrier affinity must not receive reference tensors.
- Original Hyperion directory is read-only.
- Paper prose lives in `AAAI/main.tex`; Markdown paper notes are summaries.

## Next Gates

1. Complete matched multi-seed K/V, score match, GoalCarrier, and CA+DMD.
2. Run carrier identity, shuffled, uniform, and reference-affinity interventions.
3. Add blinded human style/content preference.
4. Report wall time, forward/backward counts, and peak memory.
5. Add a larger license-documented reference set.
6. Switch to official AAAI style and rerun page/render/font checks.
