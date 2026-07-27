# Current TODO

Status date: 2026-07-14 (positioning note added 2026-07-19).
`PROJECT_STATE.md` defines current claims.

> ⚠️ **STALE — CURRENT-TRUTH POINTER (2026-07-21):** this file predates the TSD pivot (07-20) and
> the 07-21 Controllable Style Locality campaign. Authoritative: `docs/HANDOFF.md`. The P0/P1/P2
> lists below are the OLD `score_carrier`/GoalCarrier evidence plan — **SUPERSEDED**, kept as trail.
>
> **Current remaining TODO (07-21):**
> - ~~SRSD↔CSD validation~~ DONE 07-21 → **REJECTED** (positive correlation via content-collapse, not a valid style metric; kept as honest null in paper §4). Keep 3 external axes.
> - Main-line semantic multi-prompt breadth: kp1 castle / kp2 cat DONE & strong; **kp3 sunflowers /
>   kp4 fisherman + full 38-style breadth still remaining.**
> - Global-Transfer layout-free primitive (`v_cR_scram`) stays PARKED (niche, not adopted) unless a
>   fix for its 2/5 subject-destroy failure mode is found — not currently queued work.
> - Standing constraints unchanged: no K/V ever (gen-1 taboo); don't headline "FLUX.2 [klein] 9B"
>   in title/abstract; method is training-free-of-weights but optimization-based (state honestly).

## Gen-2 Positioning (2026-07-19)

Second-generation paper: self-contained goal--carrier--flow, lead with
differentiation. No head-to-head against the first-generation (sec/Hyperion)
method. SLT/AdaIN are first-generation carryovers, already disabled in the
formal setting, and are not part of the method; strip their lineage framing.
LRS is absent from gen-2 (no-op). See `PROJECT_STATE.md` "Gen-2 Positioning".
Open user decision Q1: keep-but-reframe vs drop the K/V baseline -- do not touch
the K/V table until answered.
**(RESOLVED 07-20, still true 07-21): K/V is off-limits entirely — never a comparison, never the
method's mechanism. See `docs/HANDOFF.md` taboo #1. This Q1 is no longer open.)**

## Completed

- Isolated the project from the read-only Hyperion source.
- Implemented same-state native image-CFG residuals and exact FLUX.2 conversion.
- Implemented reference-independent carrier transport.
- Implemented score match, GoalCarrier, CA-only, CA+DMD, and K/V modes.
- Added optional real-image-code student and image sampling.
- Added Adam/AdamW latent optimization and gradient audits.
- Verified 53 tests on the H200 server.
- Reproduced 0.689619 strict GoalCarrier CSD on the formal pair.
- Completed first three-pair GoalCarrier vs. K/V comparison.
- Completed CA-only 3-reference x 3-seed stability diagnostic.
- Completed negative online CSD reward/GRPO protocol audit.
- Rewrote the AAAI paper to 7 technical + 2 reference pages and tracked figures.
- Created a complete Mac Git mirror and archived old local patch folders.

## P0: Evidence Completion

1. Run K/V, score match, and GoalCarrier for all 3 pairs x seeds 64,65,66.
2. Run persistent CA+DMD on the exact same matrix.
3. Report paired bootstrap confidence intervals and per-pair win rates.
4. Run blinded human style-fidelity/content-preservation preference.
5. Measure wall time, peak memory, native forwards, and latent backwards.
6. Add a larger license-documented reference set.

## P1: Mechanism

- Identity carrier (`lambda_A=0`).
- Uniform affinity.
- Shuffled content tokens.
- Reference-derived affinity as an expected leakage-positive control.
- Carrier temperature/strength Pareto sweep.
- Optional multiscale or sparse content-only carrier.

## P1: Submission

- Obtain and apply official `aaai2026.sty`.
- Recompile and preserve the 7-page technical budget.
- Add final multi-seed and human tables/figures.
- Complete dataset licenses, compute statement, and release manifest.

## P2: Conditional Work

- Strengthen the frozen image encoder only for optional amortization.
- Run early/carrier/late layer probes only if layer-emergence returns as a paper
  claim.
- Run true final-trajectory best-of-N only as a supplementary compute study.

## Guardrails

- Never edit the original Hyperion directory.
- Never call K/V replacement a native teacher.
- Never call CA-only CA+DMD.
- Never make the optional student a requirement.
- Never compare unmatched seeds/prompts/checkpoints without labeling it.
- Never call local noisy-state reward final-image best-of-N.
- Never force-add large result trees or checkpoints.
- Never frame GoalCarrier as an increment over the first-generation method or
  add a head-to-head against it.
- Never present SLT/AdaIN warm-start as part of the method; they are disabled
  first-generation carryovers.
