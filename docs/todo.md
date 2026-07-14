# Current TODO

Status date: 2026-07-14. `PROJECT_STATE.md` defines current claims.

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
