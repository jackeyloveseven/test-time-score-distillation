# Paper Pitch: Test-Time Score Distillation

Status date: 2026-07-23.

Canonical prose and numbers live in `AAAI/main.tex` and
`AAAI/supplement.tex`. This document is the reviewer-facing story and
claim/evidence map.

## One-Sentence Pitch

Test-Time Score Distillation transfers a reference medium by repeatedly
applying a bounded same-state conditional velocity contrast to the current
latent, while the frozen reverse process supplies the inference-time
regularization that a learned fake score provides during generator training.

## Problem

Reference style transfer conflates two tasks. Semantic Style Editing preserves
the reference composition while changing its subject; Global Style Transfer
keeps the medium but forms a more independent layout. We treat these as the two
ends of a style-locality axis rather than unrelated pipelines.

## Method

At each reverse step, TSD evaluates a frozen reference-conditioned branch and a
frozen reference-free branch on the identical state. Their velocity contrast
is rms-normalized and used to build a detached clean-state pseudo-target. A few
latent-only optimization steps are followed by the model's unchanged reverse
step. TSD trains no weights and adds no module, fake network, or generator.

The reverse-step regime controls locality:

- Semantic: 25 reverse steps, 5 inner iterations, step size 0.08.
- Global: 8 reverse steps, dynamic inner schedule, step size 0.12.

## Contributions

1. **One frozen-model operator for two style-locality tasks.** TSD supports
   composition-preserving semantic editing and composition-dissolving global
   transfer without masks or task-specific modules.
2. **Empirical regularization substitution.** Interleaving bounded task updates
   with the frozen reverse process is sufficient in the tested single-sample
   regime; adding a learned fake-score term suppresses structured media.
3. **Separable evaluation.** CSD, CLIP-T, and layout leak report style, subject,
   and composition separately, preventing one scalar from hiding collapse.
4. **Auditable Agent benchmark.** A six-axis primary judge, independent
   reliability audit, wrong-reference control, explicit failure logging, and
   raw machine-readable outputs expose the perceptual trade-offs.

## Main Evidence

- 40 fixed references, seed 65: CSD `0.719` versus `0.267` for Semantic and
  `0.615` versus `0.247` for Global, with 40/40 paired wins in both tasks.
- The CSD advantage survives a matched-layout analysis; it is not explained by
  composition copying alone.
- Five fixed styles x four additional prompts yield 20/20 paired wins.
- Primary Agent preferences: 39/40 Semantic pairs and 37/40 Global pairs.
- Secondary reliability audit: 68/80 coverage, 68/68 preferred-method
  agreement on shared cells, no imputation.
- Five-style public-editor audit shows strong style-axis performance against
  FireRed and Qwen under matched inputs, while preserving the boundary that it
  is not a compute-matched leaderboard.
- SRSD is rejected because it rewards the same content-collapse pathology that
  makes CSD alone unsafe at extreme stylization.

## Reviewer-Facing Boundaries

- TSD is training-free of weights but optimization-based at inference.
- The fixed-seed SD is cross-style dispersion, not seed uncertainty.
- The reverse process is an implicit empirical regularizer, not an exact
  manifold projection.
- Exact structured motifs and attention/K/V transfer are outside the method.
- The paper validates one frozen backbone; multi-backbone generality is future
  work.
- Agent evaluation is not automatic ground truth or a replacement for humans.

## Paper Structure

1. Introduction: two tasks, one style-locality axis.
2. Related Work: two compact paragraphs covering style transfer and test-time
   score/flow editing.
3. Method: same-state contrast, bounded latent update, reverse-process
   regularization, two operating modes.
4. Evaluation protocol: matched native baseline and three separable metrics.
5. Results: parameter study, 40-style benchmark, Agent benchmark, robustness,
   and mechanism/negative evidence.
6. Limitations and conclusion: optimization cost, one-backbone scope, external
   audit boundary, exact motifs, and empirical status of the regularizer.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| One operator serves both locality modes | Mode-specific parameter study and layout-leak separation | Supported |
| TSD improves style over the matched native edit | 40-style seed-65 benchmark, 40/40 wins | Supported at fixed seed |
| Gain is not only layout copying | Global matched-layout subset and weak CSD-leak correlation | Supported |
| Reverse process regularizes inference-time updates | Direct rho-knockout ablation (rho=0 collapses all 5 styles to noise; Fig. 4 + supp), stable high-guidance behavior, fake-score harm | Empirically supported (qualitative; CSD/VLM aggregation not run) |
| Agent benchmark is auditable | Blinding, six axes, coverage, retries, no imputation, raw logs | Supported |
| Results generalize across seeds | Seeds 64--66 paired matrix | Pending |

## Final Self-Review Questions

- Does every broad claim say whether it is fixed-seed or multi-seed?
- Is reverse-process regularization consistently described as empirical?
- Are primary, secondary, and local-control judge roles kept separate?
- Is the FireRed/Qwen audit clearly bounded as matched-input, not compute-matched?
- Are optimization cost, one-backbone scope, and exact-motif limits visible?

