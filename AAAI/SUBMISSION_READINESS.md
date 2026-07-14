# Submission Readiness Audit

Status: strong method draft, not yet evidence-complete for a top-tier claim.

## Stable Thesis

Reference-conditioned style transfer is a goal--carrier--flow problem:

1. the same-state native image-CFG residual specifies the visual goal;
2. a reference-independent content affinity supplies spatial support;
3. the rectified-flow clean-state target executes the transported field.

The main method is online and training-free. It optimizes the current latent,
not an adapter. A field student is optional amortization. CA+DMD is a separate
extension that is valid only with a persistent dynamic fake-score backend.

## Completed Paper Improvements

- Replaced the broad title and abstract with a falsifiable method statement and
  bounded numerical claims.
- Reduced four overlapping contributions to three distinct contributions.
- Added a method/result overview and the complete three-pair qualitative grid.
- Corrected the score-matching equations to match code detach boundaries.
- Added reference-independence and non-expansion properties of the carrier.
- Added a claim--evidence audit and explicit method/extension contracts.
- Corrected DMD/DMD2 references and added closest score-editing and
  training-free style-transfer literature.
- Added the AAAI reproducibility checklist after the references.
- Verified 7 technical + 2 reference pages, US Letter, embedded fonts, resolved
  citations, and no box or undefined-reference warnings.

## P0 Evidence Gates

These are required before broad top-tier empirical claims:

1. Run the full matched matrix for `kv_match`, `score_match`, and
   `score_carrier` over all references, prompts, and at least three seeds.
   Report paired mean, standard deviation, confidence interval, and win rate.
2. Add blinded human preference for style fidelity and content preservation.
   Randomize left/right order and report annotator count and agreement.
3. Report runtime, peak memory, native forward count, and latent backward count
   against K/V replacement and the unmodified native edit sampler.
4. Add a broader, license-documented reference set. The current three-pair
   result is a mechanism study, not a benchmark.

## P1 Scientific Gates

- Sweep carrier strength and temperature on the matched multi-seed matrix and
  report the style--layout Pareto frontier.
- Add identity, shuffled-token, uniform-affinity, and reference-affinity carrier
  interventions.
- Calibrate CSD on the chosen reference set and retain CLIP, DINO, Gram, and
  human evaluation so the conclusion does not depend on raw CSD alone.
- Run layer probes and causal activation swaps only if the paper reintroduces
  a claim that style appears in early transformer layers. The current paper
  makes only a noise-schedule claim.

## Format Gate

The working PDF uses a local article approximation. Before submission, obtain
the official AAAI-26 author kit, switch to `aaai2026.sty`, preserve anonymity,
remove acknowledgements, and rerun the full render, page, and font audit.

## Claims That Must Not Drift

- Do not call K/V replacement a native score teacher.
- Do not describe `ca_only` as CA+DMD.
- Do not call the noisy-state CSD diagnostic final-image best-of-N.
- Do not claim broad generalization from three pairs or one matched seed.
- Do not claim an adapter is required by the main method.
- Do not claim style emerges at a specific layer without layer-probe evidence.

## Canonical Evidence Files

- `docs/formal25_teacher_comparison.md`
- `trial/online_rl_critic/PROTOCOL_AUDIT.md`
- `results/formal25_multi_matched_metrics_summary.json`
- `results/online25_ca_only_multiseed_metrics_summary.json`
- `AAAI/main.tex`
- `AAAI/supplement.tex`

Any numerical change in the paper must trace to one of these files or a newly
committed manifest and summary JSON.
