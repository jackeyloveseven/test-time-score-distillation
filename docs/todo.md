# Current TODO

The authoritative status and command details are in
`docs/AGENT_HANDOFF.md`. This file is intentionally short enough to use as a
working queue.

## Done

- Isolated all implementation under `style_residual_distill_20260709`.
- Implemented native same-state with-style score/velocity residuals.
- Implemented score-to-velocity conversion for FLUX.2 rectified flow.
- Implemented content-affinity carrier transport.
- Added online `ca_only`, `score_carrier`, and `ca_dmd` execution paths.
- Added optional offline carrier-anchored image-code student.
- Added AdamW latent optimizer support with explicit weight decay.
- Added regression tests; last verified result: `53 tests, OK`.
- Completed online CA-only 3-reference x 3-seed validation.
- Updated paper draft, story blueprint, README, and handoff documentation.

## P0: Next Experiments

1. Run nine matched `score_carrier` samples with the same protocol as CA-only.
2. Run nine matched `ca_dmd` samples with persistent fake-score state.
3. Sweep `carrier_loss_weight` while fixing all other flags.
4. Evaluate every method using the same manifest schema and metric models.
5. Assemble a qualitative grid before making an image-quality claim.

## P1: Method Validation

- Verify fake-score optimizer, timestep sampling, replay, save, and resume.
- Compare CA-only, score-carrier, score-match, K/V match, and CA+DMD at equal
  compute budgets.
- Add explicit teacher/reference forward counts and wall-clock measurements.
- Add exact unseen-reference and unseen-style-family splits.

## P2: Paper Strengthening

- Test a stronger frozen image encoder against the pooled VAE mean/std code.
- Run layer probes for early goal, spatial carrier, and late flow executor.
- Report field cosine, spatial coherence, frequency energy, and layout leakage.
- Keep the DMD claim conditional until the dynamic fake-score experiment passes.

## Guardrails

- Never edit `/home/ubuntu/chengyanli/items/save/attention_20260426_090948`.
- Do not call offline student training a requirement of the main method.
- Do not compare runs with different seeds, prompts, timesteps, or evaluator
  caches without labeling the comparison.
- Do not commit generated checkpoints or image dumps.
