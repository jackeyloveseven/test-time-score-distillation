# Experiment Plan and Reporting Protocol

This is the current experiment plan. Earlier adapter and residual plans remain
in the other notes for provenance; they do not replace this protocol.

## Main Hypothesis

The native reference-conditioned score identifies **what** visual change is
desired, while a content-derived carrier determines **where** that change can
be rendered without copying the reference layout. Online refinement should
therefore outperform unconstrained score matching on the style/layout tradeoff
at comparable compute.

## Fixed Formal Protocol

Use one fixed protocol for every method in a comparison:

```text
styles/prompts: 00/kp1, 13/kp3, 27/kp0
seeds:          64, 65, 66
resolution:     512 x 512
outer steps:    25
inner iters:    5
start-time:     1000
optimizer:      AdamW, latent weight decay 0
SLT:            disabled
warm AdaIN:     disabled
final AdaIN:    disabled
dist weight:    0.3
content weight: 0.4
```

Change one factor at a time. Use a fresh output prefix for each run and add a
manifest row immediately after the image is produced.

## Required Methods

```text
normal/content baseline
kv_match
score_match
ca_only                  current main method
score_carrier            native score plus carrier comparison
ca_dmd                   CA plus dynamic fake-score correction
offline full residual    optional acceleration ablation
offline carrier student  optional acceleration ablation
```

The `ca_dmd` row is not complete until the fake-score backend state, replay
behavior, and update schedule are recorded. A code path existing is not the
same as a validated DMD experiment.

## Metrics

Image-level evaluator:

```text
CSD style similarity  higher is better
CLIP-T                higher is better
DINO style similarity higher is better
VGG Gram distance     lower is better
DINO layout leakage   lower is better
```

Field-level evaluator:

```text
MSE to native field
direction cosine
spatial neighbor coherence
low/high-frequency energy
one-step image effect
```

Systems metrics:

```text
wall-clock time
native teacher forward count
reference-conditioned forward count
backward count
peak VRAM
```

## Current Completed Evidence

CA-only nine-run mean/std:

```text
CSD style similarity: 0.5605 +/- 0.0662
CLIP-T:               0.3287 +/- 0.0171
DINO style similarity:0.0602 +/- 0.0225
VGG Gram distance:    3.9673 +/- 1.3332
DINO layout leakage:  0.3444 +/- 0.0363
```

Field conditioning results are documented in `AGENT_HANDOFF.md` and
`full_paper_draft.md`. The current pooled VAE reference code is reproducible,
but its grouped holdout result is not enough to claim universal image-level
generalization.

## Next Run Matrix

For each method, produce the same nine cases:

```text
(style=00, prompt=kp1, seed=64/65/66)
(style=13, prompt=kp3, seed=64/65/66)
(style=27, prompt=kp0, seed=64/65/66)
```

First run `score_carrier`, then `ca_dmd`, then the carrier weight sweep. Store:

```text
agent_runs/<method>_<case>.log
results/<method>_<case>.png
results/<method>_manifest.csv
results/<method>_metrics.csv
results/<method>_metrics_summary.json
```

## Analysis Rules

- Use mean and standard deviation across seeds, not the best seed.
- Keep style/prompt pairing fixed when comparing methods.
- Report scalar metrics with qualitative grids.
- Separate field fidelity from final image quality.
- Separate online compute from offline training cost.
- Treat layout leakage and Gram distance as tradeoffs until all methods use the
  same preprocessing and evaluator checkpoints.
- Mark hypotheses as hypotheses until causal layer probes are complete.
