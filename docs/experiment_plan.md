# Active Experiment Plan

Status date: 2026-07-14. This plan begins from the existing GoalCarrier paper,
not the earlier adapter or CA-only proposals.

## Objective

Test whether a native same-state image-CFG goal transported by a content-only
carrier improves the style--content frontier over K/V replacement and
score-only optimization.

## P0 Matched Matrix

Fixed pairs:

```text
00/kp1  castle on a cliff by the sea
13/kp3  field of sunflowers at sunset
27/kp0  panda eating bamboo
```

Seeds: `64,65,66`.

Methods:

```text
kv_match
score_match
score_carrier  (GoalCarrier)
ca_dmd         (persistent fake-score extension)
```

Fixed formal settings:

```text
resolution: 512x512
outer steps: 25
nominal inner iterations: 5, native dynamic schedule
start time: 1000
optimizer: AdamW
latent weight decay: 0
style gradient RMS target: 2.159326726314248e-06
distance weight: 0.3
content weight: 0.4
carrier temperature: 0.1
carrier strength: 0.75
carrier high-pass gain: 0
carrier loss weight: 0.25
SLT/AdaIN: disabled
```

Only method-required flags may differ. Every row must share reference, prompt,
seed, initial noise, evaluator, and sampler configuration.

## P0 Mechanism Interventions

Run on the same matrix or a predeclared subset:

| intervention | purpose |
|---|---|
| `lambda_A=0` / identity | isolate native score goal |
| uniform affinity | test global smoothing |
| shuffled content tokens | destroy spatial carrier correspondence |
| reference-derived affinity | expected layout-leakage positive control |
| temperature sweep | test locality |
| strength sweep | style--layout Pareto curve |

## Metrics

Primary paired outcomes:

```text
CSD style similarity      higher
CLIP-T prompt alignment   higher
DINO style similarity     higher
VGG Gram distance         lower
DINO layout leakage       lower
```

Add:

- paired bootstrap 95% confidence intervals;
- per-pair and per-seed win rates;
- blinded human style fidelity and content preservation;
- wall time, peak GPU memory, native forward count, latent backward count.

No single automatic metric is sufficient. A claim requires the visual grid and
at least style, prompt, and layout evidence.

## CA+DMD Validity Gate

Record fake-score model, optimizer, update frequency, timestep sampling,
replay/on-policy source, persistent state path, resume behavior, and train/eval
mode. A reset, frozen, or absent fake backend invalidates a DMD headline.

## Output Contract

Each run receives a unique prefix and writes:

```text
image
reference copy/path
per-step CSV
summary JSON
stdout/stderr log
manifest row
```

After all rows finish, create one unified matched manifest and evaluate once.
Do not hand-average metrics from different evaluator runs.

## Submission Gate

The broad empirical claim is unlocked only after:

1. all matched rows complete without protocol drift;
2. GoalCarrier improves the paired style--layout frontier with uncertainty;
3. human preference supports the automatic result;
4. runtime/memory are reported honestly;
5. a larger license-documented reference set confirms the direction.

Until then, the existing three-pair result is a controlled mechanism study.
