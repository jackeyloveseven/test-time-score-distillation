# Results Catalog

Status date: 2026-07-14. Raw generated files remain under `results/` on
`ubuntu-107` and are ignored by Git.

Metric directions: CSD, CLIP-T, and DINO style are higher-is-better; VGG Gram
distance and DINO layout leakage are lower-is-better.

## Headline Representative Pair

Style 27, prompt `kp0`, seed 64, 512x512, 25 outer steps, start time 1000.

| Method | CSD | CLIP-T | DINO | Gram | Leak |
|---|---:|---:|---:|---:|---:|
| K/V replacement | 0.5614 | 0.3465 | 0.0677 | 2.9437 | 0.2893 |
| Score match | 0.6770 | 0.3476 | 0.0749 | 2.4451 | 0.3018 |
| GoalCarrier | **0.6882** | 0.3282 | 0.0727 | **2.2915** | **0.2780** |
| Strict GoalCarrier audit | 0.689619 | 0.337495 | 0.074046 | 2.390118 | 0.280550 |

Sources: `docs/formal25_teacher_comparison.md` and
`trial/online_rl_critic/PROTOCOL_AUDIT.md`.

## Three-Pair Matched Comparison

Pairs: `00/kp1`, `13/kp3`, `27/kp0`; seed 64; matched protocol.

| Method | CSD | CLIP-T | DINO | Gram | Leak |
|---|---:|---:|---:|---:|---:|
| K/V replacement | 0.402729 | 0.328626 | 0.047569 | **4.029874** | **0.335180** |
| GoalCarrier | **0.566176** | **0.329624** | **0.061848** | 4.089123 | 0.347909 |

The CSD standard deviations across pairs are `0.065980` and `0.163541`. This is
one seed per pair, not a matched multi-seed benchmark.

Canonical server artifact:
`results/formal25_multi_matched_metrics_summary.json`.

## CA-Only Multi-Seed Diagnostic

Three references x three seeds (`64,65,66`), 25/5 protocol:

```text
CSD:         0.560468 +/- 0.062460
CLIP-T:      0.328679 +/- 0.016119
DINO style:  0.060192 +/- 0.021182
VGG Gram:    3.967326 +/- 1.256918
Layout leak: 0.344434 +/- 0.034204
```

This validates online stability for `ca_only`; it is not a comparison against
GoalCarrier or K/V at the same nine cases. Canonical server artifact:
`results/online25_ca_only_multiseed_metrics_summary.json`.

## Adam/AdamW Check

| Optimizer | Decay | CSD | CLIP-T | Gram |
|---|---:|---:|---:|---:|
| Adam | 0 | 0.6764 | 0.3187 | 2.3796 |
| AdamW | 0 | 0.6758 | 0.3269 | 2.4706 |
| AdamW | 1e-4 | 0.7045 | 0.3198 | 2.4039 |
| AdamW | 1e-3 | 0.6813 | 0.3158 | 2.4985 |

Across three pairs, decay `1e-4` changes mean CSD from `0.5662` to `0.5607`.
The formal setting remains AdamW with zero decay.

## Optional Image-Code Student

```text
random matched field cosine: image code 0.4988 vs style ID 0.4486
held-out style-27 cosine:     image code 0.3094 vs style ID 0.3164
```

Real-image conditioning works as an interface; broad generalization is not
established.

## Image-Reward Diagnostic

| Method | CSD | CLIP-T | DINO | Gram | Leak |
|---|---:|---:|---:|---:|---:|
| GoalCarrier baseline | **0.689619** | 0.337495 | 0.074046 | 2.390120 | **0.280550** |
| GRPO every stage | 0.644513 | 0.332246 | 0.066161 | 2.330238 | 0.308398 |
| Same-state local oracle | 0.681706 | 0.341500 | 0.067942 | **2.270237** | 0.298368 |
| GRPO late only | 0.683222 | **0.354927** | **0.074363** | 2.280977 | 0.304994 |

The diagnostic ranks noisy intermediate decodes; it is not final-image
best-of-N. Source: `trial/online_rl_critic/PROTOCOL_AUDIT.md`.

## Evaluator Provenance

Exact model paths, checkpoint SHA-256 values, and preprocessing are embedded in
each `*_metrics_summary.json` on the server. Do not replace evaluator caches
inside a matched comparison. Current metrics use CSD-CLIP ViT-L/14, OpenCLIP
ViT-B/32, DINOv2 base, and torchvision VGG19.
