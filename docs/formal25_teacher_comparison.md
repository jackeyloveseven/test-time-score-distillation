# Formal With-Style Teacher Comparison

This note records the first matched 25-step comparison for the second paper.
All code, manifests, logs, and generated outputs belong to this repository.

## Protocol

- model: `main=/mnt/local/klein9b`, native teacher=`/mnt/local/klein9b-base`
- style: `style_id=27`, reference image `sty/27.png`
- content: `prompt_id=kp0`, `seed=64`, `512x512`
- sampler: 25 outer steps, 5 inner iterations, start time 850
- loss isolation: SLT disabled, warm-start AdaIN disabled, final AdaIN disabled
- gradient matching: `style-grad-rms-target=2.159326726314248e-06`
- evaluation: CSD style similarity, CLIP-T, DINO style similarity, VGG Gram distance, DINO layout leakage

The native teacher uses the same post-Step1 `z_sigma`, timestep, model, and
image IDs for the positive and negative branches. `score_match` and `ca_only`
use the reference-conditioned native edit velocity; `kv_match` is the existing
attention-carrier baseline.

## Results

| method | CSD style sim | CLIP-T | DINO style sim | VGG Gram dist | DINO layout leak |
|---|---:|---:|---:|---:|---:|
| `kv_match` | 0.56138 | 0.34651 | 0.06769 | 2.94373 | 0.28934 |
| `score_match` | 0.42778 | 0.34322 | 0.06098 | 3.12333 | 0.29593 |
| `ca_only` | 0.38698 | 0.34400 | 0.06220 | 3.19797 | 0.30009 |

The score teacher is a genuine model-native with-style signal, but this single
formal pair does **not** establish superiority over the attention carrier.
The useful positive signal is that score matching has lower layout leakage
than `ca_only` and is substantially closer to the style target than `ca_only`.

## Reproduction

Run the three methods with the commands recorded in the experiment log. Then
build `results/formal25_matched_manifest.csv` and evaluate it with:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \\
  scripts/evaluate_matched_teacher_outputs.py \\
  --manifest-csv results/formal25_matched_manifest.csv \\
  --out-csv results/formal25_matched_manifest_metrics.csv \\
  --summary-json results/formal25_matched_manifest_metrics_summary.json \\
  --device cuda
```

## Next experiment

The next method should preserve the native score field as the semantic/style
goal while restoring a spatial carrier. The clean ablation is:

```text
L = L_score_goal + lambda_carrier * L_spatial_carrier
```

where the carrier is applied to the main DiT spatial self-output or attention
transport, and is detached from the teacher score branch. This tests the paper
hypothesis directly: the score field identifies what style change is desired,
while the carrier determines where it can be rendered without copying
reference layout.

