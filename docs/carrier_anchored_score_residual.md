# Carrier-Anchored Score Residual

## Method

The method separates the reference-conditioned edit into two roles:

```text
native score residual = what style change is desired
content affinity carrier = where the change should be rendered
```

For a native edit teacher evaluated at the same `(z_t, t)`:

```text
goal_score = score(with-reference) - score(without-reference)
raw_goal_theta = G_theta(z_t, t, prompt, reference_code)
rendered_residual = Transport(raw_goal_theta, Affinity(z_t))
```

The affinity uses only the current content latent. It is detached before the
student update, so the reference cannot enter the spatial carrier. The
reference-conditioned native score is the style goal; the content state is the
spatial rendering mechanism.

The implementation is exposed as:

```bash
--prediction-mode carrier_anchored
--carrier-temperature 0.15
--carrier-strength 1.0
--carrier-highpass-gain 0
```

`hard_delta` remains the explicit K/V-derived carrier baseline. The new mode
does not use reference K/V at inference. `ScoreResidualStudentRuntime` stores
the prediction mode, carrier parameters, and target space in each checkpoint.

## Native Goal Data

The native probe now supports:

```bash
--dump-fields --dump-latents
```

Each dump records the exact `z_t`, timestep, reference metadata, and both:

```text
native_delta_velocity = v_with - v_without
native_delta_score    = -((1-sigma)/sigma) * native_delta_velocity
```

The manifest builder accepts `--teacher-kind native_delta_score` and rejects
field dumps without the matching latent. Score targets are converted back to
velocity residuals before scheduler execution through the inverse coefficient;
the training and sampling spaces are therefore explicit and consistent.

## First Matched Run

Dataset:

```text
180 samples
styles: 00, 13, 27
prompts: kp0, kp1, kp2, kp3, kp4
timesteps: 12 scheduler states
latent: [1, 1024, 128]
target: native_delta_score
seed: 65
```

The matched students used the same token MLP, split seed, optimizer, epochs,
and loss. The carrier temperature `0.1` was nearly an identity transform:

```text
transport cosine ~= 0.9995
```

Increasing the temperature to `0.15` produced a real but bounded spatial
change:

```text
transport cosine ~= 0.914 on the training run
```

Random validation field results:

| student | best epoch | best val MSE | val cosine at best checkpoint |
|---|---:|---:|---:|
| full residual | 18 | 0.000327 | 0.4369 |
| carrier anchored, temperature=0.15 | 26 | **0.000223** | **0.4486** |

At scale `0.25`, the held-out score-field evaluation gives:

| student | MSE | cosine | target norm ratio |
|---|---:|---:|---:|
| full residual | 0.0000985 | 0.4369 | 0.573 |
| carrier anchored | **0.0000976** | **0.4486** | **0.435** |

The zero-field baseline is `0.0001060`, so raw MSE alone is not enough to
establish a useful native score student. The directional improvement and the
non-identity carrier are the current evidence; multi-seed, grouped holdout,
and stronger image-level evaluation remain required.

Image-level smoke after fixing score-to-velocity conversion, 512x512,
6-step sampling, style 27 / prompt kp0, scale 1.0:

| student | CSD style sim | CLIP-T | DINO style sim | VGG Gram dist | DINO layout leak |
|---|---:|---:|---:|---:|---:|
| full residual | -0.0061 | 0.3246 | 0.0480 | 4.0151 | 0.2739 |
| carrier anchored | -0.0235 | 0.3260 | 0.0482 | 4.1000 | **0.2726** |

This is a runtime smoke, not a paper benchmark. It confirms that the score
space conversion is active and that both checkpoints decode successfully.

## Reproduction

Collect native fields:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/probe_native_edit_teacher.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp2,kp3,kp4 \
  --seed 65 --height 512 --width 512 --num-steps 12 \
  --dump-fields --dump-latents \
  --output-prefix results/native_carrier_goal_s65
```

Build the score manifest:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/build_score_residual_manifest.py \
  --field-dumps 'results/native_carrier_goal_s65_fields/*.pt' \
  --out-prefix score_residual_manifest_native_carrier_s65 \
  --results-dir results \
  --teacher-kind native_delta_score \
  --verify-teacher
```

Train the carrier-anchored student:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_native_carrier_s65.jsonl \
  --out-dir results/native_score_student_s65_carrier_t015 \
  --epochs 120 --batch-size 8 --hidden-dim 512 \
  --model-type token_mlp --lr 0.001 --weight-decay 0 \
  --prediction-mode carrier_anchored \
  --carrier-temperature 0.15 --carrier-strength 1.0 \
  --carrier-highpass-gain 0 --cosine-weight 0.05 \
  --device cuda
```

For native-only image sampling, pass
`--score-student-native-only`. This adds the converted student correction to
`v_normal` and does not capture or inject reference K/V tensors.

## Next Validation Gate

The next gate is not another optimizer tweak. It is a grouped and image-level
evaluation with at least three seeds and held-out exact references. The main
comparisons are:

```text
full native score residual
carrier-anchored native score residual
hard_delta K/V carrier baseline
normal no-style baseline
```

The claim should remain conditional until carrier-anchored residual improves
style similarity without increasing layout leakage across those splits.
