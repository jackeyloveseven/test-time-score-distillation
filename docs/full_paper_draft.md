# Goal, Carrier, and Flow: Image-Conditioned Score Residual Distillation for Style Transfer

## Abstract

Image-conditioned style transfer asks a diffusion model to use a reference
image as a visual instruction while preserving the content layout supplied by
the prompt. We argue that this operation is not a layer-by-layer revealing of
the reference. The style goal can become available early, the current content
state supplies a spatial carrier that determines where the goal is rendered,
and late denoising blocks translate the carried goal into an executable flow
update. We turn this observation into a decoupled image-CFG formulation. A
native with-reference score is differenced against the same-state
without-reference score to obtain the style goal, while a reference-independent
content affinity transports that goal across the current latent tokens. A
student then distills the resulting score residual, optionally with a
carrier-anchored parameterization that does not require reference K/V at
inference. The main conditioning path uses a frozen, deterministic reference
image code derived from the FLUX VAE; style-ID conditioning is retained only as
an ablation. The current implementation includes reproducible native teacher
dumps, score/velocity conversion, carrier-anchored training, image-conditioned
runtime sampling, and CA-only / CA+DMD teacher objectives. Initial matched field
experiments favor carrier anchoring over unconstrained full residual
regression, while image-level and multi-seed results remain an active validation
gate.

## 1. Introduction

Reference-based style transfer is often implemented as feature injection,
attention replacement, or iterative latent optimization. These methods expose
a useful but incomplete question: how does a reference image become an
instruction for the current generation rather than a second scene to copy?

Our starting observation is that editing is not progressively drawing the
reference through depth. The computation has three roles:

```text
native score:  what style change is desired
CA carrier:    where that change should be rendered
flow executor: how the goal becomes the scheduler velocity
```

This decomposition explains a recurring failure pattern. A method may recover
the desired palette or material but place texture on the wrong object. The goal
is present, but the spatial carrier is wrong. Conversely, a strong content
carrier can preserve layout while failing to express the reference style. The
student should therefore not learn both roles as one unconstrained residual.

We study this problem through the Think with Image lens: the reference is an
instruction-bearing image, not a class label or a retrieval key. The paper has
three contributions:

1. A goal-carrier-executor view of image-conditioned editing, with same-state
   native score differences and carrier interventions as measurable probes.
2. A decoupled image-CFG teacher that separates reference-dependent style goal
   from reference-independent content transport.
3. A carrier-anchored score residual student conditioned on a real reference
   image code, with CA-only and CA+DMD objectives as explicit variants.

## 2. Problem Formulation

Let `z_t` be the current latent state, `p` the content prompt, and `x_s` a
reference image. Let `v_0` be the native velocity without the reference and
`v_s` the native velocity with the reference, evaluated at exactly the same
`(z_t, t, p)`.

The reference-conditioned edit field is:

```text
Delta v_native(z_t,t,p,x_s) = v_s(z_t,t,p,x_s) - v_0(z_t,t,p)
```

In score space, the corresponding same-state difference is:

```text
Delta s = -((1-sigma)/sigma) Delta v_native
```

The scalar conversion is part of the data contract. A score target must be
converted back to velocity before it is added to a rectified-flow scheduler.
At the endpoint where the coefficient vanishes, the implementation returns a
zero velocity correction instead of introducing a numerical blow-up.

## 3. Method

### 3.1 Native image-CFG teacher

The teacher evaluates the without-reference and with-reference branches on the
same saved state. The reference enters only through the model's native edit
condition. This produces a style goal that is tied to the current content
state, not a K/V replacement heuristic.

The native teacher dump records:

```text
z_t, t, sigma
v_with - v_without
native_delta_score
prompt text and reference path
reference image code
model and sampling metadata
```

### 3.2 Reference image code

The main student condition is a frozen deterministic code from the reference
image. The current reproducible implementation encodes the resized reference
with the FLUX VAE, flattens its spatial latent tokens, and concatenates their
channel-wise mean and standard deviation:

```text
c_s = [ mean_tokens(VAE(x_s)), std_tokens(VAE(x_s)) ]
```

This code is 256-dimensional for the current 128-channel VAE and does not
depend on a style-ID lookup. It is intentionally simple and auditable. A
stronger image encoder can replace it without changing the student or teacher
field interface. The style-ID model is retained as a memorization ablation.

### 3.3 Content affinity carrier

The current content state defines a token affinity:

```text
A(z_t) = softmax( normalize(z_t) normalize(z_t)^T / tau )
```

The affinity is detached and therefore cannot become an unintended second
optimization target. Given a raw student goal `g_theta`, the carrier-anchored
field is:

```text
g_rendered = g_theta + lambda_CA ( A(z_t) g_theta - g_theta )
```

An optional high-pass term preserves local variation. The reference is absent
from `A(z_t)`, so the carrier cannot directly copy the reference layout.

### 3.4 Carrier-anchored student

The main student is:

```text
g_theta = G_theta(z_t, t, p, c_s)
g_rendered = Transport(g_theta, A(z_t))
```

The field loss is applied to the rendered goal:

```text
L_field = ||g_rendered - Delta s_native||_2^2
        + lambda_cos (1 - cos(g_rendered, Delta s_native))
        + lambda_hf L_highpass
```

At inference, the score residual is converted back to velocity and added to
the normal native velocity. Carrier-anchored mode does not capture or inject
reference K/V. The older hard-delta path remains a direct K/V carrier baseline.

### 3.5 CA-only and CA+DMD

CA has two distinct roles in the paper and they must not be conflated.

The carrier-anchored student uses CA as a structural renderer. The TPAMI-style
latent objective additionally exposes an explicit CA pseudo-guidance loss:

```text
g_CA = (cfg_scale - 1) Delta s_CA
```

For CA+DMD, an online fake score supplies the distribution correction:

```text
g_DMD = Delta s_with - Delta s_fake
g_total = lambda_CA g_CA + lambda_DM g_DMD
```

The pseudo-gradient is detached and converted into an x0 target before the
latent optimizer update. This is different from minimizing the squared norm of
the score difference itself. A real dynamic fake-score backend, persistent
state, and replay/on-policy data are required before making a formal DMD claim.

## 4. Experiments

### 4.1 Field protocol

The first native image-code smoke uses 3 references, 2 prompts, 4 scheduler
states, and seed 66, producing 24 exact same-state samples with latent shape
`[1, 1024, 128]` and image-code shape `[256]`. The first matched carrier run
uses 180 samples from 3 references, 5 prompts, 12 scheduler states, and seed
65.

### 4.2 Matched field results

For the style-ID native-score comparison, the best random-validation results
were:

| student | best validation MSE | validation cosine |
|---|---:|---:|
| full residual | 0.000327 | 0.4369 |
| carrier anchored, tau=0.15 | **0.000223** | **0.4486** |

At a correction scale of 0.25, the carrier-anchored field has MSE 0.0000976
versus 0.0000985 for full residual, with cosine 0.4486 versus 0.4369. The
zero-field MSE is 0.0001060, so MSE alone is not sufficient evidence of useful
style transfer.

Under the same 180-sample carrier-anchored protocol, replacing the memorized
style ID with the frozen reference image code gives:

| conditioning | best scale | MSE | cosine | raw best-checkpoint val MSE |
|---|---:|---:|---:|---:|
| style ID | 0.25 | 0.0000976 | 0.4486 | 0.000223 |
| reference image code | 0.25 | **0.0000961** | **0.4988** | 0.000264 |

The image-code model therefore improves field direction after the same scale
calibration, even though its unscaled regression curve is noisier. This is
still a three-reference experiment and should be treated as evidence for the
conditioning path, not as a final unseen-family generalization result.

On a grouped holdout where all style 27 samples are excluded from training,
the two conditioning paths are close:

| conditioning | best scale | held-out MSE | held-out cosine |
|---|---:|---:|---:|
| style ID | 0.10 | 0.0001211 | 0.3164 |
| reference image code | 0.25 | 0.0001206 | 0.3094 |

This is an important boundary result: the current pooled VAE code passes the
unseen-reference runtime and field gate, but does not yet prove a universal
quality advantage over the style-ID ablation. More style families and a
stronger reference encoder are required for the main generalization claim.

The image-code smoke reaches best validation MSE 0.000225 after seven epochs.
An inference check with an unseen style string succeeds because the checkpoint
uses the reference code rather than a style-ID embedding.

### 4.3 Image-level validation

The first score-space-corrected image smoke is 512x512, 6 steps, style 27,
prompt kp0, seed 65. Full residual and carrier-anchored variants both decode
successfully. Current metrics are mixed: full residual has better CSD and Gram
distance, while carrier anchoring is slightly better on CLIP-T, DINO style,
and layout leakage. These are smoke results, not the final benchmark.

### 4.4 Required ablations

The paper benchmark must report the following under identical seeds, prompts,
references, and scheduler settings:

```text
normal no-style baseline
hard/KV carrier baseline
CA-only native objective
CA+DMD with persistent fake-score state
full native score residual
carrier-anchored native score residual
style-ID conditioning ablation
image-code conditioning main model
reference-layout carrier swap / token shuffle
```

Splits must include unseen exact references and, where available, unseen style
families. Metrics should include style similarity, prompt alignment, Gram or
texture distance, layout leakage, field cosine, spatial neighbor coherence,
runtime, and number of teacher/reference forwards.

## 5. Reproducibility

All source and generated experiment metadata live under:

```text
/home/ubuntu/chengyanli/items/style_residual_distill_20260709
```

The original source directory is read-only reference material and is not part
of the implementation.

Native field collection:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/probe_native_edit_teacher.py \
  --style-ids 00,13,27 --prompt-ids kp0,kp1,kp2,kp3,kp4 \
  --seed 65 --height 512 --width 512 --num-steps 12 \
  --dump-fields --dump-latents \
  --output-prefix results/native_carrier_goal_s65
```

Image-conditioned training:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_native_carrier_s65.jsonl \
  --out-dir results/native_image_code_student_s65 \
  --conditioning-mode image_code \
  --prediction-mode carrier_anchored \
  --carrier-temperature 0.15 --carrier-strength 1.0 \
  --epochs 120 --batch-size 8 --model-type token_mlp \
  --device cuda
```

The image sampler computes the same pooled VAE code from the supplied
reference image. The checkpoint records conditioning mode, target space,
carrier parameters, style/prompt maps, and the complete training config.

## 6. Limitations and Claim Boundary

The present evidence establishes a working native score/velocity data contract,
a non-identity content carrier, and a real image-code training/runtime path.
It does not yet establish universal image-quality superiority over K/V
replacement. The main remaining gates are multi-seed grouped holdouts, exact
unseen-reference evaluation, a stronger image encoder ablation, layer-level
goal/carrier/executor probes, and a dataset-level dynamic CA+DMD experiment.

Until those gates are complete, the correct claim is that carrier anchoring is
a promising decomposition that improves native field direction and provides a
reference-independent spatial rendering constraint, not that it dominates all
style-transfer baselines.
