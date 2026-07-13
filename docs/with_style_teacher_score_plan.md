# With-Style Teacher Score for One-Shot Style Distillation

Status: replacement direction for the previous reference-K/V teacher proposal.

## 1. Decision

The original Hyperion branches must not be reused as the main teacher score.
They already define style through a hand-designed attention intervention:

```text
SCFG / SLT:
  same content Q + replaced or transported reference K/V -> v_styled

SDS / hybrid:
  Attn(stopgrad(Q_content), K_ref, V_ref) -> attention-output target
```

Calling either output a new teacher would only distill the original method's
assumptions and artifacts. The new teacher must estimate a distribution that
contains real styled images:

```text
s_with(z_tau, tau | text, reference)
  ~= grad_z log p_real-style(z_tau | text, reference)
```

The recommended hierarchy is:

```text
V0, zero-training diagnostic:
  FLUX.2 Klein Base 9B native image-edit conditional field

V1, paper teacher:
  episodically trained one-shot reference-conditioned StyleScore field
  learned from real same-style, different-content image pairs
```

The distilled four-step Klein9B checkpoint is the student or fast baseline. It
is not the preferred real-score teacher. The locally available undistilled
checkpoint is:

```text
/mnt/local/klein9b-base
```

It contains the full 9B transformer, text encoder, VAE, tokenizer, and scheduler.

## 1.1 Two Separate Plans, One Frozen TPAMI Architecture

The first experiment must compare two losses, not combine them prematurely.
Both plans copy the original TPAMI/Hyperion optimization skeleton exactly:

```text
standard denoise
-> reference/style preparation
-> GRAG / AdaIN / SLT warm-start as configured
-> optimize the current latent with Adam for the same number of iterations
-> keep the existing distribution and content-anchor losses
-> scheduler step and decode unchanged
```

Only the old attention-output `style_loss` is replaced.

```text
Plan A: direct score-difference matching
Plan B: decoupled CA + DMD KL pseudo-loss
```

This makes the causal comparison clean: any improvement comes from the score
objective and its teacher, not from a new generator architecture, adapter, query
module, or sampling pipeline.

### Plan A: Score-Difference Matching

At the current optimizable latent, run the no-reference branch with gradients
and the native with-reference teacher branch with stop-gradient:

```text
v_pre   = T_base(z, tau, y, edit_image=None)
v_style = stopgrad(T_base(z, tau, y, edit_image=reference))

s_pre   = velocity_to_score(z, v_pre, tau)
s_style = velocity_to_score(z, v_style, tau)

L_score = rho(N(s_pre), N(s_style))
```

`rho` starts as normalized MSE or Smooth-L1. `N` removes the timestep-dependent
global scale while preserving direction and relative token energy. Raw MSE is
reported as an ablation.

The original loss becomes:

```text
L_A = lambda_score * L_score
    + lambda_dist * L_dist_original
    + lambda_content * L_content_original
```

This is direct field-output matching through the frozen base model. Because the
optimized object is the latent, it differentiates through `s_pre`; the style
teacher is detached. It should be described as iterative score-field matching,
not DMD.

### Plan B: CA + DMD KL

The TPAMI latent remains the only optimized generator variable, but its style
gradient is replaced by the decoupled KL pseudo-gradient:

```text
Delta_CA = s_with - s_without
Delta_DM = s_with - s_fake

g_KL = lambda_CA * (cfg_scale - 1) * Delta_CA
     + lambda_DM * Delta_DM
```

Here:

```text
s_with    = Base 9B native edit-conditioned with-style score
s_without = the same Base 9B score without the reference condition
s_fake    = an online score model fitted to current optimized outputs
```

Do not minimize `||g_KL||^2`. Inject it as a detached pseudo-gradient. In the
TPAMI loop, obtain a clean-valued straight-through variable from the current
noisy latent while detaching the no-reference velocity branch:

```text
v_without = stopgrad(T_base(z_sigma, sigma, y, no_ref))
x0_ST     = z_sigma - sigma * v_without
x_target  = stopgrad(x0_ST + eta * RMSNorm(g_KL))
L_KL      = 0.5 * ||x0_ST - x_target||^2
```

Thus `dx0_ST/dz_sigma = I`: all with/without/fake score branches are detached,
and the Plan B gradient is the intended pseudo-gradient rather than a gradient
through the Base9B score Jacobian. `RMSNorm` is computed per sample over all
non-batch dimensions, so its per-element scale does not shrink with image
resolution. Unit-L2 normalization is retained only as an explicit ablation.
The sign is checked by verifying that a small update increases the teacher's
with-style likelihood direction. The original regularizers remain:

```text
L_B = lambda_KL * L_KL
    + lambda_dist * L_dist_original
    + lambda_content * L_content_original
```

Plan B adds a fake-score side model because this is mathematically required by
DMD, but the TPAMI generator architecture and optimized latent are unchanged.
The fake model is training machinery, not an inference adapter.

### Exact TPAMI Replacement Site

In the original core, replace only the block that currently computes:

```text
style_loss = sum(L1(self_out, stopgrad(ref_self_out)))
```

and preserve:

```text
optimizer = Adam([latents])
dist_loss
content_loss with its existing iteration decay
SLT/AdaIN warm-start
number of outer denoising steps
number of inner Adam iterations
```

Reference K/V may remain for the original SLT/AdaIN warm-start during the first
controlled test, but it must not enter either new score teacher. A later
ablation removes each warm-start component independently.

## 2. Why Native Image Conditioning Is a Different Teacher

Klein's native edit path concatenates encoded reference image tokens to the
noisy target tokens, assigns them separate image coordinates, runs the model's
learned joint transformer, and returns only the target-token prediction:

```text
target noisy tokens z_tau
reference edit tokens e_ref
text tokens y
  -> frozen Base Klein9B
  -> target velocity v_with
```

This differs from Hyperion replacement in three important ways:

1. The model was trained to use image conditions; no attention tensor is
   manually replaced.
2. Target and reference tokens remain distinct inputs throughout the learned
   MM-DiT computation.
3. The output is a full conditional velocity field at the same target state,
   rather than an attention feature chosen by us as a proxy.

For a fixed target state, timestep, prompt, and noise, define:

```text
v_with    = T_base(z_tau, tau, y, edit_image=reference)
v_without = T_base(z_tau, tau, y, edit_image=None)
Delta_v_style = v_with - v_without
```

Only the reference condition changes. The target `z_tau`, prompt, timestep,
scheduler, and model weights must be identical.

V0 is still only a diagnostic. A generic image-edit model may copy reference
objects or interpret the instruction as semantic editing instead of pure style.
The decisive controls are wrong-reference, prompt-swap, reference-layout crop,
and same-style/different-content tests.

### 2.1 Native-field audit, Base 9B

A same-state audit was run with references `00/13/27`, prompts `kp0/kp1/kp3`,
seed `65`, resolution `512`, and the 12-step FLUX.2 schedule. Each prompt uses
one no-reference trajectory. At every step, all reference fields are evaluated
before the trajectory advances, so the target latent and timestep are exactly
shared. No attention hooks or K/V replacement are registered.

At `t=1000`, changing the clean native edit reference gives:

```text
metric                                      kp0       kp1       kp3
between-reference energy fraction         48.51%    33.25%    39.48%
mean pairwise reference cosine              0.280     0.513     0.417
mean pairwise reference relative L2         1.213     1.003     1.093
mean with-minus-without velocity norm       89.26     83.92     61.28
```

Across the three prompts, the mean between-reference energy fraction is
`40.4%`. The old dynamically noised K/V teacher produced exactly `0%`
reference energy at the same fully noised target timestep. Native image
conditioning therefore fixes the old teacher's reference-blind initial state.

The style-condition delta is not small: averaged over references and prompts,
its norm is `20.7%` of the no-reference velocity norm at `t=1000`. This ratio
falls monotonically to `2.2%` at the final sampled timestep `t=395.47`.
Reference discriminability falls from `40.4%` to roughly `15-21%` in later
steps, while prompt-dependent energy rises from `21.8%` to `60.6%`.

This is evidence that the native field contains reference information from the
start, but not yet evidence that the information is pure style. The large
early-time reference component may encode layout or object identity. V0 passes
the reference-identifiability gate and remains subject to the layout-purity
gate.

A three-reference native sampling check used the same explicit style-only
instruction, panda content, seed `66`, 512 resolution, and 28 Base9B steps.
References `00`, `13`, and `27` produced clearly separated comic/high-saturation,
ink-wash/grayscale, and warm impasto appearances. None of the seed-66 outputs
obviously copied the reference object or full composition. However, an earlier
style-27 seed-64 sample reproduced much of the reference house background.
Layout leakage is therefore stochastic rather than absent, and must be measured
over multiple seeds instead of accepted or rejected from one cherry-picked row.

Reproducible artifacts:

```text
scripts/probe_native_edit_teacher.py
results/native_edit_teacher_field_audit_base9b_s65_summary.json
results/native_edit_teacher_field_audit_base9b_s65.csv
results/native_edit_teacher_probe/style00_panda_seed66_base9b_28step.png
results/native_edit_teacher_probe/style13_panda_seed66_base9b_28step.png
results/native_edit_teacher_probe/style27_panda_seed66_base9b_28step.png
```

## 3. Paper Teacher: Episodic One-Shot StyleScore

### 3.1 Training episodes

Construct episodes from real images:

```text
(x_ref, x_target, y_target, style_group)
```

Required positive episode:

```text
style(x_ref) = style(x_target)
content(x_ref) != content(x_target)
x_ref != x_target
```

`x_target` is the clean sample used for flow matching. `x_ref` is only the
style condition. Pairing different content within the same style is the main
mechanism that prevents the teacher from learning reference reconstruction.

If exact style-family labels are unavailable, bootstrap groups with a style
embedding and nearest-neighbor retrieval, then filter near-duplicate content.
Self-pairs or two crops of one image are permitted only as an ablation because
they encourage instance and texture memorization.

### 3.2 Conditional flow-matching objective

For the actual Klein schedule, sample a clean target latent `z_0`, noise
`epsilon`, and timestep `tau`, then construct `z_tau` with the repository's
scheduler. Train a native edit-conditioned Base Klein teacher:

```text
v_psi = T_psi(z_tau, tau, y_target, x_ref)
L_style_FM = E[ w(tau) * ||v_psi - v_target(z_0, epsilon, tau)||^2 ]
```

The exact target must be taken from the FLUX.2 scheduler convention in code.
Do not copy a DDPM epsilon target into the flow model.

Use reference dropout in the same teacher:

```text
v_psi_with = T_psi(z_tau, tau, y, x_ref)
v_psi_null = T_psi(z_tau, tau, y, null_ref)
```

The paper-facing style direction is:

```text
Delta_v_style = v_psi_with - v_psi_null
```

To prevent null-branch drift from being misidentified as style, preserve the
base field on dropped-reference examples:

```text
L_null = ||T_psi(z_tau, tau, y, null_ref) - stopgrad(T_base(z_tau, tau, y))||^2
```

Recommended teacher loss:

```text
L_teacher = L_style_FM
          + lambda_null * L_null
          + lambda_swap * L_reference_swap
```

`L_reference_swap` penalizes predicting the same style direction for a true
reference and a reference from a different style group. It is a conditioning
identifiability term, not a replacement for flow matching.

### 3.3 Training ownership

The method is defined by the conditional distribution, not by whether training
uses full fine-tuning or low-rank updates. With the available H200 memory, the
preferred order is:

```text
1. train the native image-conditioning path and late/middle transformer blocks
2. unfreeze the full transformer if the held-out-reference score test saturates
3. keep LoRA only as a cost baseline
```

This avoids making an adapter the conceptual contribution. The teacher can be
large and expensive because it is frozen after training and removed from the
final fast sampler.

## 4. Decoupled Style DMD

In this document, CA means **CFG Augmentation**, not cross-attention matching.

Let the student generator produce a current clean prediction:

```text
x_0^G = G_theta(z_t, t, y, x_ref)
```

Re-noise it at `tau` and evaluate all fields at the same `x_tau`.

### 4.1 CA engine

The style-CFG augmentation is:

```text
Delta_CA(x_tau)
  = s_with(x_tau, tau | y, x_ref)
  - s_without(x_tau, tau | y)
```

This is the main style-driving term. It bakes the one-shot style condition into
the student rather than distilling a manually replaced attention output.

### 4.2 DM regularizer

Train a separate fake-score model on the current student outputs:

```text
s_fake_phi(x_tau, tau | y, x_ref)
  ~= grad_x log p_G_theta,tau(x_tau | y, x_ref)
```

The distribution-matching term is:

```text
Delta_DM(x_tau)
  = s_with(x_tau, tau | y, x_ref)
  - s_fake_phi(x_tau, tau | y, x_ref)
```

The fake score must be updated as the student distribution changes. A frozen
second model or an offline regression target is not DMD.

### 4.3 Combined update

The generator update follows the decoupled form:

```text
grad_theta L
  = E[-(
      lambda_DM * Delta_DM(x_tau_DM)
    + lambda_CA * Delta_CA(x_tau_CA)
    ) * dG_theta/dtheta]
```

Use distinct re-noising schedules:

```text
tau_CA: focused on unresolved style information; first baseline tau_CA > t
tau_DM: broad corrective schedule over the full valid range
```

The final schedules should be selected from a measured curve of
`||Delta_CA||`, reference discriminability, and frequency response over time.
The old K/V reference-onset curve cannot be assumed to apply to this teacher.

## 5. Score, Velocity, and Clean-Prediction Forms

Klein predicts a flow/velocity-like quantity. Paper equations may use scores,
but the implementation must convert them with the exact FLUX.2 schedule.

DiffSynth's FLUX.2 scheduler implements the following linear path exactly:

```text
z_sigma = (1 - sigma) * z_0 + sigma * epsilon
v       = epsilon - z_0
```

The model velocity therefore gives:

```text
x0_hat  = z_sigma - sigma * v
eps_hat = z_sigma + (1 - sigma) * v
s       = -eps_hat / sigma
        = -(z_sigma + (1 - sigma) * v) / sigma
```

At the same state and schedule, the state term cancels, so the exact conversion
for all field differences is:

```text
Delta_s = -(1 - sigma) / sigma * Delta_v
Delta_s =  (1 - sigma) / sigma^2 * Delta_x0
```

`sigma` must be read from `scheduler.sigmas` at the scheduler index selected for
the current timestep; using `timestep / 1000` as an independent approximation
is not an acceptable implementation shortcut. Logs must save raw velocity,
converted clean prediction, converted score, and `(1-sigma)/sigma`.

This also changes how the teacher audit is interpreted. At `sigma=1`, an exact
conditional score difference is zero because every condition shares the same
pure-noise endpoint. The large native-edit velocity difference observed at
`t=1000` proves that the model consumes and distinguishes the reference there,
but it is not itself nonzero score guidance after conversion. Plan A and Plan B
should either skip the exact endpoint or let the exact coefficient zero it;
they must use an explicit minimum-sigma guard near the clean endpoint. Raw and
normalized field variants are retained because the conversion coefficient is
strongly timestep dependent.

There is a detach-order requirement when the optimized variable is the latent
itself. Do not separately form `s_pre(z)` and `stopgrad(s_style(z))` and then
subtract them. Their shared `-z/sigma` term cancels numerically but, after that
detach, does not cancel in the latent gradient and introduces an unintended
identity-gradient term. Plan A must instead cancel the common state term
analytically before the detach boundary:

```text
Delta_v = v_pre - stopgrad(v_style)
Delta_s = -(1 - sigma) / sigma * Delta_v
L_score = rho(N(Delta_s), 0)
```

Equivalently, the loss may compare a transformed differentiable velocity to a
transformed detached target velocity, provided the shared state term is never
included. Plan B must construct both `s_with-s_without` and `s_with-s_fake`
from same-state velocity differences for the same reason. A regression test
must show that adding an arbitrary common state tensor to all branches leaves
both the loss value and the latent gradient unchanged.

## 6. Direct Score MSE and KL Score Gradients Are Different Plans

The proposed expression

```text
||score_pre - score_style||^2
```

has two different meanings.

If training a score network `s_theta`, this is valid field regression:

```text
L_field = ||s_theta(z_tau, tau, y, x_ref) - stopgrad(s_with)||^2
```

It is score matching/distillation, but not DMD.

If optimizing an image, latent trajectory, query tensor, or generator output,
backpropagating the squared score difference differentiates through the score
network Jacobian. Plan A intentionally tests this field-matching behavior while
detaching the with-style target branch. It must not be presented as CA or DMD.

Plan B instead detaches the score differences and injects them as the KL
pseudo-gradient, matching the existing DMD2 pseudo-target pattern. The two
experiments answer different questions and should remain separate through the
first image-quality gate.

This distinction must be explicit in the paper and code.

## 7. Role of the Original Hyperion Attention Optimization

The original attention method remains useful, but no longer defines truth.

Acceptable roles:

```text
1. initialize a better content-preserving student trajectory
2. provide a content carrier or low-frequency anchor
3. serve as a baseline teacher in ablations
4. provide an auxiliary attention-consistency loss after the score objective works
```

Forbidden main claim:

```text
reference K/V replacement output == real with-style score
```

A useful decomposition is:

```text
native/learned with-style score = goal field
original content trajectory      = carrier
CA update                        = style engine
DM update                        = stability regularizer
```

## 8. Minimum Decisive Experiments

### Gate A: V0 teacher validity

For at least 8 references, 3 prompts, and 3 seeds:

```text
same z_tau, y, tau
  compare native with-reference vs without-reference field
```

Measure:

```text
reference discriminability of Delta_v_style
prompt sensitivity at fixed reference
wrong-reference direction change
reference-layout leakage
native edit sampling quality
```

Reject V0 as a teacher if it mainly reconstructs reference objects or if field
differences do not track style identity on held-out prompts.

### Gate B: loss-only TPAMI comparison

Reuse the original per-sample latent optimization loop and replace only the old
attention-output `style_loss`.

Compare:

```text
normal Klein
Hyperion K/V optimization
Plan A raw score MSE
Plan A normalized score MSE / Smooth-L1
Plan B CA-only pseudo-gradient
Plan B CA + DM KL pseudo-gradient
```

All variants use identical denoising steps, inner Adam iterations, warm-start,
content loss, and random seeds. This is the fastest test of whether the better
teacher and loss fix the current image quality.

### Gate C: DM stabilization

After CA shows a style gain, add the dynamically trained fake score and compare:

```text
CA only
DM only
CA + DM with shared tau
CA + DM with decoupled tau
```

Track training curves for saturation, high-frequency noise, latent variance,
style similarity, prompt alignment, and reference-content leakage.

### Gate D: real-data StyleScore

Train V1 on real episodes and evaluate on:

```text
held-out exact references
held-out style families
cross-dataset references
same-style references with different layouts
```

The key comparison is V1 against V0 at identical CA/DM schedules. V1 must
improve style identity without increasing reference-layout leakage.

## 9. Baselines and Novelty Boundary

Style Matching Score already uses a fixed style-specific LoRA as the target
style score and a dynamically trained fake score. Repeating that formulation
with another style LoRA is not enough for a new paper.

The defensible new contributions are:

```text
1. one-shot, arbitrary-reference conditional style score learned episodically
   from real same-style/different-content pairs

2. style-CFG augmentation as the main transfer engine, separated from a
   reference-conditioned DM regularizer with independently chosen schedules

3. distillation into a content-preserving Hyperion/Klein execution path while
   removing reference K/V replacement as the teacher definition
```

The first contribution is essential. Without it, the method is too close to
existing style-LoRA score matching and Decoupled DMD.

## 10. Immediate Implementation Order

```text
Phase 0
  implement a same-state native edit-field probe on Klein Base 9B
  dump v_with, v_without, x0_with, x0_without, and Delta fields

Phase 1
  replace the original style loss with direct score-difference matching
  run Plan A on the unchanged TPAMI latent loop

Phase 2
  add the detached CA pseudo-gradient to the same latent loop
  adapt DMD2 fake score and complete Plan B with the KL term

Phase 3
  build real same-style/different-content episodes
  train and validate the episodic StyleScore teacher

Phase 4
  distill the successful field into the final fast student execution path
```

No pooled reference code or style-ID student should be optimized further until
Gate B establishes that the new teacher itself produces a better image gradient.

## 11. Implementation Status (2026-07-13)

Phase 1 and the CA-only subset of Phase 2 are now implemented in
`style_residual_distill_20260709` without modifying the original Hyperion/TPAMI
directories. The new runner keeps the TPAMI latent-optimization skeleton and
replaces only the style objective:

```text
src/flux2_score.py
src/style_objectives.py
src/native_edit_teacher.py
src/ca_dmd_backend.py
src/klein9b_loading.py
src/csv_rows.py
scripts/run_tpami_with_style_teacher.py
scripts/test_tpami_with_style_teacher.py
```

### 11.1 Locked contracts in code

The implementation now enforces the following runtime contracts:

```text
Plan A score_match:
  main denoiser = /mnt/local/klein9b
  teacher       = /mnt/local/klein9b-base
  main may use VRAM offload
  teacher DiT must remain full CUDA
  teacher prompt/edit conditions are detached constants

Plan B ca_only:
  main denoiser = /mnt/local/klein9b
  teacher       = /mnt/local/klein9b-base
  all with/without branches detached
  pseudo-gradient injected through straight-through clean variable

Plan B ca_dmd:
  interface present, but it still errors without a real dynamic
  Base9B-compatible fake-score backend
```

The TPAMI defaults were restored:

```text
NUM_STEPS=25
ITERS=5
LR=0.01
START_TIME=850
STYLE_W=2
DIST_W=0.15
CONTENT_W=0.2
ADAIN_ALPHA=0.5
SLT_ALPHA=0.2
SLT_TAU=0.02
SLT_ITERS=10
SLT_GRAG=0.7
GRAG_SHALLOW/DEEP=1/1
```

The smoke runs below use explicit overrides only:

```text
num_steps=12
iters=1
disable_dynamic_iters=True
```

### 11.2 Regression coverage

The current regression suite contains 26 tests, all passing on
`ubuntu-107`.

```text
1. sigma is resolved from scheduler.timesteps/sigmas, never from t/1000
2. score delta is exactly zero at sigma=1
3. Plan A removes the shared state term before the detach boundary
4. native triplet evaluates same-state branches and changes only the reference condition
5. teacher detach is preserved while the pre branch remains differentiable to latent
6. condition freeze removes prompt graph / grad_fn
7. static-condition assert rejects live graphs and accepts detached tensors
8. RMS pseudo-gradient normalization is resolution-stable
9. Plan B is straight-through on latent only; no fake/teacher parameter grads
10. Plan B update direction matches the intended score-space direction
11. DMD2 clean flow target regression locks `noise - fake_latents_clean`
12. `validate_style_mode('ca_dmd', 'none')` errors as required
13. the placeholder DMD2 backend bridge raises NotImplementedError
14. heterogeneous skip/optimized rows write a stable CSV schema
15. runner source contains exactly one `scheduler.step(...)`
16. reference-K/V GRAG only compresses image tokens
17. kv_match loss detaches the Q-target branch while keeping self-out gradients
18. captured reference K/V preserves the passed timestep tensor exactly
19. gradient-audit autograd.grad does not pollute latents.grad
20. style-grad target formula matches `target / (|style_weight| * raw_rms)`
21. style-grad calibration freezes after the first measured active step
22. default `style_grad_rms_target=None` leaves the total loss unchanged
23. unified gradient-audit entry supports score_match / ca_only / kv_match
24. score_match forces the teacher to full CUDA under legacy/main VRAM limits
25. kv_match forces the main DiT to full CUDA and skips external teacher load
26. runner reuses the same `timestep_style` tensor for ref capture and inner kv forward, and clears K/V cache around the main denoise forward
```

### 11.3 Runtime failure root cause and fix

The first real Plan A smoke failed at the first style backward with:

```text
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:0 and cpu!
```

Root cause:

```text
teacher_prompt_embeds was produced without torch.no_grad()
-> its autograd graph still referenced the teacher text encoder
-> runner moved teacher text encoder back to CPU to save VRAM
-> total_loss.backward() tried to propagate through a mixed cuda/cpu graph
```

Fix:

```text
encode_prompt_for_pipe:
  wrap prompt encode in torch.no_grad()
  detach prompt_embeds / text_ids / null_* explicitly

prepare_edit_condition:
  wrap native edit-image encode in torch.no_grad()
  detach edit_latents / edit_image_ids explicitly

runner:
  assert all teacher condition tensors are static constants before optimization
  keep only teacher DiT on CUDA for score_match
  move teacher text encoder and VAE back to CPU and empty_cache after encoding
```

An unrelated late-run smoke failure was also fixed:

```text
write_csv previously used rows[0].keys()
skip rows and optimized rows had different schemas
-> csv.DictWriter raised ValueError on extra fields
```

It now uses a stable union fieldname order:

```text
shared keys in first-row order
+ remaining keys in sorted order
```

### 11.4 Real smoke commands

CA-only smoke, accepted:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --main-model-root /mnt/local/klein9b \
  --teacher-model-root /mnt/local/klein9b-base \
  --style-mode ca_only \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 64 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --iters 1 \
  --disable-dynamic-iters \
  --main-vram-limit-gb 24 \
  --teacher-vram-limit-gb 24 \
  --out-prefix tpami_native_teacher_smoke_dualvram24 \
  --results-dir results
```

Score-match smoke, accepted:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --main-model-root /mnt/local/klein9b \
  --teacher-model-root /mnt/local/klein9b-base \
  --style-mode score_match \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 64 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --iters 1 \
  --disable-dynamic-iters \
  --main-vram-limit-gb 16 \
  --out-prefix tpami_native_teacher_smoke_main16 \
  --results-dir results
```

### 11.5 Smoke artifacts and quantitative results

Artifacts were written remotely under:

```text
/home/ubuntu/chengyanli/items/style_residual_distill_20260709/results/
```

and synchronized back to the local mirror under:

```text
remote_patch/style_residual_distill_20260709/results/
```

CA-only artifacts:

```text
tpami_native_teacher_smoke_dualvram24_ca_only_style27_kp0_seed64.png
tpami_native_teacher_smoke_dualvram24_ca_only_style27_kp0_seed64_reference.png
tpami_native_teacher_smoke_dualvram24_ca_only_style27_kp0_seed64.csv
tpami_native_teacher_smoke_dualvram24_ca_only_style27_kp0_seed64_summary.json
```

Score-match artifacts:

```text
tpami_native_teacher_smoke_main16_score_match_style27_kp0_seed64.png
tpami_native_teacher_smoke_main16_score_match_style27_kp0_seed64_reference.png
tpami_native_teacher_smoke_main16_score_match_style27_kp0_seed64.csv
tpami_native_teacher_smoke_main16_score_match_style27_kp0_seed64_summary.json
```

Both smokes satisfied the scheduler contract:

```text
ca_only:      scheduler_step_calls_total = 12
score_match:  scheduler_step_calls_total = 12
```

Peak allocated CUDA memory:

```text
ca_only:      66.9902 GB
score_match:  45.5204 GB
```

Final optimized rows recorded in the summaries:

```text
ca_only
outer=7  sigma=0.78125  coeff=-0.279296875  style=0.0012491082  dist=0.0568847656  content=0.1118164062  total=0.0333820060
outer=8  sigma=0.70703  coeff=-0.4140625    style=0.0012543197  dist=0.0781250000  content=0.0708007812  total=0.0283875465
outer=9  sigma=0.58984  coeff=-0.6953125    style=0.0012504867  dist=0.0688476562  content=0.0654296875  total=0.0258774385
outer=10 sigma=0.39453  coeff=-1.53125      style=0.0012506000  dist=0.0639648438  content=0.0732421875  total=0.0267321579

score_match
outer=7  sigma=0.78125  coeff=-0.279296875  style=0.0021880134  dist=0.0568847656  content=0.1118164062  total=0.0352598168
outer=8  sigma=0.70703  coeff=-0.4140625    style=0.0013792382  dist=0.0800781250  content=0.0698242188  total=0.0287594534
outer=9  sigma=0.58984  coeff=-0.6953125    style=0.0010372784  dist=0.0712890625  content=0.0649414062  total=0.0257561971
outer=10 sigma=0.39453  coeff=-1.53125      style=0.0009247961  dist=0.0659179688  content=0.0732421875  total=0.0263857245
```

### 11.6 Visual smoke conclusion

At this smoke setting:

```text
style27 / kp0 / seed64 / 512 / 12 denoise steps / 1 inner iter
```

both `ca_only` and `score_match` decode successfully and produce visually close
images. Neither smoke run is obviously broken, blank, or collapsed, and both
preserve the panda content prompt. At the same time, the two outputs are close
enough that they do not yet support a strong qualitative claim that one loss is
better than the other. This is consistent with the runs being only a functional
smoke rather than a formal TPAMI comparison.

### 11.7 Gradient-scale audit protocol and diagnostic runs

The runner now supports a non-default gradient-scale audit that does not change
the default algorithm:

```text
--log-gradient-components
  measure only the first active inner step with torch.autograd.grad(...)
  do not populate latents.grad
  then run the normal total_loss.backward() path unchanged

--style-grad-rms-target FLOAT
  optional first-step frozen calibration
  semantics = target RMS of the final weighted style term
```

Implementation contract:

```text
raw_style_grad_rms
  = RMS( d style_loss / d latents )

if style_grad_rms_target is None:
  calibration_scale = 1
  effective_style_weight = style_weight

if style_grad_rms_target is set:
  require |style_weight| * raw_style_grad_rms > eps
  calibration_scale = target / (|style_weight| * raw_style_grad_rms)
  effective_style_weight = style_weight * calibration_scale
  freeze this Python float for the entire run from the first active inner step onward
```

The first measured active step is recorded in both CSV and summary as:

```text
style_grad_rms
style_weighted_grad_rms_pre_calibration
style_weighted_grad_rms_applied
dist_weighted_grad_rms
content_weighted_grad_rms
calibration_scale
effective_style_weight
style_dist_grad_cosine
style_content_grad_cosine
```

Three diagnostic runs were executed at:

```text
style27 / kp0 / seed64 / 512 / 12 denoise steps / 1 inner iter
disable_dynamic_iters=True
style_grad_rms_target=None
log_gradient_components=True
```

Commands:

```bash
# ca_only
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --main-model-root /mnt/local/klein9b \
  --teacher-model-root /mnt/local/klein9b-base \
  --style-mode ca_only \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 64 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --iters 1 \
  --disable-dynamic-iters \
  --main-vram-limit-gb 24 \
  --teacher-vram-limit-gb 24 \
  --log-gradient-components \
  --out-prefix tpami_grad_audit \
  --results-dir results

# score_match
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --main-model-root /mnt/local/klein9b \
  --teacher-model-root /mnt/local/klein9b-base \
  --style-mode score_match \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 64 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --iters 1 \
  --disable-dynamic-iters \
  --main-vram-limit-gb 16 \
  --log-gradient-components \
  --out-prefix tpami_grad_audit \
  --results-dir results

# kv_match
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --main-model-root /mnt/local/klein9b \
  --teacher-model-root /mnt/local/klein9b-base \
  --style-mode kv_match \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 64 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --iters 1 \
  --disable-dynamic-iters \
  --main-vram-limit-gb 24 \
  --log-gradient-components \
  --out-prefix tpami_grad_audit \
  --results-dir results
```

Observed first-active-step weighted gradient RMS values:

```text
ca_only
  style_raw      = 3.8133e-07
  style_weighted = 7.6267e-07
  dist_weighted  = 1.6052e-06
  content_weighted = 1.5189e-06
  peak_cuda_gb   = 34.3205

score_match
  style_raw      = 4.1155e-06
  style_weighted = 8.2311e-06
  dist_weighted  = 1.6052e-06
  content_weighted = 1.5189e-06
  peak_cuda_gb   = 45.7015

kv_match
  style_raw      = 1.1260e-04
  style_weighted = 2.2521e-04
  dist_weighted  = 1.6019e-06
  content_weighted = 1.5187e-06
  peak_cuda_gb   = 31.5483
```

So under the original `style_weight=2` protocol, the effective style gradient
scale differs by orders of magnitude:

```text
ca_only      ~ 7.6e-07
score_match  ~ 8.2e-06
kv_match     ~ 2.25e-04
```

This is an audit result, not yet a claim that any teacher is better. Formal
comparison must therefore report both:

```text
1. original-weight runs
2. matched-gradient runs using a frozen first-step style_grad_rms_target
```

### 11.8 Current limitations and next gates

These accepted smokes are only execution checks:

```text
12 denoise steps
1 inner Adam iteration
single style
single prompt
single seed
```

They are not sufficient for the actual loss-only comparison. The next formal
gate still requires:

```text
25 steps
5 inner iterations
multiple styles
multiple prompts
multiple seeds
paired visual and metric comparison
TPAMI-full and warm-start-off / loss-isolation ablations reported side by side
```

Current open limitations:

```text
1. Plan B now has a bounded real `ca_dmd` plumbing path with a standalone full
   fake scorer, but this is still only a single-run smoke.
2. Formal DMD claims still require a dataset-level persistent fake critic
   state, replay buffer and/or cross-sample training runner. The current
   bounded smoke therefore remains `experimental_scope = plumbing_only`.
3. The existing DMD2 codebase may be reused only as infrastructure. Its current
   `target = noise - noisy_latents` flow target is incorrect for clean
   fake-latent supervision and must not be copied as correctness logic.
4. The current visual gap among `kv_match`, `ca_only`, `score_match`, and the
   bounded `ca_dmd` smokes is small in the smoke regime, so quality claims must
   wait for the full 25-step/5-iter setting.
5. The current smoke outputs are visually very close. Shared AdaIN/SLT
   warm-start may be masking differences in the style objective itself. Formal
   comparison must therefore include a warm-start-off / loss-isolation ablation
   such as `disable_slt + adain_alpha=0` while keeping dist/content losses and
   the outer denoise schedule unchanged.
6. The current `ca_dmd` evidence is bounded to 256/8 and 512/12/1iter
   diagnostics; it does not yet establish stable longer-horizon critic behavior
   in the formal TPAMI 25-step/5-iter setting.
```

In particular, for flow training the regression lock is:

```text
correct clean flow target = noise - fake_latents_clean
wrong DMD2 shortcut       = noise - noisy_latents
```

Therefore `ca_dmd` is now only allowed through a real dynamic fake score
backend rather than a placeholder loss that only squares score differences.
The remaining block is on formal dataset-level DMD claims, not on plumbing.

### 11.9 score_match backward nondeterminism baseline

Current-code `score_match` was checked with same-parameter back-to-back reruns.
The divergence is now treated as an accepted baseline of the differentiable
teacher path itself, not as evidence that the sidecar gradient audit is still
changing the algorithm.

Measured image deltas:

```text
no_audit_run1 vs no_audit_run2
  pixel_mae = 0.879898
  pixel_max = 68

no_audit_run1 vs audit_run
  pixel_mae = 1.066675
  pixel_max = 72

no_audit_run2 vs audit_run
  pixel_mae = 1.156160
  pixel_max = 58
```

Key trace:

```text
all three runs match through outer step 7 before the first active latent update
row 8 is the first differing row
pre-update style_loss / total_loss are identical at that step
divergence appears only after the first differentiable teacher backward
```

So the accepted interpretation is:

```text
score_match current path has inherent CUDA-side nondeterminism in the
differentiable native-teacher backward
sidecar audit drift is inside that pairwise baseline range
```

The sidecar design was still tightened to be safer:

```text
probe uses an isolated latent clone
probe saves/restores CPU + CUDA RNG state
kv probe restores hook state after use
probe no longer trains the fake critic again
```

But pixel-exact A/B is no longer a blocking gate for proceeding to real
`ca_dmd`.

## 12. CA+DMD plumbing status

A real bounded full-fake backend is now wired under:

```text
fake_score_backend = klein_full_fake
```

Current backend contract:

```text
1. full-parameter standalone fake DiT critic, no LoRA / adapter path
2. fake prompt embeds come from fake_model_root's own text encoder
3. fake with-style scoring uses fake_model_root's own native edit condition
4. critic train_step uses exact scheduler sigma lookup, never timestep/1000
5. critic train_step samples independent noise and (uniform or current) train timestep
6. predict_velocity scores current z_sigma / timestep after the update
7. sampled parameter update logs record whether BF16 critic weights actually moved
8. optional fake_grad_clip_norm clips critic gradients and logs pre/post norms
9. backend state can be saved/resumed; this is required for future cross-sample
   persistent training, even though the present smoke is single-run only
```

Important scope limit:

```text
single-image latent optimization is only a plumbing smoke
it does not by itself establish a dataset-level p_fake distribution
summary therefore records experimental_scope = plumbing_only
```

### 12.1 4B bounded plumbing smokes

Reference backend:

```text
fake_model_root = /mnt/local/klein4b-base
main / teacher = /mnt/local/klein9b and /mnt/local/klein9b-base
main / teacher kept on 24 GB offload to reserve headroom for fake backward
```

Unclipped SGD `lr=1e-3` overshot:

```text
step1: 0.454674 -> 0.427071
step2: 0.718790 -> 1.889140
```

`clip10` reduced the blow-up but was still unstable:

```text
step1: 0.454674 -> 0.427046
step2: 0.719010 -> 0.763626
pre/post grad step2: 41.86 -> 10.02
sample nonzero step2: 9
```

Accepted 4B bounded smoke:

```text
command family:
  style_mode=ca_dmd
  fake_score_backend=klein_full_fake
  fake_model_root=/mnt/local/klein4b-base
  fake_optimizer=sgd
  fake_lr=1e-3
  fake_grad_clip_norm=5
  fake_train_timestep_mode=uniform
  256x256, 8 steps, 1 inner iter

step1: 0.454674 -> 0.432624
step2: 0.687558 -> 0.679378
sample update nonzero: 25 / 9
peak CUDA GB: 49.02
```

So 4B passes the plumbing gate, but remains only an engineering smoke and not a
formal DMD result.

### 12.2 9B full-fake bounded smoke

Full-parameter 9B fake critic:

```text
fake_model_root = /mnt/local/klein9b-base
fake_architecture = standalone_full_model
fake_init_checkpoint_root = /mnt/local/klein9b-base
optimizer = sgd
lr = 1e-3
fake_grad_clip_norm = 5
fake_train_timestep_mode = uniform
main / teacher remain on 24 GB offload so fake backward keeps the headroom
```

Accepted 9B bounded smoke (`256x256`, `8 steps`, `1 iter`):

```text
scheduler_step_calls_total = 8
optimized_outer_steps      = 2
experimental_scope         = plumbing_only
peak_cuda_gb               = 68.90

outer 5
  critic loss    0.394292 -> 0.390357
  grad pre/post  3.08729  -> 3.08729
  sample nonzero 37
  sample update rms 7.19e-08
  CA score rms   0.02066
  DM score rms   0.01079
  combined rms   0.04517
  CA/DM cosine   0.24173
  train sigma    0.87500
  scoring sigma  0.69922

outer 6
  critic loss    0.624369 -> 0.615884
  grad pre/post  4.47085  -> 4.47085
  sample nonzero 57
  sample update rms 8.95e-08
  CA score rms   0.03982
  DM score rms   0.02977
  combined rms   0.09237
  CA/DM cosine   0.27402
  train sigma    0.92188
  scoring sigma  0.49805
```

This is enough to show:

```text
1. full 9B fake critic loads and trains online
2. fake conditioning is aligned with the real with-style branch
3. train_sigma and scoring_sigma are distinct and both scheduler-resolved
4. CA and DM are both non-zero in score space
5. sampled fake parameters do move under BF16 full-model training
```

It is still not sufficient to claim true DMD behavior beyond plumbing because
the critic is reset per run and no cross-sample persistent state or replay
buffer is active yet.

### 12.3 9B full-fake 512 bounded smoke

Larger bounded smoke, still single-run only:

```text
command family:
  style_mode=ca_dmd
  fake_score_backend=klein_full_fake
  fake_model_root=/mnt/local/klein9b-base
  fake_optimizer=sgd
  fake_lr=1e-3
  fake_grad_clip_norm=5
  fake_train_timestep_mode=uniform
  512x512, 12 steps, 1 inner iter
  main / teacher on 16 GB offload, fake full-parameter 9B on CUDA
```

Accepted 9B bounded smoke (`512x512`, `12 steps`, `1 iter`):

```text
scheduler_step_calls_total = 12
optimized_outer_steps      = 4
fake update_count          = 4
experimental_scope         = plumbing_only
peak_cuda_gb               = 70.798

outer 7
  critic loss    0.505664 -> 0.504941
  CA score rms   0.01256
  DM score rms   0.00611
  CA/DM cosine   0.109
  sample nonzero > 0

outer 8
  critic loss    0.266222 -> 0.264425
  CA score rms   0.01569
  DM score rms   0.00871
  CA/DM cosine   0.248
  sample nonzero > 0

outer 9
  critic loss    0.465370 -> 0.463486
  CA score rms   0.02495
  DM score rms   0.01649
  CA/DM cosine   0.295
  sample nonzero > 0

outer 10
  critic loss    0.511169 -> 0.510634
  CA score rms   0.05310
  DM score rms   0.04100
  CA/DM cosine   0.409
  sample nonzero > 0
```

Matched-image comparison against the accepted `ca_only` 512 smoke
(`style27/kp0/seed64`, same 12-step / 1-iter schedule):

```text
pixel MAE  = 0.544329
pixel max  = 28
pixel RMSE = 1.00846
```

So the 9B full-fake `ca_dmd` path is now plumbing-validated at 512 resolution,
but still visually close to the matched `ca_only` smoke and still not a formal
dataset-level DMD result.

## 13. Style-Contrast Teacher Candidate

### 13.1 Why this candidate was needed

The current native-teacher implementation originally used the same full style
prompt text for all native branches:

```text
positive / with-style:
  full style prompt text
  + reference edit condition

negative / without-style:
  full style prompt text
  + no reference edit condition
```

So the negative branch still read text such as:

```text
Use the visual style of the reference image ...
```

while receiving no reference latent at all. This means the old CA signal could
mix two effects:

```text
1. genuine reference-conditioned style direction
2. missing-reference semantics under a prompt that still asks for a reference style
```

This does not prove the old teacher is wrong, but it is a real confound for any
claim that the CA direction itself is stronger than the original K/V baseline.

### 13.2 Explicit conditioning contract

The bounded `style_contrast` candidate therefore keeps the same latent state and
the same model/timestep, but changes the negative conditioning explicitly:

```text
positive branch:
  full style prompt
  + native reference edit condition

negative / pre branch:
  content-only prompt
  + no reference edit condition
```

For the bounded runs here the negative prompt is:

```text
Create A panda eating bamboo.
```

The summary contract is now explicit:

```text
same_state_contract:
  same z_sigma / timestep_style / model / image_ids
  but positive vs negative conditioning both change

teacher_conditioning_contract:
  positive branch = full style prompt + reference
  negative/pre branch = content-only prompt + no reference
```

This candidate does not replace the existing CA definitions. It is only a
minimal teacher-conditioning ablation for bounded screening.

### 13.3 Old matched result: screening failure

The old `full_style_prompt` matched-gradient single-sample screen still does not
support the claim that native score teachers are better than the original K/V
baseline.

Matched-gradient `full_style_prompt` metrics (`style27/kp0/seed64`, 512, 12
steps, 1 iter, warm-start-off):

```text
kv_match
  CSD        = 0.397564
  CLIP-T     = 0.347221
  DINO style = 0.062297
  Gram dist  = 3.25924
  Layout     = 0.294697

score_match
  CSD        = 0.359191
  CLIP-T     = 0.343200
  DINO style = 0.060008
  Gram dist  = 3.26974
  Layout     = 0.289723

ca_only
  CSD        = 0.352390
  CLIP-T     = 0.343640
  DINO style = 0.060270
  Gram dist  = 3.28267
  Layout     = 0.288226

ca_dmd
  CSD        = 0.349556
  CLIP-T     = 0.343563
  DINO style = 0.060515
  Gram dist  = 3.28181
  Layout     = 0.288436
```

Interpretation:

```text
1. kv_match still leads CSD / CLIP-T / DINO style and also keeps the best Gram distance.
2. native teachers mainly show lower layout leak, not stronger style evidence.
3. therefore this single-sample bounded screen does not support "score teacher > KV".
```

### 13.4 Content-only bounded contrast with final AdaIN still enabled

These runs fixed the negative-branch conditioning contrast, but they still kept
one hidden style confound: the runner was unconditionally applying a final
global AdaIN at decode time. So `--disable-slt --adain-alpha 0` removed SLT and
warm-start AdaIN only; it did **not** yet produce a true loss-only output. The
numbers below therefore remain useful as a pre-fix screen, not as final
loss-only evidence.

All three `style_contrast` bounded runs hit the same matched-gradient target
exactly:

```text
style_grad_rms_target = 2.159326726314248e-06
score_match applied weighted style grad RMS = 2.159326726314248e-06
ca_only     applied weighted style grad RMS = 2.159326726314248e-06
ca_dmd      applied weighted style grad RMS = 2.159326726314248e-06
```

Resulting effective weights:

```text
score_match = 0.897243
ca_only     = 5.657315
ca_dmd      = 5.657738
```

Bounded `content_only` contrast metrics, evaluated together with the existing
matched `kv_match` control:

```text
kv_match
  CSD        = 0.397564
  CLIP-T     = 0.347221
  DINO style = 0.062297
  Gram dist  = 3.25924
  Layout     = 0.294697

score_match_style_contrast
  CSD        = 0.354513
  CLIP-T     = 0.343627
  DINO style = 0.060802
  Gram dist  = 3.27182
  Layout     = 0.288524

ca_only_style_contrast
  CSD        = 0.359874
  CLIP-T     = 0.343430
  DINO style = 0.061227
  Gram dist  = 3.28787
  Layout     = 0.289444

ca_dmd_style_contrast
  CSD        = 0.354061
  CLIP-T     = 0.344070
  DINO style = 0.061380
  Gram dist  = 3.28906
  Layout     = 0.289074
```

Key old -> new CSD changes:

```text
score_match  0.359191 -> 0.354513
ca_only      0.352390 -> 0.359874
ca_dmd       0.349556 -> 0.354061
```

So the conditioning fix helps CA / CA+DMD in CSD, but hurts score_match in CSD.
This is already enough to reject any simple "conditioning fix solves it"
positive story.

For the `ca_only` vs `ca_dmd` comparison under `content_only`:

```text
ca_only_style_contrast
  CSD        = 0.359874
  CLIP-T     = 0.343430
  DINO style = 0.061227
  Gram dist  = 3.28787
  Layout     = 0.289444

ca_dmd_style_contrast
  CSD        = 0.354061
  CLIP-T     = 0.344070
  DINO style = 0.061380
  Gram dist  = 3.28906
  Layout     = 0.289074
```

Interpretation:

```text
1. CA+DMD remains lower than CA on CSD.
2. CA+DMD is slightly higher than CA on CLIP-T and DINO style.
3. CA+DMD has slightly lower layout leak than CA.
4. CA+DMD is slightly worse than CA on Gram distance.
```

The image-space gap between the two is still modest:

```text
pixel MAE(ca_only_style_contrast, ca_dmd_style_contrast) = 0.621498
pixel max                                                 = 67
pixel RMSE                                                = 1.19799
```

### 13.5 CA+DMD content-only bounded critic check

The 9B full-fake critic remains a real online full-model critic in this run:

```text
fake_model_root = /mnt/local/klein9b-base
optimizer       = SGD
lr              = 1e-3
grad clip       = 5
train timestep  = uniform
peak CUDA GB    = 70.821
```

Critic-update sanity check:

```text
outer 7   0.599551 -> 0.595243   sample update nonzero = 48
outer 8   0.244602 -> 0.243855   sample update nonzero = 17
outer 9   0.421608 -> 0.420394   sample update nonzero = 47
outer 10  0.637829 -> 0.637697   sample update nonzero = 20
```

So the bounded `ca_dmd` run still satisfies the plumbing gate:

```text
1. full 9B fake critic loaded
2. four online critic updates executed
3. all four critic losses decreased
4. sampled parameter updates stayed nonzero
```

### 13.6 Pre-fix bounded conclusion

The correct reading of this pre-fix bounded evidence is:

```text
1. old full_style_prompt single-sample matched screening does not support score > KV.
2. the content-only conditioning fix helps CA / CA+DMD relative to their own old
   full_style_prompt variants, but still does not beat kv_match on this sample.
3. score_match becomes lower-layout but weaker on CSD under content_only.
4. therefore the current teacher direction remains weak as a single-sample
   bounded screening result.
5. none of these runs should be packaged as positive evidence that the native
   score teacher is better than the original K/V baseline.
```

This remains only a screening result:

```text
single style
single prompt
single seed
12 denoise steps
1 inner Adam iteration
```

The DMD critic also still resets every run, so it is not yet a persistent fake
distribution learner. That persistence gap remains a real limitation.

### 13.7 True loss-only rerun after disabling final AdaIN

The decode-path confound above was then removed explicitly. The runner now
supports:

```text
--disable-final-adain
```

and the true loss-only contract is logged in every summary as:

```text
final_adain_enabled = false
loss_isolation_contract.true_loss_only = true
```

For this bounded re-screen, the common protocol was:

```text
style27 / kp0 / seed64
512 x 512
12 denoise steps
1 inner Adam iter
disable_dynamic = true
disable_slt = true
adain_alpha = 0
disable_final_adain = true
style_grad_rms_target = 2.159326726314248e-06
```

Mode-specific notes:

```text
kv_match      = original matched K/V objective
score_match   = content_only negative/pre branch
ca_only       = content_only negative branch
ca_dmd        = content_only negative branch + 9B full fake critic
```

All four runs again hit the matched-gradient target at the first active inner
step:

```text
kv_match   effective_style_weight = 0.0421914
score      effective_style_weight = 0.8973290
ca_only    effective_style_weight = 5.6573154
ca_dmd     effective_style_weight = 5.6575353
applied weighted style grad RMS   = 2.159326726314248e-06
```

True-loss-only metrics:

```text
kv_match_true_loss_only
  CSD        = 0.143606
  CLIP-T     = 0.358907
  DINO style = 0.064747
  Gram dist  = 3.82295
  Layout     = 0.292638

score_match_content_only_true_loss_only
  CSD        = 0.128168
  CLIP-T     = 0.354548
  DINO style = 0.061678
  Gram dist  = 3.84842
  Layout     = 0.288704

ca_only_content_only_true_loss_only
  CSD        = 0.125281
  CLIP-T     = 0.356091
  DINO style = 0.062605
  Gram dist  = 3.86345
  Layout     = 0.289443

ca_dmd_content_only_true_loss_only
  CSD        = 0.125840
  CLIP-T     = 0.356595
  DINO style = 0.063016
  Gram dist  = 3.86666
  Layout     = 0.289098
```

The final-AdaIN confound was materially large. Relative to the earlier
`content_only` matched-gradient screen, CSD dropped for every method:

```text
kv_match   0.397564 -> 0.143606
score      0.354513 -> 0.128168
ca_only    0.359874 -> 0.125281
ca_dmd     0.354061 -> 0.125840
```

So the earlier "loss-isolation" screen should be read as:

```text
conditioning contrast fixed
but final global style-stat transfer still on
```

not as a true loss-only comparison.

Within the corrected true-loss-only screen:

```text
1. kv_match remains best on CSD, CLIP-T, DINO style similarity, and Gram distance.
2. score / CA / CA+DMD remain lower-layout variants, not stronger style winners.
3. therefore the bounded true-loss-only evidence still does not support
   "native score teacher > original K/V baseline".
```

Among the native-teacher variants only:

```text
1. score_match has the best CSD and lowest layout leak.
2. ca_dmd is slightly above ca_only on CSD, CLIP-T, and DINO style.
3. ca_dmd is slightly below ca_only on layout leak.
4. ca_dmd is slightly worse than ca_only on Gram distance.
```

The corrected image-space gaps are:

```text
pixel MAE(kv_match, score_match) = 9.659419
pixel MAE(kv_match, ca_only)     = 9.389676
pixel MAE(kv_match, ca_dmd)      = 9.476966
pixel MAE(score_match, ca_only)  = 2.090174
pixel MAE(score_match, ca_dmd)   = 2.228019
pixel MAE(ca_only, ca_dmd)       = 0.814027
```

The `ca_dmd` run still passed the bounded plumbing gate in this corrected
setting:

```text
fake backend         = klein_full_fake
fake init checkpoint = /mnt/local/klein9b-base
optimizer            = SGD
lr                   = 1e-3
grad clip            = 5
update_count         = 4
peak CUDA GB         = 70.821
all four critic updates decreased loss
sampled parameter updates stayed nonzero
```

Conclusion for the true-loss-only bounded screen:

```text
1. previous content_only loss-isolation retained a final AdaIN confound and
   should not be used as the loss-only answer.
2. after removing that confound, score / CA / CA+DMD still do not beat the
   matched K/V baseline on this sample.
3. single-sample bounded screening remains insufficient for any superiority
   claim.
4. CA+DMD plumbing is now validated, but its critic still resets per run and
   therefore is not yet a persistent fake distribution learner.
```

### 13.8 Next-stage protocol: persistent full critic / replay

Do not launch this yet. The next stage should be a reproducible protocol only:

```text
1. Real-data scope and split discipline
   development set = Hyperion 40 real style references
   fixed split = 32 train / 8 held-out by reference_sha256
   if family labels do not already exist, do not invent or pseudo-stratify them
   generated ext_styleshot = diagnostic only, never primary evidence
   MultiBanana 156 = fully held-out external test only
   9k identity references = excluded from this stage

2. Reproducible JSONL manifest
   every sample row must include:
     sample_id
     split
     source_dataset
     reference_path
     reference_sha256
     content_prompt
     prompt_id
     seed
     main_model_manifest_sha256
     teacher_model_manifest_sha256
     fake_init_model_manifest_sha256
   split isolation is by reference_sha256, not by rendered image path
   train / held-out must never share the same reference hash

3. Replay storage contract
   replay stores detached clean student latent x0 as the primary training state
   replay also stores conditioning + metadata:
     positive prompt condition
     negative/content-only prompt condition
     reference edit condition
     sample_id / split / reference_sha256 / prompt_id / seed
     originating outer step may be recorded as metadata
   replay must not primarily train on a fixed archived scoring z/t pair
   for replay, always resample independent epsilon and exact scheduler timestep/sigma
   flow target remains:
     target_v = epsilon - x0

4. Critic train / score separation
   critic train noise/timestep and fake scoring noise/timestep must be independent
   for each generator sample:
     first enqueue latest detached x0 + conditioning into replay
     then run K critic updates, with preregistered K = 4
   critic update defaults for the first persistent protocol:
     batch_size = 1
     gradient_accumulation = 4
     latest:replay = 1:1
     replay_capacity = 256
     replay sampling stratified by reference_sha256 and prompt_id
   these are tunable only if changed values are logged explicitly in the run summary

5. Full fake critic persistence
   fake critic starts from /mnt/local/klein9b-base
   use strict save / strict resume
   checkpoint every 64 critic updates with:
     full fake weights
     optimizer state
     CPU + CUDA RNG state
     replay buffer index / manifest
     checkpoint sha256 / manifest sha256
   add a save/resume equivalence test:
     uninterrupted N updates
     vs save at update m, resume, continue to N
     must match on critic loss and sampled parameter fingerprints within tolerance

6. Staged gates before TPAMI-full
   P0 plumbing gate:
     8 Hyperion references
     3 prompts each
     2 seeds each
     persistent critic + replay must run end-to-end
   P1 persistent-critic gate:
     full Hyperion 32 train / 8 held-out split
     held-out critic loss must beat both:
       reset-per-run critic
       no-replay persistent critic
     DM must stay non-zero and numerically stable on held-out probes
   only after P1 passes:
     run TPAMI-full comparisons
   MultiBanana 156 is the final external evaluation only, never the tuning stage

7. Compared methods and reporting matrix
   persistent CA+DMD must be compared against:
     reset CA+DMD
     CA-only
     kv_match
   and where needed against:
     score_match(full_style_prompt)
     score_match(content_only)
     ca_only(full_style_prompt)
     ca_only(content_only)
     ca_dmd(full_style_prompt)
     ca_dmd(content_only)
   report both:
     original-weight protocol
     matched-gradient protocol
   keep warm-start-off / loss-isolation and TPAMI-full side by side so warm-start
   dominance cannot hide teacher-direction weakness

8. Primary metrics, guardrails, and failure conditions
   primary metric = paired CSD
   guardrails:
     CLIP-T must not decrease
     layout leak must not increase
   additional reported metrics:
     VGG Gram distance
     DINO style similarity
   statistical reporting:
     paired bootstrap confidence intervals for all primary comparisons
   failure conditions for advancing:
     persistent CA+DMD does not beat reset CA+DMD on primary CSD
     or CLIP-T drops
     or layout leak rises
     or DM collapses to near-zero / unstable on held-out probes
```

Until that persistent-critic stage is run, the correct status remains:

```text
bounded screening only
teacher direction still weak
no positive superiority claim over kv_match
```
