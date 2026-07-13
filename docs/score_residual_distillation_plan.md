# Score Residual Distillation Plan

## Paper Framing Update (2026-07-10)

The paper story is now centered on a Goal-Carrier-Flow Executor decomposition:

```text
editing is not layer-by-layer revealing
early blocks expose the reference-derived visual goal
the content state supplies the spatial carrier for clean rendering
late blocks translate goal-bearing image code into executable flow
```

The main method is a carrier-anchored image-conditioned score correction:

```text
r* = Delta v_teacher - Delta v_carrier
r_theta = R_theta(z_t, t, text_code, image_code)
v = v_normal + Delta v_carrier + alpha * r_theta
```

This places the work in Think with Image rather than finite style-ID transfer.
The current style-ID MLP is a proof-of-learnability scaffold. The paper model
must consume a real reference image and be evaluated on unseen reference images
and unseen style families.

Detailed story, equations, claim gates, reproducibility contract, three
contributions, and experiment plan:

```text
docs/paper_story_goal_carrier_flow.md
```

## Position

Adapter is not the main method. It is a narrow baseline:

```text
adapter baseline:
pred = hard_scfg + g_theta(hard_scfg, t, style_id, prompt_id)
```

This is useful for ablation because existing residual-only dumps already contain
`hard_scfg` and `teacher_hp_lam1`. It does not prove the stronger claim that
style transfer can be represented as a learned score/velocity correction at the
current denoising state.

The paper-aligned main line should be carrier-anchored, image-conditioned
score/velocity residual distillation:

```text
carrier:
Delta v_carrier = v_hard - v_normal

student correction:
r_theta(z_t, t, prompt, image_code) ~= Delta v_style* - Delta v_carrier

inference:
v = v_normal + Delta v_carrier + alpha * r_theta(z_t, t, prompt, image_code)
```

Direct prediction of the full `Delta v_style*` remains an important ablation.
The current `style_id` conditioning remains a diagnostic scaffold until the
reference-image encoder path is implemented.

## CFG Interpretation

Classifier-free guidance can be written as a correction on the base score:

```text
score_cfg = score_uncond + w * (score_cond - score_uncond)
```

For style transfer, the analogous object is a style-conditioned correction:

```text
score_style = score_content + Delta score_style(z_t, t, prompt, style)
```

In velocity-parameterized Klein9B/Flux-style code, the trainable target is the
corresponding velocity residual:

```text
Delta v_style* = v_teacher_style - v_normal
```

The current teacher proxy is:

```text
hard_scfg = v_hard - v_normal
otcfg = v_ot - v_normal
Delta v_style* = hard_scfg + highpass(otcfg - hard_scfg)
```

The 45-sample sweep currently selects `teacher_hp_lam1` as the best teacher
proxy. Later DMD-style or human/preference teachers can replace this target
without changing the student interface.

## Required Data Schema

Residual-only dumps are insufficient for the main method because they do not
store the denoising state. Direct score residual training requires each sample
to include:

```text
z_t                       # latent at the same point where fields are measured
t                         # scheduler timestep
prompt_id / prompt_text
style_id / style_path
fields.hard_scfg
fields.otcfg
fields.teacher_hp_lam1    # or another selected teacher target
height, width             # latent spatial grid
```

`scripts/probe_klein9b_residuals.py` therefore supports:

```bash
--dump-fields --dump-latents
```

When enabled, each `.pt` field dump contains:

```text
latents.z_t
latents.z_t_shape
latents.z_t_dtype
has_latents = true
```

Older dumps with `has_latents` missing or false are still valid for adapter
baselines but should not be used for direct score residual student training.

Manifest building for the direct path should use:

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/build_score_residual_manifest.py \
  --field-dumps 'results/field_dumps/YOUR_LATENT_DUMPS_*.pt' \
  --out-prefix score_residual_dataset_lam1 \
  --results-dir results \
  --teacher-kind highpass_delta \
  --teacher-lambda 1.0 \
  --teacher-highpass-kernel 5 \
  --verify-teacher
```

This builder rejects residual-only dumps by default.

## Student Families

The implementation order should be:

1. Direct residual MLP/token net:

```text
input:  z_t, t embedding, prompt/style ids or embeddings
target: teacher_hp_lam1
loss:   MSE + cosine + optional spectral/highpass loss
```

This is the fastest way to verify that direct `z_t -> residual` is learnable.

2. CFG correction head:

```text
input:  z_t, t, prompt/style conditioning, optional hard_scfg
target: teacher_hp_lam1 - hard_scfg or teacher_hp_lam1
```

This explicitly tests whether hard attention style CFG is a good base branch.

3. DiT LoRA residual distillation:

```text
train: selected DiT blocks or residual LoRA modules
target: teacher velocity/residual at sampled timesteps
inference: one forward path with learned style correction
```

This is the best paper-facing version if the direct residual target is stable.

## Immediate Experiments

- Regenerate real field dumps with `--dump-fields --dump-latents`.
- Build a manifest with `scripts/build_score_residual_manifest.py`, which rejects samples without `latents.z_t`.
- Train a direct residual student against `teacher_hp_lam1` with:

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_real_3step_style27_kp0.jsonl \
  --out-dir results/score_residual_student_real_3step_style27_kp0 \
  --epochs 200 \
  --batch-size 1 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.003 \
  --weight-decay 0 \
  --val-fraction 0.34 \
  --device cuda
```

- Keep the existing adapter result as a baseline:

```text
adapter baseline val_mse ~= 0.4233 at scale 0.75
hard baseline val_mse ~= 0.4376
```

The direct student must beat this baseline and show image-level improvement
before it becomes the paper's main empirical claim.

## Current Direct Student Evidence

Implemented:

- `src/score_residual_student.py`
- `scripts/train_score_residual_student.py`

The first direct student is a token MLP with timestep/style/prompt conditioning
and learned token position embeddings. Position embedding is required because
the target residual is a spatial field; a position-free token MLP only receives
the local `z_t` vector and has weak capacity to model style layout/frequency
structure.

Smoke evidence:

```text
single real sample:
results/score_residual_student_real_overfit_capacity
best_epoch = 163
best_metric ~= 0
hard_baseline_mse = 0.556178
```

This proves the direct student path can overfit a measured Klein9B residual
field.

Three-timestep evidence:

```text
manifest = results/score_residual_manifest_real_3step_style27_kp0.jsonl
samples = 3
train = 2 timesteps
val = 1 timestep

best val_mse = 0.780175 at epoch 48
final train_mse = 0.010613
final train_hard_baseline_mse = 0.456898
final val_mse = 1.339989
final val_hard_baseline_mse = 0.583576
```

Interpretation:

- The direct student can fit real teacher residuals.
- This tiny dataset does not generalize across timestep; hard CFG remains a
  stronger validation baseline.
- Next improvement should be data scale and structured student capacity, not
  adapter-first training.

Recommended next step:

```text
collect >= 3 styles x 5 prompts x 3 timesteps with --dump-latents
train direct student with validation split by prompt/style/timestep
then compare against hard_scfg and adapter baseline
```

## 45-Sample CFG Correction Result

The first full direct-score dataset has been collected:

```text
manifest = results/score_residual_manifest_latent45_lam1.jsonl
num_samples = 45
styles = 00:15, 13:15, 27:15
prompts = kp0:9, kp1:9, kp2:9, kp3:9, kp4:9
timesteps = 363.147003:15, 472.075500:15, 718.322083:15
latent_shape = [1, 1024, 128]
target = teacher_hp_lam1
```

Two direct student objectives were tested on the same random 80/20 split:

```text
plain residual:
pred = f_theta(z_t, t, prompt, style)

hard CFG correction:
pred = hard_scfg + f_theta(z_t, t, prompt, style)
```

Results:

```text
plain residual
  out_dir = results/score_residual_student_latent45_lam1_plain
  best_epoch = 92
  val_mse = 0.547354
  val_hard_baseline_mse = 0.418083
  val_cosine = 0.778109
  val_hard_baseline_cosine = 0.838770

hard CFG correction
  out_dir = results/score_residual_student_latent45_lam1_hard_delta
  best_epoch = 12
  val_mse = 0.390818
  val_hard_baseline_mse = 0.418083
  val_mse_improvement = 0.027265
  val_cosine = 0.854317
  val_hard_baseline_cosine = 0.838770
```

Interpretation:

- Directly predicting the full style residual from `z_t` is still too hard at
  this data/model scale.
- The CFG-form correction objective is already better than hard CFG on the
  validation split.
- This supports the paper direction: style transfer can be framed as
  `base score + learned style correction`, then distilled from the teacher
  score/velocity field.

Current recommended method statement:

```text
Delta v_teacher = hard_scfg + highpass(otcfg - hard_scfg)
Delta v_student = hard_scfg + f_theta(z_t, t, prompt, style)
v = v_normal + alpha * Delta v_student
```

Adapter remains an ablation:

```text
adapter: hard_scfg + g_theta(hard_scfg, t, prompt, style)
direct CFG correction: hard_scfg + f_theta(z_t, t, prompt, style)
```

## Group Holdout Validation

Random validation split is positive but not sufficient. The training script now
supports grouped validation:

```bash
--split-mode prompt --holdout-value kp4
--split-mode style --holdout-value 27
--split-mode timestep --holdout-value 363
```

These runs keep the same direct CFG correction objective:

```text
Delta v_student = hard_scfg + f_theta(z_t, t, prompt, style)
```

Summary file:

```text
results/score_residual_student_latent45_hard_delta_holdout_summary.json
```

Results:

```text
random split
  train/val = 36/9
  best_epoch = 12
  val_mse = 0.390818
  hard_baseline_mse = 0.418083
  mse_improvement = 0.027265
  cosine_improvement = 0.015547

holdout prompt kp4
  train/val = 36/9
  best_epoch = 7
  val_mse = 0.397504
  hard_baseline_mse = 0.420769
  mse_improvement = 0.023265
  cosine_improvement = 0.013331

holdout style 27
  train/val = 30/15
  best_epoch = 6
  val_mse = 0.410652
  hard_baseline_mse = 0.436566
  mse_improvement = 0.025914
  cosine_improvement = 0.015162

holdout timestep 363.147003
  train/val = 30/15
  best_epoch = 7
  val_mse = 0.307673
  hard_baseline_mse = 0.334272
  mse_improvement = 0.026599
  cosine_improvement = 0.015623
```

Interpretation:

- The correction student beats hard CFG under random, prompt-holdout,
  style-holdout, and timestep-holdout validation.
- The consistent improvement supports a stronger claim than the initial random
  split: the learned correction is not only memorizing paired residual fields.
- The effect size is still small, so the next engineering target is a stronger
  correction architecture or DiT/LoRA-side residual head, not more adapter
  tuning.

## Token-Mixer Student Ablation

The direct score student now supports a spatial token-mixer architecture:

```text
model_type = token_mixer
input = z_t + timestep/style/prompt/position conditioning
mixing = depthwise spatial convs on the latent grid + pointwise channel mixing
output = f_theta(z_t,t,prompt,style)
```

This tests whether local spatial interaction helps the direct CFG correction
objective, since the original token MLP predicts each latent token independently
except for learned position conditioning.

Implemented:

- `ScoreResidualTokenMixer`
- `build_score_residual_student(config)`
- `scripts/train_score_residual_student.py --model-type token_mixer`
- runtime loading through the same `ScoreResidualStudentRuntime`

Old checkpoints remain compatible because `model_type` defaults to
`token_mlp` when missing from the saved config.

Main random split run:

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_latent45_lam1.jsonl \
  --out-dir results/score_residual_student_latent45_lam1_hard_delta_token_mixer \
  --epochs 120 \
  --batch-size 3 \
  --hidden-dim 384 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --model-type token_mixer \
  --mixer-layers 4 \
  --mixer-kernel 3 \
  --dropout 0.05 \
  --lr 0.0015 \
  --weight-decay 0.0001 \
  --val-fraction 0.2 \
  --prediction-mode hard_delta \
  --device cuda
```

Random split comparison:

```text
token MLP hard_delta
  best_epoch = 12
  val_mse = 0.390818
  hard_baseline_mse = 0.418083
  improvement = 0.027265

token mixer hard_delta
  best_epoch = 10
  val_mse = 0.389370
  hard_baseline_mse = 0.418083
  improvement = 0.028714
```

Group holdout comparison for token mixer:

```text
holdout prompt kp4
  best_epoch = 5
  val_mse = 0.398062
  hard_baseline_mse = 0.420769
  improvement = 0.022708

holdout style 27
  best_epoch = 4
  val_mse = 0.412534
  hard_baseline_mse = 0.436566
  improvement = 0.024032

holdout timestep 363.147003
  best_epoch = 7
  val_mse = 0.305963
  hard_baseline_mse = 0.334272
  improvement = 0.028309
```

Interpretation:

- Token mixer gives the best random-split and timestep-holdout MSE so far.
- It is slightly worse than the token MLP on prompt/style holdout, so it should
  not replace the current default checkpoint yet.
- Training overfits quickly after the first 5-10 epochs. Spatial mixing is
  useful, but the 45-sample dataset is too small to exploit a higher-capacity
  mixer reliably.
- The right next move is to collect a larger latent manifest before making
  token mixer the main architecture, or to add stronger regularization/gating.

Image-level smoke:

```text
results/student_samples/score_student_mlp_vs_token_mixer_style27_kp0_s48_contact.png
```

On `style27/kp0/seed48`, token mixer preserves content and gives a similar
visual result to the token MLP. It is currently a structural ablation rather
than a clear qualitative upgrade.

## 120-Sample Style Expansion

The 45-sample dataset was expanded with five additional style categories:

```text
old styles = 00,13,27
new styles = 02,05,08,22,34
prompts = kp0,kp1,kp2,kp3,kp4
timesteps = 718.322083, 472.075500, 363.147003
seed = 45
```

Collection command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_style_sweep.py \
  --probe-mode run-model \
  --style-ids 02,05,08,22,34 \
  --prompt-ids kp0,kp1,kp2,kp3,kp4 \
  --seed 45 \
  --selected-timesteps 700,500,300 \
  --out-prefix score_residual_latent120_extra_styles_lam1 \
  --results-dir results \
  --teacher-lambdas 1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 3 \
  --runtime-height 512 \
  --runtime-width 512 \
  --min-free-gb 40 \
  --dump-latents
```

New style-sweep result:

```text
results/score_residual_latent120_extra_styles_lam1_global_teacher_ranking.json
best teacher = teacher_hp_lam1
score = 0.726073
```

Merged manifest:

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/build_score_residual_manifest.py \
  --field-dumps "results/field_dumps/score_residual_latent45_lam1*_probe_*.pt" \
  --field-dumps "results/field_dumps/score_residual_latent120_extra_styles_lam1*_probe_*.pt" \
  --out-prefix score_residual_manifest_latent120_lam1 \
  --results-dir results \
  --teacher-kind highpass_delta \
  --teacher-lambda 1.0 \
  --teacher-highpass-kernel 5 \
  --verify-teacher
```

Manifest summary:

```text
manifest = results/score_residual_manifest_latent120_lam1.jsonl
num_samples = 120
styles = 00,02,05,08,13,22,27,34 each 15
prompts = kp0,kp1,kp2,kp3,kp4 each 24
timesteps = 363.147003,472.075500,718.322083 each 40
field_shape = [1,1024,128] for all samples
latent_dtype = float16 for all samples
verified_teacher = true
```

Random split training:

```text
latent45 token MLP hard_delta
  val_mse = 0.390818
  hard_baseline_mse = 0.418083
  improvement = 0.027265
  val_cosine = 0.854317
  hard_baseline_cosine = 0.838770

latent120 token MLP hard_delta
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_mlp
  best_epoch = 35
  val_mse = 0.368352
  hard_baseline_mse = 0.406864
  improvement = 0.038513
  val_cosine = 0.861675
  hard_baseline_cosine = 0.841386

latent120 token mixer hard_delta
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_token_mixer
  best_epoch = 24
  val_mse = 0.369588
  hard_baseline_mse = 0.406864
  improvement = 0.037276
  val_cosine = 0.861949
  hard_baseline_cosine = 0.841386
```

Holdout style 34:

```text
latent120 token MLP
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_mlp_holdout_style34
  best_epoch = 19
  val_mse = 0.399988
  hard_baseline_mse = 0.430052
  improvement = 0.030065
  val_cosine = 0.861231
  hard_baseline_cosine = 0.844364

latent120 token mixer
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_token_mixer_holdout_style34
  best_epoch = 16
  val_mse = 0.401241
  hard_baseline_mse = 0.430052
  improvement = 0.028811
  val_cosine = 0.862225
  hard_baseline_cosine = 0.844364
```

Interpretation:

- Expanding style coverage is the clearest improvement so far: random-split
  MLP MSE improves from `0.390818` to `0.368352`.
- The MLP also improves the held-out style 34 validation by `0.030065` over
  hard CFG, which is stronger evidence for style generalization than the
  earlier 3-style dataset.
- Token mixer remains useful as an ablation but still overfits more easily and
  does not beat the MLP on MSE.
- The current default checkpoint should be upgraded to:

```text
results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt
```

Image-level check:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 48 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_latent120_mlp_recommended_s48 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_batch_latent120_mlp_recommended_s48_summary.json
results/student_samples/score_student_batch_latent120_mlp_recommended_s48_contact.png
```

Visual conclusion:

- The latent120 MLP checkpoint preserves prompt content across the same 9
  style/prompt combinations used for the seed48 latent45 comparison.
- No new style-reference content leak was observed at
  `inject_beta=0.5, score_student_scale=1.0`.
- The image-space effect remains a modest correction on top of hard CFG. The
  stronger empirical gain is currently in residual-space teacher matching and
  held-out style validation.

New-style image batch:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 02,05,08,22,34 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 49 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_latent120_mlp_newstyles_s49 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_batch_latent120_mlp_newstyles_s49_summary.json
results/student_samples/score_student_batch_latent120_mlp_newstyles_s49_contact.png
```

Ratio summary:

```text
ratio_mean range = 0.2383 to 0.4772
ratio_mean average = 0.3303
strongest style = 08
weakest style = 22
```

Visual conclusion:

- The five additional style categories preserve prompt content under the
  current default setting.
- No severe style-reference content leak was observed across the 15 new-style
  cases.
- The correction strength varies by style. Style 08 receives the strongest
  correction, while style 22 is relatively conservative.
- This supports using the latent120 checkpoint as the current default for both
  the original and newly added style categories.

### Scale calibration

Implemented:

```text
scripts/evaluate_score_residual_student.py
```

This script evaluates:

```text
Delta v_student(scale) = hard_scfg + scale * f_theta(z_t,t,prompt,style)
```

against the teacher residual on the selected validation split.

Random split sweep with the latent120 MLP checkpoint:

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/evaluate_score_residual_student.py \
  --manifest results/score_residual_manifest_latent120_lam1.jsonl \
  --checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --out-prefix score_residual_student_latent120_mlp_eval_random \
  --results-dir results \
  --student-scales 0,0.25,0.5,0.75,1,1.25,1.5,1.75,2 \
  --batch-size 6 \
  --device cuda \
  --seed 43 \
  --val-fraction 0.2
```

Result:

```text
scale  val_mse   improvement_vs_hard
0.00   0.406864  0.000000
0.50   0.374702  0.032162
0.75   0.368300  0.038564
1.00   0.368352  0.038513
1.50   0.387813  0.019051
1.75   0.407223 -0.000359
2.00   0.433086 -0.026222
```

Held-out style 34 slice:

```text
scale  val_mse   improvement_vs_hard
0.00   0.430052  0.000000
0.50   0.385284  0.044768
0.75   0.373071  0.056982
1.00   0.367637  0.062415
1.25   0.368983  0.061069
1.50   0.377109  0.052943
2.00   0.413701  0.016351
```

Image batch with `score_student_scale=0.75`:

```text
results/score_student_batch_latent120_mlp_s0p75_s48_summary.json
results/student_samples/score_student_batch_latent120_mlp_s0p75_s48_contact.png
```

Scale conclusion:

- `scale=0.75` and `scale=1.0` are both safe at image level for the current
  9-case seed48 batch.
- `scale=0.75` is slightly better on the random validation split.
- `scale=1.0` is better on the style34 slice and gives marginally stronger
  correction.
- Do not use `scale >= 1.75`; residual-space matching collapses back to or
  below hard CFG.

Current image default remains:

```text
inject_beta = 0.5
score_student_scale = 1.0
checkpoint = results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt
```

Conservative setting:

```text
inject_beta = 0.5
score_student_scale = 0.75
```

### Per-style scale check

Implemented optional style-specific scale overrides:

```text
scripts/sample_student_images.py --score-student-style-scales "08:0.75,22:1.25"
scripts/run_score_image_batch.py --score-student-style-scales "08:0.75,22:1.25"
```

The override is recorded in sample metadata:

```text
score_student_style_scales
score_student_effective_scale
step_summaries[*].score_student_scale
```

Smoke test:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sample_student_images.py \
  --style-id 02 \
  --prompt-id kp0 \
  --seed 50 \
  --height 512 \
  --width 512 \
  --num-steps 4 \
  --out-prefix score_student_style_scale_smoke_style02_kp0_s50 \
  --results-dir results \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --score-student-scale 1.0 \
  --score-student-style-scales 02:0.5 \
  --inject-beta 0.5 \
  --modes normal,hard,score_student
```

Verified metadata:

```text
score_student_effective_scale = 0.5
score_student_style_scales = {"02": 0.5}
step scale = 0.5 for all sampled steps
```

Per-style residual-space sweep using the latent120 MLP checkpoint:

```text
style=00 best_scale=1.00 best_mse=0.378285 hard=0.429302 improvement=0.051017
style=02 best_scale=1.00 best_mse=0.350537 hard=0.411211 improvement=0.060674
style=05 best_scale=1.00 best_mse=0.317695 hard=0.365963 improvement=0.048268
style=08 best_scale=1.00 best_mse=0.303077 hard=0.367472 improvement=0.064395
style=13 best_scale=1.00 best_mse=0.343779 hard=0.401224 improvement=0.057446
style=22 best_scale=1.00 best_mse=0.327990 hard=0.367699 improvement=0.039709
style=27 best_scale=1.00 best_mse=0.358062 hard=0.436376 improvement=0.078314
style=34 best_scale=1.00 best_mse=0.367637 hard=0.430052 improvement=0.062415
```

Conclusion:

- The code now supports per-style scale maps for future experiments.
- Current per-style calibration does not justify a nontrivial map: all eight
  measured styles select `scale=1.0`.
- No new image batch is needed for a per-style map because the calibrated map
  is equivalent to the existing global `score_student_scale=1.0` default.

### Timestep window support

Motivation:

- The latent120 score student was trained on three teacher timesteps:
  `718.322083`, `472.075500`, and `363.147003`.
- The 12-step image sampler applies the student at timesteps from `1000.0` down
  to `395.470`.
- A timestep window is useful as a safety/diagnostic switch: only apply the
  learned score correction where teacher supervision exists.

Implemented optional runtime controls:

```text
scripts/sample_student_images.py --score-student-timestep-min
scripts/sample_student_images.py --score-student-timestep-max
scripts/run_score_image_batch.py --score-student-timestep-min
scripts/run_score_image_batch.py --score-student-timestep-max
```

Smoke command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sample_student_images.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 51 \
  --height 512 \
  --width 512 \
  --num-steps 4 \
  --out-prefix score_student_timestep_window_smoke_style27_kp0_s51 \
  --results-dir results \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --score-student-scale 1.0 \
  --score-student-timestep-max 718.4 \
  --inject-beta 0.5 \
  --modes normal,hard,score_student
```

Smoke meta:

```text
score_student_timestep_min = null
score_student_timestep_max = 718.4
step scales = [(1000.0,0.0),(958.085,0.0),(883.982,0.0),(717.497,1.0)]
```

Same-seed image batch comparison, `styles=00,13,27`, `prompts=kp0,kp1,kp3`,
`seed=52`, `beta=0.5`, `scale=1.0`:

```text
full:
  outputs = results/score_student_batch_latent120_mlp_full_s52_summary.json
            results/student_samples/score_student_batch_latent120_mlp_full_s52_contact.png
  active_step_count = 12 / 12
  step_scale_mean = 1.0
  scaled_ratio_mean = 0.348995
  scaled_ratio_range = 0.262272 to 0.470553

timestep max 718.4:
  outputs = results/score_student_batch_latent120_mlp_twmax718_s52_summary.json
            results/student_samples/score_student_batch_latent120_mlp_twmax718_s52_contact.png
  active_step_count = 3 / 12
  step_scale_mean = 0.25
  scaled_ratio_mean = 0.137189
  scaled_ratio_range = 0.113911 to 0.162451
```

Visual conclusion:

- Both full and `timestep_max=718.4` preserve prompt content in this batch.
- The windowed setting mainly weakens the learned correction to about 39% of
  the full average scaled ratio.
- No clear image-level safety gain was observed for this seed; the full setting
  also did not introduce obvious content copying.

Current recommendation:

```text
default: keep score_student_timestep_min/max unset
diagnostic/conservative option: --score-student-timestep-max 718.4
```

This keeps the main result on the stronger full-runtime correction while
retaining timestep gating for ablations or riskier checkpoints.

## Image-Level Runtime

Implemented:

- `src/score_student_runtime.py`
- `scripts/sample_student_images.py` mode `score_student`

Runtime formula:

```text
hard_scfg(z_t,t) = v_hard(z_t,t) - v_normal(z_t,t)
correction = f_theta(z_t,t,prompt,style)
v_score_student = v_hard + score_student_scale * correction
```

The sampler keeps the same invariant as the residual probes: each branch uses
the same branch-local `(z_t,t)` for `v_normal`, `v_hard`, and `f_theta`, then
calls `scheduler.step(...)` exactly once per timestep.

Smoke outputs:

```text
inject_beta = 1.0
results/student_samples/score_student_image_style27_kp0_s1_steps12_contact.png

inject_beta = 0.5
results/student_samples/score_student_image_style27_kp0_s1_beta0p5_steps12_contact.png
```

Pixel/stat sanity:

```text
beta=1.0
  normal mean = [97.53, 100.11, 75.46]
  hard mean = [199.68, 142.14, 57.34]
  score_student mean = [199.33, 144.75, 52.89]
  score_delta / hard_residual norm ratio mean = 0.2526

beta=0.5
  normal mean = [97.53, 100.11, 75.46]
  hard mean = [128.68, 114.40, 75.85]
  score_student mean = [126.87, 115.81, 73.55]
  score_delta / hard_residual norm ratio mean = 0.3756
```

Qualitative observation:

- `inject_beta=1.0` overpowers the prompt/content and copies too much style
  reference content.
- `inject_beta=0.5` preserves the panda content while still applying visible
  style shift.
- Therefore image-level optimization should tune the base CFG strength
  (`inject_beta`) together with the learned correction scale; only tuning the
  correction scale is not enough.

## Image Grid Sweep

Grid runner:

```text
scripts/run_score_image_grid.py
```

Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_grid.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 46 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_grid_style27_kp0 \
  --results-dir results \
  --betas 0.35,0.5,0.65 \
  --scales 0.5,1.0,1.5 \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_grid_style27_kp0_summary.json
results/student_samples/score_student_grid_style27_kp0_contact.png
```

Numeric summary:

```text
beta=0.35:
  hard mean = [112.58, 106.17, 75.54]
  score means:
    scale 0.5 = [111.85, 106.37, 74.13]
    scale 1.0 = [111.36, 106.71, 73.14]
    scale 1.5 = [110.85, 107.40, 72.30]

beta=0.5:
  hard mean = [128.68, 114.40, 75.85]
  score means:
    scale 0.5 = [127.93, 115.14, 74.62]
    scale 1.0 = [126.87, 115.81, 73.55]
    scale 1.5 = [125.92, 116.32, 72.49]

beta=0.65:
  hard mean = [178.84, 138.04, 66.08]
  score means:
    scale 0.5 = [179.01, 140.03, 62.82]
    scale 1.0 = [178.33, 141.64, 60.42]
    scale 1.5 = [176.37, 142.29, 57.44]
```

Visual conclusion:

- `inject_beta=0.35` best preserves panda content; style strength is mild.
- `inject_beta=0.5` is the best current tradeoff: panda content remains clear
  and style shift is visible.
- `inject_beta=0.65` crosses into content leak and reproduces the style
  reference house/background instead of the panda prompt.
- `score_student_scale` mostly changes texture/color intensity within each beta
  row. It is not the main content/style control knob.

Current recommended image setting:

```text
inject_beta = 0.5
score_student_scale = 1.0
```

Conservative content-preserving setting:

```text
inject_beta = 0.35
score_student_scale = 1.0 or 1.5
```

### Latent120 grid runner update

The grid runner has been updated for the current direct-score mainline:

- default checkpoint now points to
  `results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt`
- supports `--score-student-style-scales`
- supports `--score-student-timestep-min/max`
- summary now records `score_student_effective_scale`, raw/scaled residual
  ratios, active step count, and image-level L1/L2 diffs:
  `hard_vs_normal_diff`, `score_vs_hard_diff`, `score_vs_normal_diff`

Smoke command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_grid.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 53 \
  --height 512 \
  --width 512 \
  --num-steps 8 \
  --out-prefix score_student_grid_latent120_smoke_s53 \
  --results-dir results \
  --betas 0.45,0.55 \
  --scales 0.75,1.0 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_grid_latent120_smoke_s53_summary.json
results/student_samples/score_student_grid_latent120_smoke_s53_contact.png
```

Smoke summary:

```text
beta=0.45 scale=0.75 scaled_ratio=0.3145 score_vs_hard_l1=7.310 hard_vs_normal_l1=40.210
beta=0.45 scale=1.00 scaled_ratio=0.4148 score_vs_hard_l1=9.301 hard_vs_normal_l1=40.210
beta=0.55 scale=0.75 scaled_ratio=0.2643 score_vs_hard_l1=6.632 hard_vs_normal_l1=51.065
beta=0.55 scale=1.00 scaled_ratio=0.3484 score_vs_hard_l1=8.496 hard_vs_normal_l1=51.065
```

Visual smoke result:

- All four samples preserve panda content.
- `beta=0.55` gives stronger style transfer than `beta=0.45`.
- `scale=1.0` gives a slightly larger learned correction than `0.75` without
  changing composition in this smoke.

Use this updated grid runner for future beta/scale/timestep-window sweeps.

### Latent120 beta refinement

Refined grid, style `27`, prompt `kp0`, seed `54`, 12 steps:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_grid.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 54 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_grid_latent120_beta_refine_style27_kp0_s54 \
  --results-dir results \
  --betas 0.45,0.5,0.55,0.6 \
  --scales 0.75,1.0,1.25 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_grid_latent120_beta_refine_style27_kp0_s54_summary.json
results/student_samples/score_student_grid_latent120_beta_refine_style27_kp0_s54_contact.png
```

Metric trend:

```text
beta=0.45 hard_vs_normal_l1=43.510 scaled_ratio_avg=0.4453
beta=0.50 hard_vs_normal_l1=48.774 scaled_ratio_avg=0.3986
beta=0.55 hard_vs_normal_l1=57.498 scaled_ratio_avg=0.3610
beta=0.60 hard_vs_normal_l1=64.986 scaled_ratio_avg=0.3299
```

Visual conclusion:

- `beta=0.60` is too aggressive for this style/prompt; panda face/background
  are visibly dominated by the style reference.
- `beta=0.55` gives stronger style but has higher content-risk than `0.50`.
- `beta=0.45` is the best content-preserving row; `scale=1.25` recovers some
  learned correction strength without obvious composition drift in this grid.

Candidate batch:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 55 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_latent120_beta0p45_scale1p25_s55 \
  --results-dir results \
  --inject-beta 0.45 \
  --score-student-scale 1.25 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_batch_latent120_beta0p45_scale1p25_s55_summary.json
results/student_samples/score_student_batch_latent120_beta0p45_scale1p25_s55_contact.png
```

Candidate batch summary:

```text
scaled_ratio_mean avg = 0.472435
scaled_ratio_mean range = 0.355063 to 0.655053
active_step_count = 12 / 12
```

Candidate visual conclusion:

- All 9 old style/prompt combinations preserve the main prompt content.
- Compared with hard CFG, the score student gives a clearer color/texture
  refinement than the lower-scale default.
- This is a good conservative-enhanced candidate:

```text
conservative candidate: inject_beta=0.45, score_student_scale=1.25
historical status: stable on old and new cross-seed batches
current status: superseded by the strict paired comparison below
```

New-style validation, styles `02,05,08,22,34`, prompts `kp0,kp1,kp3`, seed
`56`, 12 steps:

```text
outputs = results/score_student_batch_latent120_beta0p45_scale1p25_newstyles_s56_summary.json
          results/student_samples/score_student_batch_latent120_beta0p45_scale1p25_newstyles_s56_contact.png

ratio_mean avg         = 0.358307
scaled_ratio_mean avg  = 0.447884
scaled_ratio_mean range= 0.327278 to 0.644651
active_step_count      = 12 / 12
```

Per-style average scaled ratio:

```text
style02 = 0.473969
style05 = 0.407075
style08 = 0.562728
style22 = 0.364128
style34 = 0.431518
```

Visual conclusion:

- All 15 new style/prompt combinations preserve the requested panda, castle,
  and sunflower content.
- Style `08` has the strongest correction but does not show obvious reference
  layout takeover in this batch; style `22` is the weakest but remains visible.
- Together with the old-style batch, this promotes `beta=0.45, scale=1.25` from
  a one-grid candidate to a robust candidate. It is not yet the paper default
  because these batches are not a same-seed paired comparison against
  `beta=0.50, scale=1.00`.

### Cross-seed normalized image-difference audit

While the same-seed default batch is waiting for GPU memory, the existing four
batches were compared with two within-sample normalized quantities:

```text
refine_fraction = L1(score_student, hard) / L1(hard, normal)
extension_fraction = [L1(score_student, normal) - L1(hard, normal)]
                     / L1(hard, normal)
```

These remove much of the raw image-distance scale but do not remove seed
effects, so they are diagnostic rather than configuration-selection evidence.

```text
                         refine mean  refine median  refine std  extension mean
default old, seed48        0.216747      0.214812      0.038697       0.000053
candidate old, seed55      0.318217      0.315683      0.036526      -0.001441
default new, seed49        0.267347      0.239298      0.103885      -0.040083
candidate new, seed56      0.285010      0.277664      0.070898      -0.018979
```

Additional stability signal:

```text
default new refine_fraction range   = 0.162547 to 0.581006
candidate new refine_fraction range = 0.171076 to 0.402359
```

Interpretation:

- On old styles, the candidate produces about `46.8%` more correction relative
  to the hard branch while leaving the average normal-distance extension near
  zero. It is stronger without simply moving the result farther away globally.
- On new styles, the mean refinement rises only modestly, but the spread and
  worst outlier shrink. This is consistent with a more stable correction.
- This supports keeping `0.45/1.25` as the leading candidate, but it cannot
  replace the required same-seed comparison.

The first same-seed default smoke attempt (`style27/kp0`, seed `55`) failed
during transformer loading because unrelated processes occupied about 113 GiB
of the 144 GiB GPU. `--vram-limit-gb 20` still left insufficient headroom. No
sample was generated and no method conclusion should be drawn from this OOM.

### Lossless low-VRAM loading fix

Root cause: `--vram-limit-gb` was passed to
`Flux2ImagePipeline.from_pretrained`, but its `ModelConfig` entries did not
specify an offload dtype/device. DiffSynth therefore never enabled its VRAM
manager and loaded the 8B text encoder and 9B transformer onto CUDA together.

Implemented:

```text
src/klein9b_loading.py
scripts/sample_student_images.py
scripts/probe_klein9b_residuals.py
```

When `--vram-limit-gb` is supplied, text encoder and transformer now use:

```text
offload:     BF16 on CPU
onload:      BF16 on CUDA
preparing:   BF16 on CUDA
computation: BF16 on CUDA
```

This is lossless with respect to model dtype; it does not use FP8
quantization. Metadata records:

```text
model_loading_mode = cpu_offload_bf16
vram_limit_gb = requested threshold
```

Validation:

- remote `py_compile` passes for the helper, sampler, and probe;
- probe dry-run writes all CSV/JSON/meta outputs;
- a 12-step `style27/kp0` sample completes with only the previously available
  28 GiB GPU headroom;
- paired offload runs produce pixel-identical normal images (`L1=L2=max=0`).

### Strict same-seed three-config comparison

Eight styles were evaluated on prompt `kp0` using one numerical path
(`cpu_offload_bf16`) and 12 active student steps:

```text
default:      inject_beta=0.50, score_student_scale=1.00
conservative: inject_beta=0.45, score_student_scale=1.25
balanced:     inject_beta=0.50, score_student_scale=1.25

old styles 00,13,27: seed=55
new styles 02,05,08,22,34: seed=56
```

Outputs:

```text
results/score_student_paired_default_vs_candidate_8styles_kp0_offload16_summary.json
results/score_student_paired_three_config_8styles_kp0_offload16_summary.json
results/student_samples/score_student_paired_three_config_8styles_kp0_offload16_contact.png
```

Aggregate image-space result:

```text
                                     default   conservative   balanced
refine_fraction                       0.209310     0.288188     0.254383
palette histogram distance            32.6389      35.4462      33.1203

conservative vs default:
  hard carrier magnitude              -8.87%
  learned refinement                  +26.10%
  final displacement from normal       -9.09%

balanced vs default:
  learned refinement                  +21.62%
  final displacement from normal       +0.11%
```

Visual conclusion across the eight reference rows:

- Conservative `0.45/1.25` protects content but visibly weakens palette on
  color-dependent styles such as `00`, `13`, and `02`.
- Balanced `0.50/1.25` retains the default carrier/palette while giving clearer
  line, texture, and tonal refinement on `13`, `27`, `02`, `05`, `22`, and
  `34`.
- The simple RGB histogram is content-confounded and should remain diagnostic;
  it correctly warns about the conservative palette loss but does not capture
  the balanced setting's texture improvement.
- Balanced changes the final normal-relative displacement by only `+0.11%` on
  average, so the extra learned refinement is not obtained through a global
  increase in edit magnitude.

Current configuration decision:

```text
baseline:              beta=0.50, scale=1.00
balanced leader:       beta=0.50, scale=1.25
conservative fallback: beta=0.45, scale=1.25
status: superseded by the completed multi-prompt freeze validation below
```

Method implication: `inject_beta` controls the carrier and much of the
low-frequency palette, whereas `score_student_scale` controls the learned
teacher correction. They should be calibrated as separate axes. Lowering beta
whenever student scale rises discards useful carrier information; keeping beta
at `0.50` and strengthening only the student is the cleaner operating point.

### Balanced multi-prompt freeze validation

The balanced setting was then tested on `kp1` and `kp3` with the same eight
styles and strict same-seed pairing. Combined with the preceding `kp0` batch,
the freeze decision covers 24 pairs:

```text
styles  = 00,13,27,02,05,08,22,34
prompts = kp0,kp1,kp3
steps   = 12

baseline = inject_beta 0.50 / score_student_scale 1.00
balanced = inject_beta 0.50 / score_student_scale 1.25
```

The runs used lossless BF16 CPU offload only because other jobs left about
28 GiB free on the 144 GiB H200. This is an execution option, not a method
requirement: with an idle GPU, the same experiment can use the normal loading
path without `--vram-limit-gb`.

Outputs:

```text
results/score_student_balanced_prompt_validation_24pairs_offload16_summary.json
results/student_samples/score_student_balanced_promptval_kp1_8styles_offload16_contact.png
results/student_samples/score_student_balanced_promptval_kp3_8styles_offload16_contact.png
```

Aggregate result over all 24 pairs:

```text
default refine_fraction                 0.217179
balanced refine_fraction                0.262665
learned refinement change               +21.03%
final displacement from normal change    -0.10%
refinement wins                          24 / 24
active student steps                     12 / 12 in every run
```

Prompt-wise learned-refinement changes are `+21.62%` on `kp0`, `+20.13%` on
`kp1`, and `+21.34%` on `kp3`; each prompt has `8/8` wins. Every style has
`3/3` wins across prompts. Style-averaged final-displacement changes remain
near zero (approximately `-0.92%` to `+0.62%`).

Visual review agrees with the trajectory measurements. Castle and sunflower
composition remain stable, while balanced sampling gives a subtle but
consistent increase in style-specific line, texture, and tonal treatment. No
obvious reference-layout takeover or content collapse appears. The RGB
histogram distance is slightly worse in most pairs (`3/24` wins), but this
metric is content-confounded and does not measure the observed texture/line
gain; it must not be presented as the primary style metric.

Frozen configuration for the current latent120 image experiments:

```text
recommended paper/default experiment = beta=0.50 / scale=1.25
paired baseline                       = beta=0.50 / scale=1.00
conservative fallback                 = beta=0.45 / scale=1.25
```

Scope: this freezes the operating point for the current eight-style,
three-prompt latent120 evaluation. It is not a claim that `1.25` is a universal
style-conditioned scale. The main paper should report the paired scale sweep
and retain reference-conditioned or timestep-conditioned calibration as a
possible extension.

## Recommended-Setting Batch

To check whether the recommended setting generalizes beyond one prompt, a fixed
setting batch was run:

```text
inject_beta = 0.5
score_student_scale = 1.0
styles = 00,13,27
prompts = kp0,kp1,kp3
seed = 47
num_steps = 12
```

Runner:

```text
scripts/run_score_image_batch.py
```

Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 47 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_recommended_s47 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_batch_recommended_s47_summary.json
results/student_samples/score_student_batch_recommended_s47_contact.png
```

Observed behavior:

- All 9 style/prompt combinations keep the prompt content recognizable.
- No case shows the severe style-reference content leak seen at
  `inject_beta=0.65`.
- The learned score correction changes color/texture subtly relative to hard
  CFG; it does not dominate composition.
- Ratio `||score_delta|| / ||hard_residual||` is in the range `0.3155` to
  `0.4447` across these cases.

Representative rows:

```text
style=00 prompt=kp0 ratio=0.3155
style=13 prompt=kp1 ratio=0.3614
style=27 prompt=kp3 ratio=0.4140
```

Interpretation:

- `inject_beta=0.5, score_student_scale=1.0` is a reasonable default for the
  current model.
- The correction is currently modest at image level. Stronger visible gains
  likely require a stronger student architecture or a teacher target that
  better separates style texture from style-reference content.

### Seed 48 replication

The same recommended setting was rerun with a different generation seed:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 48 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_recommended_s48 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --force
```

Outputs:

```text
results/score_student_batch_recommended_s48_summary.json
results/student_samples/score_student_batch_recommended_s48_contact.png
```

Ratio summary:

```text
style=00 prompt=kp0 ratio_mean=0.3171 ratio_min=0.2347 ratio_max=0.5787
style=00 prompt=kp1 ratio_mean=0.3522 ratio_min=0.2764 ratio_max=0.4805
style=00 prompt=kp3 ratio_mean=0.3540 ratio_min=0.2819 ratio_max=0.6028
style=13 prompt=kp0 ratio_mean=0.3510 ratio_min=0.2625 ratio_max=0.6250
style=13 prompt=kp1 ratio_mean=0.3797 ratio_min=0.2862 ratio_max=0.5320
style=13 prompt=kp3 ratio_mean=0.3523 ratio_min=0.2643 ratio_max=0.6017
style=27 prompt=kp0 ratio_mean=0.3957 ratio_min=0.2803 ratio_max=0.7359
style=27 prompt=kp1 ratio_mean=0.4502 ratio_min=0.3431 ratio_max=0.6516
style=27 prompt=kp3 ratio_mean=0.4171 ratio_min=0.3164 ratio_max=0.7339
```

Visual replication:

- The seed 48 contact sheet matches the seed 47 conclusion: prompt content is
  preserved across all 9 style/prompt combinations at `inject_beta=0.5`.
- No sample shows the severe style-reference content copying observed at
  `inject_beta=0.65`.
- The score student remains a secondary correction on top of hard CFG. Its
  effect is visible mostly as color, contrast, and texture changes.
- Style 27 produces the largest correction ratios, which is consistent with
  its stronger visual style shift in the contact sheet.

Current image-level claim:

```text
Across seeds 47 and 48, beta=0.5 / score_student_scale=1.0 is stable enough
for qualitative examples. The learned correction improves residual-space
teacher matching and gives modest but consistent image-space style refinement.
It is not yet strong enough to be the only style-control mechanism.
```
