# Agent Handoff: Style Residual Distillation

This is the authoritative takeover document for a new coding/research agent.
It describes what is implemented, what is measured, what is still only a
hypothesis, and how to run the repository without touching the original
Hyperion source.

## 1. Mission

The project is the second paper after Hyperion. The target problem is
image-conditioned style transfer on Klein9B / FLUX.2: use a real reference
image as a visual instruction while preserving the content prompt's objects
and layout.

The paper's current thesis is:

```text
editing is not progressively revealing the reference layer by layer
the native score forms the style goal early
the content state supplies the spatial carrier
late MM-DiT computation translates the goal into executable flow
```

The main method is online carrier-anchored native score refinement. It is
performed during sampling for the current reference and trajectory. It does
not require training a global adapter or student before inference.

## 2. Non-Negotiable Boundaries

### Repository

```text
repo = /home/ubuntu/chengyanli/items/style_residual_distill_20260709
host = ubuntu-107
```

All code and documentation changes go in `repo`. Before editing:

```bash
ssh ubuntu-107 git -C /home/ubuntu/chengyanli/items/style_residual_distill_20260709 status --short
```

Never reset or clean unrelated user changes. Generated results are ignored by
`.gitignore`; do not force-add checkpoints, images, or large field dumps.

### Original source

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

This is read-only reference material. In particular, do not edit its
`custom_main_klein9b.py`, copy files back into it, or use destructive Git/file
commands there. Style references are read from its `sty/` directory only.

### Runtime and models

The fixed runtime for current benchmark commands is:

```text
/home/ubuntu/chengyanli/flux/bin/python3
```

The default model paths used by the runner are:

```text
main model:    /mnt/local/klein9b
native teacher:/mnt/local/klein9b-base
style root:    /home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
```

The alternate environment `/home/ubuntu/miniconda3/envs/yzh-dmd/bin/python`
has also passed the lightweight runtime import check. Do not switch Python
environments in the middle of a matched benchmark.

## 3. What Is Actually Implemented

### Online path: main method

`scripts/run_tpami_with_style_teacher.py` runs the native teacher on the exact
same latent and timestep with and without the reference. The difference is a
model-native style field. The online loop can use these modes:

| mode | interpretation | paper status |
|---|---|---|
| `ca_only` | native score goal rendered through content carrier | main validated method |
| `score_carrier` | score matching plus explicit carrier objective | promising comparison |
| `ca_dmd` | CA term plus fake-score distribution correction | implementation path; needs formal validation |
| `score_match` | native with-style score matching without carrier | ablation |
| `kv_match` | reference K/V replacement baseline | legacy baseline |
| `carrier_score` | older carrier-score variant | exploratory compatibility mode |

The exact implementation is the source of truth. Use `--help` after any CLI
change and inspect `src/ca_dmd_backend.py` before claiming DMD.

### Optional offline path

`probe_native_edit_teacher.py` dumps native fields. The optional student path
then trains `student(z_t,t,p,c_s)` to approximate the field, where `c_s` is a
deterministic pooled FLUX VAE reference code. This path is for amortized
latency, not required for the main paper method.

Supported conditioning:

```text
style_id   memorization/control ablation
image_code real-reference conditioning path
```

Supported prediction modes:

```text
residual          predicts full field
hard_delta        predicts correction added to hard_scfg
carrier_anchored  predicts native goal, then content affinity renders it
```

## 4. Mathematical Contract

Let `z_t` be the exact current latent, `t` the exact timestep, `p` the content
prompt, and `x_s` the reference image. The native velocity residual is:

```text
Delta v_native = v_with_reference(z_t,t,p,x_s)
                 - v_without_reference(z_t,t,p)
```

The score-space residual uses the FLUX2 conversion implemented in
`src/flux2_score.py`:

```text
Delta s = -((1 - sigma) / sigma) Delta v_native
```

The branch pair must share the same latent, timestep, prompt, scheduler state,
and image IDs. If they do not, the difference is not a valid score/velocity
residual.

The content carrier is a detached token affinity built from the current content
latent:

```text
A(z_t) = softmax(normalize(z_t) normalize(z_t)^T / tau)
g_rendered = g_goal + lambda_CA (A(z_t) g_goal - g_goal)
```

The reference enters the goal branch, not the carrier affinity. This is the
mechanism intended to preserve content layout and avoid copying the reference
composition.

For rectified flow, score targets must be converted back to velocity before
the scheduler update. Do not add a score tensor directly to a velocity update.

## 5. Verified Commands

### Regression tests

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 -m py_compile src/*.py scripts/*.py
/home/ubuntu/chengyanli/flux/bin/python3 -m unittest -q \
  scripts.test_score_residual_student \
  scripts.test_tpami_with_style_teacher
```

Last verified result: `53 tests, OK`.

### Main online smoke

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 25 --iters 5 --start-time 1000 \
  --style-mode ca_only \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix online_ca_only_formal25_s27_kp0_seed64
```

For quick debugging, use `--num-steps 12 --iters 1` and a distinct
`--out-prefix`. Do not overwrite a prior run when diagnosing a change.

### Metric evaluation

The evaluator consumes a manifest with columns `method`, `output_path`,
`style_path`, `prompt`, `style_id`, `prompt_id`, and `seed`:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/evaluate_matched_teacher_outputs.py \
  --manifest-csv results/online25_ca_only_multiseed_manifest.csv \
  --out-csv results/online25_ca_only_multiseed_metrics.csv \
  --summary-json results/online25_ca_only_multiseed_metrics_summary.json \
  --device cuda
```

Reported metrics are CSD style similarity, CLIP-T, DINO style similarity,
VGG Gram distance, and DINO layout leakage. Check directionality in the JSON
summary; layout leakage and Gram distance are lower-is-better.

### Native field dump

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/probe_native_edit_teacher.py \
  --style-ids 00,13,27 --prompt-ids kp0,kp1,kp2,kp3,kp4 \
  --seed 65 --height 512 --width 512 --num-steps 12 \
  --dump-fields --dump-latents \
  --output-prefix results/native_carrier_goal_s65
```

Then build a manifest with `scripts/build_score_residual_manifest.py` and train
the optional image-code student:

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

This student command is optional and is not needed to reproduce the online
main method.

## 6. Current Evidence

### Online CA-only multi-seed

Protocol: styles/prompts `00/kp1`, `13/kp3`, `27/kp0`; seeds `64,65,66`;
25 outer steps; 5 inner iterations; `start-time=1000`; AdamW; weight decay
zero; SLT and AdaIN disabled.

```text
CSD style similarity: 0.5605 +/- 0.0662
CLIP-T:               0.3287 +/- 0.0171
DINO style similarity:0.0602 +/- 0.0225
VGG Gram distance:    3.9673 +/- 1.3332
DINO layout leakage: 0.3444 +/- 0.0363
```

Artifact files:

```text
results/online25_ca_only_multiseed_manifest.csv
results/online25_ca_only_multiseed_metrics.csv
results/online25_ca_only_multiseed_metrics_summary.json
```

### Native field / offline student

The current field collection uses 3 references, 5 prompts, 12 scheduler states,
and seed 65. The image-code field has latent shape `[1,1024,128]` and pooled
reference code shape `[256]`.

Random matched field sweep at scale `0.25`:

```text
style-ID carrier:  MSE 0.0000976, cosine 0.4486
image-code carrier:MSE 0.0000961, cosine 0.4988
```

Grouped holdout with all style 27 rows excluded:

```text
style-ID carrier:  MSE 0.0001211, cosine 0.3164
image-code carrier:MSE 0.0001206, cosine 0.3094
```

Interpretation: image-code conditioning works on an unseen reference at the
field/runtime level, but the current pooled VAE code does not yet establish a
universal image-quality advantage over style-ID conditioning.

### Native teacher comparison

The formal teacher note reports that `score_carrier` is stronger than the
K/V replacement baseline on the first matched three-pair validation for CSD,
CLIP-T, and DINO style, while Gram distance and layout leakage remain tradeoff
metrics. Treat this as a promising comparison, not a multi-seed dominance
claim. See `docs/formal25_teacher_comparison.md`.

## 7. Priority Queue for the Next Agent

### P0: Finish the online ablation matrix

Run the same nine `(style,prompt,seed)` cases for:

1. `score_carrier` with the documented start-time and carrier settings.
2. `ca_dmd` with persistent fake-score state.
3. `score_match` and `kv_match` as matched baselines where missing.
4. `carrier_loss_weight` sweep, with all other flags fixed.

Create one manifest per method, evaluate with the same evaluator, and add
mean/std tables to `docs/full_paper_draft.md`.

### P1: Validate DMD honestly

Before using the phrase “DMD loss” in a headline claim, record fake-score
backend, optimizer, update count, timestep sampling, replay state path, and
resume behavior. The current implementation has a backend path, but a formal
dataset-level dynamic fake-score validation is still a gate.

### P1: Build the image grid

For each method, render the same reference/prompt/seed cases and assemble a
visual grid containing reference, baseline, K/V baseline, CA-only,
score-carrier, and CA+DMD. Inspect object preservation and reference-layout
leakage before interpreting scalar metrics.

### P2: Strengthen image conditioning

Keep the pooled VAE mean/std code as the reproducible baseline. Add a stronger
frozen image encoder only as a controlled ablation, and preserve exact
unseen-reference and unseen-style-family splits.

### P2: Layer probes

The “goal early, carrier spatially, executor late” story is a motivated
hypothesis. Do not rewrite it as a causal empirical finding until the layer
probe and intervention experiments in `klein9b_probe_plan.md` are complete.

## 8. Common Failure Modes

- Comparing with-style and without-style predictions at different latent or
  timesteps invalidates the residual.
- Adding score-space tensors directly to the FLUX velocity scheduler breaks
  scale and can produce unstable images.
- `start-time=1000` is the current formal setting; the earlier `850` setting
  weakens the native score signal in the documented comparison.
- `--latent-weight-decay 0` is intentional. Nonzero AdamW decay is an explicit
  regularization ablation, not part of the main protocol.
- A new run should use a new `--out-prefix`; generated files are ignored and
  can otherwise be mistaken for fresh evidence.
- The original source directory is not the experiment workspace.
- A good field MSE does not prove good image-level style transfer. Always run
  image metrics and inspect a visual grid.
- Do not call the offline student necessary for the paper's main method.

## 9. Handoff Checklist

Before handing the project to another agent:

- `git status --short` is understood and unrelated changes are preserved.
- Source files compile and the two regression test modules pass.
- Every generated run has a unique prefix, manifest row, and log path.
- The method flags and environment are recorded in the relevant doc.
- Results are reported with exact seeds, prompts, styles, steps, and metrics.
- Claims distinguish validated CA-only, promising score-carrier, and unvalidated
  formal CA+DMD evidence.
- No file under `/home/ubuntu/chengyanli/items/save/attention_20260426_090948`
  was modified.
