# Migration Plan

Status: completed historical plan. The isolated repository now exists; do not
repeat these copy phases. See `AGENT_HANDOFF.md` for current boundaries.

This document decides what to copy from the Hyperion workspace and what to
leave behind.

## Source

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

## Destination

```text
/home/ubuntu/chengyanli/items/style_residual_distill_20260709
```

## Copy in Phases

### Phase 1: Minimal Klein9B Runtime

Copy or refactor only:

- local DiffSynth/Klein9B path setup
- model loading
- prompt embedding
- style VAE encoding
- image id construction
- FlowMatch scheduler loop
- save/decode utilities

Source candidates:

```text
custom_main_klein9b_t2i.py
custom_main_klein9b.py
```

Target modules:

```text
src/klein_runtime.py
src/prompting.py
src/latents.py
```

### Phase 2: Attention Hooking

Copy and clean:

- `Klein9BCache`
- `Klein9BController`
- `register_klein9b_attn_hooks`
- injection mode
- sinkhorn mode

Important cleanup:

- split cache/injection logic from experiment loop
- make controller state explicit
- support normal/styled pair forward without side effects

Target module:

```text
src/attention_hooks.py
```

### Phase 3: Residual Baselines

Port these branches from `custom_main_klein9b_t2i.py`:

- `scfg`: hard style-injection velocity residual
- `slt`: Sinkhorn style-injection velocity residual
- `sds`: one-backward style gradient baseline
- `hybrid`: Hyperion teacher displacement

Target module:

```text
src/residuals.py
```

Expected functions:

```python
compute_normal_velocity(...)
compute_styled_velocity(...)
compute_style_velocity_residual(...)
apply_velocity_guidance(...)
compute_sds_gradient(...)
compute_hyperion_teacher_update(...)
```

### Phase 4: Metrics and Diagnostics

Copy only the metrics needed for the second paper:

- CSD
- CLIP-T
- spatial neighbor cosine
- frequency decomposition
- residual cosine similarity
- layout leakage if retained

Source candidates:

```text
scripts/eval_all.py
scripts/eval_gradient_source.py
scripts/eval_freq_decompose.py
scripts/eval_spatial_viz.py
```

Target modules:

```text
src/metrics.py
src/diagnostics.py
```

### Phase 5: Dataset Dumping

Create a dataset writer that records:

```text
z_t
t
prompt_id / prompt text
style_id / style image path
v_normal
v_styled
delta_v
adam_gradient
hyperion_delta_z
rho(delta_v)
cos(delta_v, adam_gradient)
frequency stats
```

Target scripts:

```text
scripts/dump_residual_dataset.py
scripts/profile_residual_fields.py
```

Storage format:

- `.pt` shards for tensors
- `.jsonl` metadata
- `.csv` summary metrics

### Phase 6: Student

Do not start here. Only after residual fields are validated.

Target modules:

```text
src/student.py
src/train_distill.py
src/infer_student_guidance.py
```

## Do Not Copy Initially

- old gallery generation
- old baseline wrappers for external methods
- paper figure scripts unless needed
- full benchmark sweep scripts
- archive folders
- old CSD-in-loop experiments
- one-off parameter search images

## Copy Criteria

Before copying a file or function, answer:

1. Does it help compute a style residual?
2. Does it help compare residuals to Hyperion gradients?
3. Does it help evaluate the second-paper claim?
4. Does it help train or test a distilled residual predictor?

If all answers are no, leave it in the old project.
