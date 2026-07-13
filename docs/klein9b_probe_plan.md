# Klein9B Single-Sample Residual Probe Plan

This plan defines the next step after `profile_residual_fields.py --dry-run`
passes in the `diffsynth` and `yzh-dmd` environments.

The goal is not full benchmark generation. The goal is to extract trustworthy
per-timestep fields from one or a few samples:

```text
hard_scfg      = v_hard_inject - v_normal
otcfg          = v_sinkhorn - v_normal
hyperion_grad  = -grad(style_loss)
hyperion_delta = z_after_hyperion - z_before_hyperion
```

These fields then feed the existing profiler.

## Non-Negotiable Invariant

For every timestep:

```text
v_normal, v_hard_inject, v_sinkhorn
```

must be evaluated at the exact same `(z_t, t)` before any scheduler update.

The sampling loop may then call:

```python
z_next = scheduler.step(v_guided, t, z_t)
```

exactly once.

Do not reproduce the old double-step risk.

## Initial Probe Scope

Use the smallest meaningful scope:

```text
style_id: 27
prompt_id: kp0
seed: 43
resolution: 512
steps: 25
active timesteps: start with 3 selected timesteps, then all active timesteps
```

The first run should not decode images unless needed. It can dump field metrics
only.

## Required Runtime Components

Source from old project:

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948/custom_main_klein9b_t2i.py
```

Reuse or adapt:

- Klein9B model loading via `Flux2ImagePipeline.from_pretrained`
- local `KLEIN9B_ROOT`
- prompt embedding
- style latent VAE encoding
- image id construction
- `FlowMatchScheduler("FLUX.2")`
- style forward for reference K/V
- GRAG/LRS transform over style K/V

New project modules already available:

```text
src/attention_hooks.py
src/latent_ops.py
src/residuals.py
src/field_metrics.py
src/profiling.py
```

## Proposed New Script

```text
scripts/probe_klein9b_residuals.py
```

CLI:

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_klein9b_smoke
```

Default behavior:

- no image decode
- no full benchmark
- one prompt/style
- selected timesteps only
- write profiler-compatible CSV/JSON

Outputs:

```text
results/residual_profile_klein9b_smoke.csv
results/residual_profile_klein9b_smoke_summary.json
results/residual_profile_klein9b_smoke_meta.json
```

Optional tensor dump, disabled by default:

```bash
--dump-fields
```

When enabled:

```text
results/field_dumps/<out-prefix>_<prompt>_<style>_t<timestep>.pt
```

## Probe Algorithm

For each timestep `t_i`:

1. Keep a frozen copy:

```python
z_probe = latents.detach().clone()
```

2. Compute normal velocity:

```python
v_normal = compute_normal_velocity(..., latents=z_probe, timestep=t_i)
```

3. Compute style K/V at the matching style timestep:

```python
style_latent_t = scheduler.add_noise(style_latent, eps, t_i)
ref_k, ref_v = style_forward(style_latent_t)
ref_k, ref_v = apply_lrs_or_grag(ref_k, ref_v)
```

4. Compute hard-SCFG velocity:

```python
v_hard = compute_styled_velocity(..., mode="inject", ref_k_list=ref_k, ref_v_list=ref_v)
hard_scfg = v_hard - v_normal
```

5. Compute OT-CFG velocity:

```python
v_ot = compute_styled_velocity(..., mode="sinkhorn", ref_k_list=ref_k, ref_v_list=ref_v)
otcfg = v_ot - v_normal
```

6. Compute Hyperion teacher fields:

```python
z_before = z_probe.detach().clone()
z_h = maybe_slt_prealign(z_before)
g_h = -grad(style_loss(z_h, ref_k, ref_v))
z_after = adam_step(z_h, g_h)
hyperion_delta = z_after - z_before
```

The first smoke probe can compute only one Adam gradient and skip the optimizer
state if speed is a concern.

7. Profile:

```python
compare_fields(
    {
      "hard_scfg": hard_scfg,
      "otcfg": otcfg,
    },
    hyperion_grad=g_h,
    hyperion_delta=hyperion_delta,
    height=32,
    width=32,
)
```

8. Continue normal generation only if necessary. For field-only probing, a
deterministic normal denoising path is enough to reach later timesteps.

## Key Design Choices To Preserve

### Velocity, not score

Use `velocity` naming in code:

```text
v_normal
v_styled
delta_v
v_guided
```

Use score notation only in paper explanation.

### No scheduler side effects inside residual wrappers

`src/residuals.py` must remain scheduler-free.

### Style K/V as condition

The first implementation should use existing style K/V hooks rather than adding
IP-Adapter or a new style encoder. That isolates the residual-field question.

### Profile before image quality

Do not tune images until field metrics answer the first question:

```text
Is OT-CFG closer to Hyperion than hard-SCFG?
```

## First Acceptance Criteria

The smoke probe is considered useful if:

1. It completes for one style/prompt/seed without decoding images.
2. It writes profiler-compatible CSV/JSON.
3. CSV contains at least `hard_scfg` and `otcfg` rows.
4. Rows include non-null:
   - `residual_norm`
   - `rho`
   - `cos_to_hyperion_grad`
   - `hi_freq_ratio`
5. It uses exactly one scheduler step per timestep on the main path.

## Known Risks

- Style timestep alignment: old code sometimes used `timesteps[i+1]` for the
  style latent. For the probe, explicitly log which timestep was used.
- Memory: storing all fields at full precision is expensive. Default should
  compute metrics immediately and discard tensors.
- Hook state: always call `cache.clear(); controller.step(); controller.reset_mode()`
  around paired normal/styled forwards.
- Prompt conditioning: Hyperion teacher in old T2I sometimes used null embeds
  for the style loss. Log this choice rather than hiding it.

