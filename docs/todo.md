# TODO

## Current Status

Created project directory:

```text
/home/ubuntu/chengyanli/items/style_residual_distill_20260709
```

Found relevant old-code prototypes:

```text
custom_main_klein9b_t2i.py: OPTIM_MODE=scfg
custom_main_klein9b_t2i.py: OPTIM_MODE=sds
custom_main_klein9b_t2i.py: OPTIM_MODE=slt
custom_main_klein9b_t2i.py: OPTIM_MODE=hybrid
```

Both Klein9B custom files pass `py_compile`:

```text
custom_main_klein9b.py
custom_main_klein9b_t2i.py
```

## Next Engineering Tasks

1. Copy minimal runtime into `src/klein_runtime.py`.
2. Copy attention hook/controller into `src/attention_hooks.py`.
3. Refactor residual computation into `src/residuals.py`.
4. Fix `scfg` and `slt` to compute normal/styled velocity before the scheduler
   step and call `scheduler.step` exactly once per timestep.
5. Add a debug flag to save per-step residual tensors.
6. Implement `scripts/profile_residual_fields.py`.
7. Run 4-style x 3-prompt sanity experiment.
8. Compare:

```text
delta_v_scfg
delta_v_otcfg
g_sds
g_hyperion
delta_z_hyperion
```

9. Write first result table:

```text
method, CSD, CLIP-T, time, fwd_count, bwd_count, rho, cos_to_hyperion
```

## Paper Notes

Use these terms consistently:

- `velocity residual` for implementation on Klein9B/FLUX.2
- `score residual` for probabilistic interpretation
- `style CFG` only as an analogy
- `style residual distillation` as the paper-level method name

Avoid claiming the residual is pure style until measured. The correct claim is:

```text
The residual can be made increasingly style-specific by conditioning design
and residual purification, and we quantify the remaining semantic/layout
components.
```

## Open Questions

1. Which style condition gives the best teacher residual?
   - hard K/V injection
   - Sinkhorn K/V injection
   - LRS/GRAG compressed K/V
   - Hyperion teacher displacement

2. Is `delta_v` aligned with Hyperion gradient across timesteps or only in a
   narrow active region?

3. Is the residual mostly high-frequency texture or also low-frequency color?

4. Does distilling `delta_v` work better than distilling `delta_z_hyperion`?

5. Does a student need prompt embeddings, or are `z_t`, `t`, and style features
   sufficient?

