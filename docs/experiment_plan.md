# Experiment Plan

## Stage 0: Sanity

Goal: make sure residual guidance is implemented without scheduler artifacts.

Run small set:

```text
styles: 00, 10, 27, 28
prompts: kp0, kp1, kp2
seeds: 43
resolution: 512
steps: 25
```

Baselines:

- text-only
- inject
- scfg fixed scale
- slt/ot-cfg fixed scale
- sds
- Hyperion/hybrid teacher

Metrics:

- CSD
- CLIP-T
- time
- forward/backward count

Required check:

- each timestep calls `scheduler.step` exactly once.

## Stage 1: Residual Field Diagnostics

For each active denoising step, record:

```text
delta_v_scfg = v_inject - v_normal
delta_v_otcfg = v_sinkhorn - v_normal
g_sds = -grad(style_loss)
g_hyperion = -grad(hyperion_style_loss)
delta_z_hyperion = z_after_hyperion - z_before_hyperion
```

Diagnostics:

```text
cos(delta_v_scfg, g_hyperion)
cos(delta_v_otcfg, g_hyperion)
cos(delta_v_scfg, delta_z_hyperion)
cos(delta_v_otcfg, delta_z_hyperion)
rho(delta_v_scfg)
rho(delta_v_otcfg)
rho(g_hyperion)
high_freq_energy(delta_v)
low_freq_energy(delta_v)
```

Success criterion:

- `delta_v_otcfg` has higher cosine with Hyperion teacher than naive injection.
- `delta_v_otcfg` has higher spatial neighbor cosine than SLT-only token update.
- residual fields show nontrivial high-frequency texture energy.

## Stage 2: Scale Sweep

Sweep:

```text
SCFG_SCALE: 0.5, 1.0, 2.0, 3.0, 5.0, 7.0
SLT_GUIDE_SCALE: 0.5, 1.0, 2.0, 3.0, 5.0, 7.0
INJECT_BETA: 0.3, 0.5, 0.7, 1.0
SLT_ALPHA: 0.05, 0.10, 0.20
SLT_TAU: 0.01, 0.02, 0.05
```

Output:

- CSD/CLIP-T Pareto curve
- residual norm distribution
- failure examples for over-guidance

Expected risk:

- large residual scale may cause layout drift or prompt semantic damage.

## Stage 3: Residual Purification

Test whether LRS/GRAG makes the residual more style-specific:

```text
GRAG_DEEP: 1.0, 0.7, 0.5, 0.3
GRAG_SHALLOW: 1.0, 0.75, 0.7
```

Measure:

- layout leakage
- CLIP-T drop
- CSD
- residual cosine to Hyperion gradient

Hypothesis:

LRS/GRAG should reduce layout-carrying components in `delta_v` while preserving
texture direction.

## Stage 4: Dataset Dump

Dump residual dataset over:

```text
styles: paper subset first, then full 39 styles
prompts: 30 prompts if available
seeds: 43, optionally 44/45
timesteps: all active steps or selected t bins
```

Shard naming:

```text
results/residual_dataset/shard_00000.pt
results/residual_dataset/metadata.jsonl
results/residual_dataset/summary.csv
```

Keep tensor dtype:

- bfloat16 for large velocity tensors
- float32 for diagnostics

## Stage 5: Distillation

Start with direction prediction, not full image quality.

Train:

```text
student(z_t, t, prompt_embed, style_embed) -> delta_v
```

Initial losses:

```tex
L = L_2 + 0.1 L_\mathrm{dir}
```

Validation:

- residual MSE
- direction cosine
- spatial neighbor cosine match
- frequency energy match
- one-step guided sampling quality

Only after this passes should we run full image generation metrics.

## Stage 6: Paper-Level Benchmarks

Compare:

- Hyperion teacher
- style CFG residual
- OT-CFG residual
- SDS one-backward
- distilled student
- trained adapters if needed

Report:

- CSD
- CLIP-T
- CLIP-I for I2I if included
- time / forward count / backward count
- spatial neighbor cosine
- field cosine to Hyperion teacher
- qualitative grid

