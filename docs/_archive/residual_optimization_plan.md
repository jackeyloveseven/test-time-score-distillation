# Residual Optimization Plan

Status: historical residual/adapter exploration. GoalCarrier/`score_carrier` is
current; use `experiment_plan.md` for active experiments.

This document defines the first optimization track for the second paper:
turning raw style guidance into a style-specific velocity residual.

## Objective

Optimize the style residual so that it behaves like a style field rather than a
generic image-editing field.

Base residual:

```tex
\Delta v = v_\theta(z_t,t,p;x_s) - v_\theta(z_t,t,p)
```

Optimized residual:

```tex
\Delta v_\mathrm{style}
= \operatorname{Purify}(\Delta v)
```

The residual should increase style fidelity while minimizing semantic/layout
drift.

## Primary Hypothesis

OT-CFG should be a better style residual than hard-SCFG:

```tex
\Delta v_\mathrm{OT}
= v_\theta(z_t,t,p;\mathrm{Sinkhorn}(x_s)) - v_\theta(z_t,t,p)
```

because Sinkhorn/LRS-style conditioning reduces layout copying and emphasizes
distributional style statistics.

## Teacher Alignment

Hyperion's Adam gradient is the first-paper teacher:

```tex
g_H = -\nabla_{z_t}\mathcal{L}_\mathrm{style}
```

The main diagnostic is:

```tex
\cos(\Delta v, g_H)
```

If OT-CFG has higher alignment than hard-SCFG, the second-paper story is strong:
the model-native velocity residual recovers the mechanism-proven Hyperion field.

## Residual Variants

### A. hard-SCFG

Hard attention injection residual:

```tex
\Delta v_\mathrm{hard}
= v_\mathrm{inject} - v_\mathrm{normal}
```

Expected behavior:

- strong style signal
- higher layout leakage risk
- good baseline, not final method

### B. OT-CFG

Sinkhorn attention residual:

```tex
\Delta v_\mathrm{OT}
= v_\mathrm{sinkhorn} - v_\mathrm{normal}
```

Expected behavior:

- better distributional style direction
- less layout copying
- likely main residual for paper

### C. LRS/GRAG OT-CFG

OT-CFG with compressed style K/V residual:

```tex
K_s' = \mu_K + \gamma(K_s-\mu_K)
```

Expected behavior:

- lower layout leakage
- possible texture loss if gamma too small

### D. Frequency-Weighted OT-CFG

OT-CFG where high-frequency style similarity receives stronger weight.

Expected behavior:

- stronger brushstroke/texture
- artifact risk at high scale

### E. Hyperion Teacher

Backward-gradient or actual latent displacement:

```tex
g_H,\quad \Delta z_H
```

Used for alignment, not as the cheap inference method.

## Metrics

Field metrics:

- residual norm
- cosine to Hyperion gradient
- cosine to Hyperion latent displacement
- spatial neighbor cosine rho
- low/high frequency energy ratio
- timestep-wise stability

Image metrics:

- CSD
- CLIP-T
- runtime
- forward/backward count
- layout leakage if available

## First Required Experiment

Run a small probe:

```text
styles: 00, 10, 27, 28
prompts: kp0, kp1, kp2
seed: 43
steps: 25
active timesteps only
```

Compare:

```text
hard-SCFG residual
OT-CFG residual
LRS/GRAG OT-CFG residual
Hyperion Adam gradient
Hyperion latent displacement
```

Required output:

```text
results/residual_profile_small.csv
results/residual_profile_small.json
```

Core columns:

```text
style_id
prompt_id
timestep
variant
residual_norm
rho
cos_to_hyperion_grad
cos_to_hyperion_delta
hi_freq_ratio
low_freq_ratio
```

## Success Criteria

The next implementation phase is justified if:

1. OT-CFG has higher average cosine to Hyperion gradient than hard-SCFG.
2. OT-CFG has comparable or higher rho than hard-SCFG.
3. LRS/GRAG reduces obvious layout leakage without collapsing CSD.
4. Residual scale sweep shows a usable CSD/CLIP-T Pareto region.

## Do Not Start Yet

Do not begin student training until the field diagnostics show that the teacher
residual is meaningful. Distillation should train against a validated residual,
not against a raw heuristic.
