# Style Residual Distillation

Second-paper workspace for formulating reference-based style transfer as a
conditional velocity/score residual problem, then distilling that residual into
a lightweight predictor.

## Core Thesis

Hyperion showed that high-quality style transfer needs spatially coherent latent
updates, and that a frozen MM-DiT can produce those updates through
backpropagation over its attention Jacobian. This project asks a different
question:

Can the same style update be expressed as a model-native conditional field
residual, analogous to classifier-free guidance, and then amortized by
distillation?

For Klein9B / FLUX.2, the primary object should be written as a velocity
residual:

```tex
\Delta v_\mathrm{style}(z_t,t,p,x_s)
= v_\theta(z_t,t,p;x_s)-v_\theta(z_t,t,p)
```

where `p` is the content prompt and `x_s` is a style reference image. The guided
velocity is:

```tex
v_\mathrm{guided}
= v_\theta(z_t,t,p) + \lambda \Delta v_\mathrm{style}(z_t,t,p,x_s)
```

In diffusion-score notation, the analogous expression is:

```tex
\Delta s_\mathrm{style}
= s_\theta(z_t,t\mid p,x_s)-s_\theta(z_t,t\mid p)
= \nabla_{z_t}\log p(x_s\mid z_t,p)
```

Use the velocity form in implementation and the score form only when discussing
the probabilistic interpretation.

## Relationship to Hyperion

Hyperion is the teacher and mechanism probe:

- It demonstrates that attention-Jacobian gradients create coherent texture.
- It supplies high-quality latent update fields.
- It provides metrics and diagnostics: CSD, CLIP-T, spatial neighbor cosine,
  frequency decomposition, layout leakage, and seed robustness.

This project should not copy the entire Hyperion codebase. It should import only
the components needed to build and evaluate residual fields.

## Initial Source Directory

Reference project:

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

New project:

```text
/home/ubuntu/chengyanli/items/style_residual_distill_20260709
```

Important source files:

```text
custom_main_klein9b.py        # I2I Hyperion-style implementation
custom_main_klein9b_t2i.py    # T2I implementation with scfg/sds/slt prototypes
scripts/eval_all.py           # evaluation harness and metrics integration
scripts/eval_gradient_source.py
scripts/eval_freq_decompose.py
scripts/eval_spatial_viz.py
scripts/eval_csd_per_iter.py
utils.py
```

## Project Layout

```text
src/          reusable implementation
scripts/      runnable experiments
configs/      named experiment configs
docs/         design notes and paper planning
results/      csv/json metrics
experiments/  one-off probes
assets/       prompt/style lists, small visual assets
```

## Immediate Rule

Every copied component must satisfy one of these roles:

- teacher model loading / scheduler loop
- style condition construction
- residual field computation
- Hyperion-gradient comparison
- metric computation
- dataset dumping for distillation
- student training or inference

Anything else stays in the old project.

