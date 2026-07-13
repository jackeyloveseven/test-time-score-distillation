# Technical Plan

## 1. Main Problem Statement

Reference-based style transfer can be treated as conditional field steering.
Given a noisy latent `z_t`, timestep `t`, prompt `p`, and style image `x_s`, we
want a direction that changes visual style while preserving prompt semantics:

```tex
\Delta v_\mathrm{style}
= v_\theta(z_t,t,p;x_s)-v_\theta(z_t,t,p)
```

This is the style analog of CFG, except that the residual is not
`conditional - unconditional`. It is:

```text
style-conditioned prediction - content-only prediction
```

The central empirical question is whether this residual has the same desirable
properties as Hyperion's backward gradient:

- high spatial neighbor cosine
- high-frequency texture energy
- strong CSD improvement
- limited CLIP-T degradation
- limited layout leakage
- directional alignment with Hyperion's Adam update

## 2. Implementation Objects

### Normal Prediction

```python
v_normal = dit(
    hidden_states=z_t,
    timestep=t,
    encoder_hidden_states=prompt_embeds,
    txt_ids=text_ids,
    img_ids=image_ids,
)
```

### Style-Conditioned Prediction

The style condition can be constructed in several ways:

1. hard K/V injection
2. Sinkhorn attention injection
3. LRS/GRAG-compressed K/V injection
4. Hyperion-style SLT pre-aligned latent followed by style-conditioned forward
5. future image-conditioned adapter or style encoder

The first two already exist in `custom_main_klein9b_t2i.py` as `scfg` and `slt`.

### Residual

```python
delta_v = v_styled - v_normal
v_guided = v_normal + scale * delta_v
```

For FLUX.2/Klein9B, keep naming as velocity residual in code and paper.

## 3. Important Fix Before Experiments

The existing `inject` branch already computes the styled prediction before the
scheduler step. The `scfg` and `slt` branches should be refactored to follow the
same single-step structure:

1. compute `v_normal` at current `z_t, t`
2. compute `v_styled` at the same `z_t, t`
3. compute `v_guided`
4. call `scheduler.step(v_guided, t, z_t)` exactly once

Avoid stepping once with `v_normal` and then again with `v_guided` at the same
timestep. That would confound the residual baseline.

## 4. Teacher Signals

We need three teacher fields:

### A. CFG-Style Residual

```tex
\Delta v_\mathrm{CFG} = v_\mathrm{styled} - v_\mathrm{normal}
```

Generated with no backward pass.

### B. Hyperion Adam Gradient

```tex
g_\mathrm{Adam} = -\nabla_{z_t} \mathcal{L}_\mathrm{style}
```

This is the mechanism-proven teacher from the first paper.

### C. Hybrid Residual

```tex
\Delta z_\mathrm{Hyperion}
= z_t^\mathrm{after\;SLT+Adam} - z_t^\mathrm{before}
```

This is the actual latent displacement used by the best first-paper pipeline.

The second paper becomes strong if we show:

```tex
\cos(\Delta v_\mathrm{CFG}, g_\mathrm{Adam}) > 0
```

and that the residual's spatial coherence/frequency profile resembles Hyperion
more than feedforward injection.

## 5. Distillation Target

Train a small model `G_phi` to predict a style residual:

```tex
\mathcal{L}_\mathrm{distill}
= \|G_\phi(z_t,t,p,x_s) - \Delta v_\mathrm{style}\|_2^2
```

Optional direction-only loss:

```tex
\mathcal{L}_\mathrm{dir}
= 1 - \cos(G_\phi(z_t,t,p,x_s), \Delta v_\mathrm{style})
```

Optional spatial-coherence regularizer:

```tex
\mathcal{L}_\rho
= |\rho(G_\phi)-\rho(\Delta v_\mathrm{style})|
```

Initial student options:

- token MLP with timestep embedding
- lightweight DiT adapter over packed latents
- LoRA branch over selected MM-DiT blocks
- ControlNet-like side branch if stronger capacity is needed

Start with token MLP or shallow transformer. Do not begin with LoRA unless the
simple student clearly fails.

## 6. Paper Contributions Candidate

1. Formulate reference style transfer as conditional velocity residual guidance.
2. Show that style residuals recover spatially coherent texture fields similar
   to Hyperion's backward gradients.
3. Introduce style residual distillation, amortizing the teacher field into a
   lightweight student.
4. Demonstrate faster style transfer at comparable CSD/CLIP-T and lower
   forward/backward cost.

