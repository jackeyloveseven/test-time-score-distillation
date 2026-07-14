# Reference-KV Conditioned Velocity Distillation Specification

Current status: historical K/V/student specification retained for provenance.
K/V is a baseline, not the paper method. Use `PROJECT_STATE.md` and current
source code for active contracts.

Status: implementation specification for the second-paper main method.

## Decision

The paper-facing student must not consume `style_id` or `prompt_id`. It must
consume the real reference image and a real text/content state. Removing prompt
ID does not justify removing prompt information: at the initial noise state,
the same `z_T` and reference can require different corrections for different
prompts.

The implementation therefore has two levels:

```text
reference image x_s
  -> VAE tokens z_s
  -> fixed reference noise epsilon_ref reused across timesteps
  -> z_s,t
  -> frozen Klein9B reference forward
  -> selected self-attention K/V tensors
  -> layout-suppressed reference representation C_ref(t, x_s, epsilon_ref)

V1 baseline:
  r_theta = R_theta(z_t, t, E_txt(prompt), C_global)

V2 paper candidate:
  C_goal = reference code at a fixed, style-identifiable canonical state
  C_dyn  = reference code at the current timestep
  r_theta = R_theta(h_hard or Q_hard, t, C_goal, C_dyn)

v_student = v_hard + alpha * r_theta
```

`E_txt(prompt)` is a frozen pooled/token representation from the same text path
used by Klein9B. `h_hard` or `Q_hard` is preferred in V2 because it is already
conditioned on the current content state and exposes the carrier query. IDs
remain metadata only for grouping, diagnostics, and split construction.

This is not an adapter. The content carrier remains the explicit hard-SCFG
field. The trainable component predicts only the teacher correction on top of
that carrier.

## Why Reference K/V

The reference K/V tensors are preferable to a new external image encoder for
the first main implementation:

1. They are computed from the actual reference image by the frozen backbone.
2. They are the same image representation used to construct hard and OT teacher
   branches, so conditioning and target share a clear causal path.
3. They already exist at inference; extracting a compact code adds negligible
   model compute.
4. They are timestep aligned with the target field.
5. An unordered representation can remove explicit reference coordinates while
   retaining appearance statistics and material modes.

An external frozen image encoder remains an ablation, not the default method.

## V1: Global-Moment Baseline

For each selected self-attention layer, keep only the final `N_img` tokens,
where `N_img` is the number of VAE reference tokens. Text tokens precede image
tokens in the current Klein9B attention sequence.

Partition the selected layer list into `B=3` contiguous depth bands. For each
band and each of K and V, compute mean and standard deviation jointly over the
layer, head, and image-token axes while preserving head dimension `D`:

```text
mu_K^b, sigma_K^b, mu_V^b, sigma_V^b in R^D
C_ref = concat_b(mu_K^b, sigma_K^b, mu_V^b, sigma_V^b)
```

With Klein9B head dimension `D=128` and three bands, V1 has dimension `1536`.
The implementation must derive the dimension from tensors instead of hard
coding it.

Required invariants:

```text
shape             = [batch, 4 * num_bands * head_dim]
layout positions  = not retained
dtype on disk     = float16 or bfloat16
training dtype    = float32
std definition    = population std (unbiased=False)
empty bands       = forbidden
K/V layer counts = equal
image token count = positive and <= sequence length
```

V1 intentionally retains coarse depth information and discards exact token
locations. It is an inexpensive, low-leakage baseline, not yet the assumed main
method. Averaging across layers and heads mixes unmatched learned coordinate
systems and may preserve little beyond energy statistics. Moment collisions can
also map distinct textures to the same code.

V1 must therefore be compared against no-reference, style-ID, wrong-reference,
and shuffled-reference controls before it is credited with using image content.

## V2: Unordered Appearance Prototypes

The paper candidate keeps token-level appearance modes without preserving the
reference's 2D coordinates or original token order:

1. Select one representative layer from each empirically measured depth region;
   do not assume equal early/middle/late thirds in the final experiment.
2. Remove text tokens and preserve head identity.
3. Project K/V tokens with region-specific projections.
4. Use set pooling with no positional encoding to obtain `M=4` unordered
   appearance prototypes per region.
5. Keep V1 moments as a global appearance bypass.
6. Query the prototypes with hard-branch content hidden states or Q tokens.
7. Use a shallow token mixer to predict the per-content-token correction.

The onset audit motivates a dual-code variant:

```text
C_goal = SetPool(K/V from a fixed canonical reference state)
C_dyn(t) = SetPool(K/V from the current noised reference state)
```

`C_goal` carries reference identity even when the current reference branch is
fully noised; `C_dyn(t)` tracks timestep-aligned appearance and can be gated by
measured identifiability. The canonical timestep is selected by the reference
onset curve, not assumed. Clean/canonical-only, dynamic-only, and dual-code are
required ablations.

```text
C_proto = SetPool({Proj_b(K_b,V_b)})       # unordered, no reference coordinates
m_i      = CrossAttention(Q_hard_i,{C_goal,C_dyn(t)})
r_i      = TokenMixer(h_hard_i,m_i,C_global,t)
```

This is content-aware matching without direct reference-layout copying. If hard
hidden/Q capture is not available in the first smoke implementation, the
minimum fallback is `z_t + frozen text embedding + token mixer`; prompt ID is
never an acceptable fallback.

## Data Contract

Each field dump adds:

```text
reference:
  code: Tensor[1, C_ref]
  code_version: "kv_stats_v1"
  code_dtype: string
  code_shape: [1, C_ref]
  num_bands: 3
  image_token_count: integer
  selected_layer_start: integer
  selected_layer_end: integer
  source: "klein9b_reference_self_attention_kv"
  reference_noise_seed: integer
  reference_noise_sha256: string
  kv_capture_site: string
  kv_normalization: string
  rope_state: string
  text_token_count: integer
  null_prompt_sha256: string
```

The existing fields remain unchanged:

```text
latents.z_t
fields.hard_scfg
fields.otcfg
teacher_hp_lam1 or reconstructible target metadata
```

Each manifest row records the reference image path and SHA-256, VAE checkpoint
and preprocessing, resize/crop policy, scheduler/timestep convention,
trajectory provenance, reference-noise identity, text encoder/tokenizer, null
prompt, and exact K/V hook site. A manifest builder must reject missing
reference codes by default. It may expose an explicit diagnostics-only override,
but training the main model must not use such rows.

The manifest keeps `style_id` and `prompt_id` only as metadata. The dataset
loader must return `reference_code`, not style/prompt indices.

## Student Contract

The V1 student is a token MLP because it is the strongest current latent120
architecture. Its conditioning is:

```text
c_t   = MLP_time(sinusoidal(t))
c_txt = MLP_text(E_txt(prompt))
c_ref = MLP_ref(LayerNorm(C_ref))
c     = MLP_cond(concat(c_t, c_txt, c_ref))

r_theta[token] = MLP(concat(LN(z_t[token]), c, position[token]))
```

Main forward signature:

```text
forward(z_t, timestep, text_condition, reference_code) -> correction
```

Forbidden in the main checkpoint/model:

```text
style_embedding
prompt_embedding
style_to_idx
prompt_to_idx
```

The target is written directly as the correction between teacher branches:

```text
r* = lambda * H_grid(v_ot(z_t,t,C_ref) - v_hard(z_t,t,C_ref))
v_teacher = v_hard + r*
v_student = v_hard + alpha * r_theta
```

`H_grid` is a local residual operator on the latent token grid. It is not
assumed to be an exact image-frequency decomposition, and the resulting field
is not assumed to be a conservative score field.

The initial image scale remains the frozen paired result:

```text
teacher lambda        = 1.00
inject_beta           = 0.50
teacher reconstruction alpha = 1.00
guidance extrapolation alpha  = 1.25
```

`alpha=1.00` is used for distillation fidelity and the first image gate.
`alpha=1.25` is a separately reported guidance extrapolation, not teacher
reconstruction. It must be revalidated after reference conditioning.

## Velocity and Score Boundary

The primary object is a reference-conditioned velocity correction. For a
schedule `z_t=alpha_t*x+sigma_t*epsilon`, velocity and score satisfy

```text
v_t = (alpha_dot/alpha) z_t
    + (alpha_dot*sigma^2/alpha - sigma_dot*sigma) s_t.
```

Thus `Delta v=b(t)Delta s` only for the same scheduler, time convention, and
identical `(z_t,t)`. The implementation must document Klein/FLUX.2's actual
`alpha_t`, `sigma_t`, direction, and velocity convention before using a formal
score-ratio claim. Until then, paper-facing terminology is
`CFG-like velocity guidance` and `velocity residual distillation`.

## Runtime Contract

At every denoising step:

1. Run the existing reference branch and capture K/V.
2. Reuse one deterministic `epsilon_ref` across all timesteps of this reference.
3. Build the reference representation with exactly the dump-time function.
4. Compute the hard velocity and capture its content hidden/Q state.
5. Predict the correction from the current content state, true text condition,
   and reference representation.
6. Form `v_student = v_hard + alpha*r_theta`.
7. Call `scheduler.step` exactly once.

Runtime metadata must save the reference code version, layer range, number of
bands/prototypes, K/V capture contract, reference-noise hash, checkpoint hash,
reference image hash, teacher lambda, inject beta, and effective alpha.

The main runtime does not require `v_normal`: it cancels algebraically in
`r*=H_grid(v_ot-v_hard)`. A normal branch may still be run for evaluation, but
must not be counted as required student NFE.

## Split Protocol

Random row splitting is only a smoke test. Paper-facing splits are grouped
before timestep expansion:

```text
A. held-out prompt/content, seen reference images
B. held-out exact reference images
C. held-out style families
D. cross-dataset real reference images
```

The same reference image hash must never cross train and validation/test. Near
duplicates should be filtered with a perceptual embedding threshold.

The first decisive comparison is:

```text
style-ID student vs reference-KV student
on held-out exact reference images
```

The style-ID student is expected to fail or be undefined on unseen references;
that is the point of the comparison, not an inconvenience to hide.

## Implementation Slices

### Slice 0: Teacher and conditioning audit

Before training, run full same-seed hard/OT/hybrid trajectories and verify that
the hybrid teacher improves the style-content Pareto frontier. Then measure
target variance across reference image, prompt, timestep, and reference-noise
view. The first quantitative heuristic is between-reference variance at least
three times within-reference-noise variance.

Initial Klein9B evidence on styles `00/13/27`, prompt `kp0`, and identical
content states shows a discontinuous onset: at `t=1000`, changing the reference
produces exactly zero correction change; by `t=909.7`, between-reference
variance is `38.4%` of correction energy. This evidence requires a broader
reference/prompt/noise replication, but it already forbids treating the fully
noised dynamic reference code as an always-valid visual goal.

### Slice 1: Pure V1 foundation

Add a pure tensor utility for `kv_stats_v1`, a reference-conditioned student,
checkpoint runtime, and CPU unit tests. Tests must cover shape, deterministic
output, no NaNs for constant K/V, rejection of malformed K/V, batching, forward
shape, checkpoint round trip, and absence of ID lookup tables.

### Slice 2: Teacher dump and manifest

Add the reference code to probe dumps. Build a dedicated reference manifest
that verifies code version/shape and hashes the source image. Preserve old dump
compatibility but reject old rows for main training.

### Slice 3: V1 training and evaluation

Train from `(z_t,t,E_txt,C_global)` with the existing hard-delta target and loss
family. Support random smoke, held-out reference hash, held-out family, and
held-out prompt splits. Save immutable split membership and all config in
checkpoints. Include no-reference, wrong-reference, and shuffled-reference
controls at identical capacity.

### Slice 4: Image sampling

Integrate the new runtime into the paired sampler without changing hard/normal
branches. Keep the existing style-ID mode available only as an ablation.

### Slice 5: V2 appearance-prototype student

Capture representative reference K/V and hard content hidden/Q. Add unordered
appearance prototypes, content-aware matching, and a shallow token mixer. V1
remains the global bypass and low-leakage baseline. Compare canonical goal,
dynamic reference, and dual-code conditioning.

### Slice 6: On-policy field aggregation

Collect student-trajectory states, evaluate the teacher correction at those same
states, append them to training, and repeat field regression. This is
trajectory-aware or on-policy field distillation, not DMD.

### Slice 7: Proper DMD, only if needed

Enter DMD only after measuring a material residual gap after field aggregation.
A proper DMD stage requires a target score, a separately and dynamically fitted
fake-distribution score on generated samples, and a generator distribution
objective. The existing DiffSynth DMD2 dual-optimizer implementation can be
adapted, but the decoupled image teacher must replace its target model.

## Acceptance Gates

Foundation gate:

```text
all CPU tests pass
old style-ID checkpoint still loads in its old runtime
main model state_dict contains no style/prompt ID embedding
dump/runtime reference codes match numerically on identical K/V
prompt condition changes the prediction at fixed z_T/reference
fixed reference noise reproduces identical code and teacher field
```

Teacher gate:

```text
hybrid teacher is Pareto-superior to hard/OT at matched content preservation
between-reference variance >= 3x within-reference-noise variance (heuristic)
prompt swap produces a measurable, modeled target change
```

Field gate:

```text
V1 beats no-reference correction MSE by >=10% on held-out exact references
wrong reference degrades direction agreement
V2 improves high-frequency/local field error by a further 5-10% or is rejected
```

Image gate:

```text
alpha=1 student retains >=70% of teacher's style gain over hard
without worse content preservation
reference-layout leakage is no worse than hard
guidance alpha=1.25 is reported separately from fidelity alpha=1
```

Paper gate:

```text
unseen-reference and unseen-family results reported
teacher/student speed and memory reported
main result reproducible from immutable config and split files
```

## Non-Goals

- No trainable adapter inserted into Klein9B.
- No style-ID fallback inside the main model.
- No cross-attention over ordered or position-coded reference image tokens.
- No universal claim for scale `1.25`.
- No DMD claim based only on offline field regression.
- No claim that latent-grid `H_grid` is a proven image-frequency or conservative
  score decomposition.
