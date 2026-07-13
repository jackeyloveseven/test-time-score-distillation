# Paper Story Blueprint: Goal-Carried Image Score Distillation

Status: second-paper research blueprint, updated 2026-07-10.

This document defines the paper story, method boundary, reproducibility
contract, contribution claims, and experiment plan. It deliberately separates
what the current code has already verified from what must still be implemented
before a claim enters the paper.

## One-Sentence Thesis

Image-conditioned editing is not a layer-by-layer rendering process: the model
forms the target visual goal early, a content-derived spatial carrier determines
whether that goal can be rendered without layout leakage, and late blocks
translate the goal-bearing image code into an executable flow field.

Our method follows this computation rather than treating reference injection as
a monolithic feature blend: it builds a decoupled image-CFG teacher, keeps its
spatial carrier explicit, and distills the remaining reference-conditioned
score/velocity correction into a student.

## Relation to Hyperion

The two papers should form a progression rather than two disconnected style
transfer systems.

Hyperion asks:

```text
Why does training-free injection fail to render coherent style texture?
```

Its answer is that the attention backward path supplies a spatially coherent
carrier that feed-forward injection lacks. Sinkhorn transport helps set global
appearance, while the attention Jacobian determines whether that appearance can
be propagated coherently across the image.

The second paper asks:

```text
Can the expensive image-conditioned edit field exposed by Hyperion be
represented as a CFG-like score correction and amortized by distillation?
```

Its answer should be broader than "we trained a smaller predictor." Hyperion's
observations suggest a computational decomposition:

```text
reference image -> visual goal
content state   -> spatial carrier
late MM-DiT     -> flow execution code
```

The new method turns this decomposition into an image-conditioned score
correction. The first paper diagnoses the missing mechanism; the second paper
applies the correction online while preserving that mechanism. A learned
student is retained only as an optional amortization path for repeated
deployment.

## Motivation: Editing Is Not Progressive Revealing

A common implicit story is that deeper transformer layers progressively reveal
the edited image. That story predicts that the target attribute, its spatial
placement, and the final denoising action should emerge together.

Our working observation is different:

1. **Goal appears early.** Style identity and other target attributes become
   decodable in early hidden states, before the final velocity is determined.
2. **Carrier controls clean rendering.** The content branch's spatial token
   organization decides where the goal is allowed to act. A poor carrier causes
   reference layout leakage, broken object boundaries, or texture pasted into
   semantically wrong regions.
3. **Execution happens late.** The last blocks translate an already available
   goal/image code into the velocity update consumed by the rectified-flow
   scheduler.

This observation gives the paper a front-to-back narrative:

```text
Intro question:
If the edit goal is already present, why is the edit still rendered badly?

Method answer:
Because having a goal is not enough; it must be transported by the right
spatial carrier and translated into the right flow correction.

Experiment answer:
Layer probes locate the goal, causal carrier swaps determine clean placement,
and late-layer patching identifies the conversion into executable velocity.

Conclusion:
Thinking with an image is a goal-carrier-execution process, not progressively
drawing the reference through the network depth.
```

The "goal appears early" statement is currently a hypothesis motivated by the
existing implementation and must be gated by the layer-probe experiments below.
It should be written as an empirical finding only after those curves and causal
interventions are complete.

## Think with Image Positioning

The core field is **Think with Image**: using an image as instruction-bearing
evidence that changes an ongoing generation or edit, rather than reducing it to
a class ID or using it only as a retrieval key.

Style transfer is the controlled testbed because it exposes all three parts of
the problem:

- the reference provides a visual goal such as palette, material, texture, and
  mark-making;
- the prompt/content state provides object boundaries and spatial layout;
- the generative model must convert both into a valid denoising flow.

The main paper model must therefore consume a real reference image at inference.
The current `style_id` embedding is a proof-of-learnability scaffold only. It
cannot establish image-conditioned reasoning or generalization to an unseen
reference.

Paper-facing conditioning:

```text
e_s = E_img(x_style)
r_theta = R_theta(z_t, t, E_txt(prompt), e_s)
```

`E_img` should be frozen and deterministic in the first reproducible version.
The main test split must hold out exact reference images, and one protocol must
hold out complete style families. Style-ID conditioning remains only as an
upper-bound or memorization ablation.

## Goal, Carrier, and Flow Executor

### Goal

The goal is the non-spatial or weakly spatial visual intent extracted from the
reference image. It describes what appearance should be realized, not where
reference pixels should be copied.

Operational evidence:

- reference/style retrieval accuracy from hidden states by layer;
- linear-probe accuracy for palette, texture, material, and style family;
- cross-reference clustering that is stable under content changes;
- early-layer activation patching that changes style identity while preserving
  the content layout.

### Spatial Carrier

The carrier is the content-derived spatial organization that transports the
goal across latent tokens while respecting the generated scene.

Operational evidence:

- spatial neighbor cosine of the induced residual field;
- object-boundary alignment of residual magnitude;
- content-layout preservation under carrier swaps;
- failure after token shuffling or replacement with the reference layout;
- recovery when the content carrier is restored while keeping the same goal.

The carrier is not a trainable adapter. In the first method version it is the
explicit hard-SCFG/content-attention correction already available from the
teacher pipeline.

### Flow Executor

The executor is the late network computation that translates the goal carried
through the current image state into the velocity consumed by the scheduler.

Operational evidence:

- late-block contribution to final velocity residual;
- velocity reconstruction from patched intermediate states;
- late-layer removal preserving goal decodability but destroying edit quality;
- late-layer patching changing the final flow while leaving early goal probes
  almost unchanged.

"Early," "carrier," and "late" must be selected from measured transition
points, not assigned by an arbitrary fixed block number.

## Score/CFG Formulation

Let `p_0(z_t | p)` be the content/prompt-conditioned base distribution and
`p_T(z_t | p, x_s)` the desired reference-conditioned edit distribution. The
ideal style correction is a conditional log-density-ratio score:

```text
Delta s*(z_t,t,p,x_s)
  = nabla_z log p_T(z_t | p,x_s) - nabla_z log p_0(z_t | p)
  = nabla_z log [p_T(z_t | p,x_s) / p_0(z_t | p)].
```

This is exactly the algebraic role played by CFG: guidance is an additive score
correction, not a second image decoder. In Klein9B's velocity parameterization,
we work with the scheduler-equivalent velocity residual:

```text
Delta v*(z_t,t,p,x_s) = v_T(z_t,t,p,x_s) - v_0(z_t,t,p).
```

All branches must be evaluated at the same `(z_t,t)`. Otherwise their
difference is not a score/velocity correction and the derivation is invalid.

## Decoupled Image-CFG Teacher

The current executable teacher decomposes the target field as:

```text
Delta v_carrier = v_hard - v_normal
Delta v_goal    = highpass(v_ot - v_hard)
Delta v_T*      = Delta v_carrier + lambda * Delta v_goal
```

The current selected target uses `lambda=1`:

```text
teacher_hp_lam1 = hard_scfg + highpass(otcfg - hard_scfg)
```

Interpretation:

- `hard_scfg` is the structure-preserving carrier anchor;
- the high-pass difference supplies reference-specific texture/appearance goal;
- the late velocity output is the executable code used by the flow scheduler.

This is a teacher construction, not yet a universal theorem. Its role must be
validated against alternative teacher compositions, including full OTCFG,
low-pass-only, no carrier, and Hyperion output-derived fields.

## Online Carrier-Anchored Score Distillation

The main method applies the reference-conditioned score correction directly to
the current latent. It is not necessary to train a global student:

```text
goal_score      = score_with_reference(z_t) - score_without_reference(z_t)
goal_rendered   = Transport(goal_score, Affinity(content_state))
z_target        = stopgrad(z_t + eta * Normalize(goal_rendered))
z_next          = argmin_z L_dist(z, z_target) + L_content(z) + L_DMD(z)
```

This parameterization matches the motivation: the online update does not ask a
single unconstrained residual to rediscover spatial structure and visual intent
simultaneously. CA decides where the goal is rendered, while the native score
decides what style change is desired.

An offline student can amortize this update for repeated deployment, but it is
an acceleration variant rather than a requirement of the method.

The main ablations are:

```text
full residual:     R_theta -> Delta v_T*
carrier anchored:  R_theta -> Delta v_T* - Delta v_carrier
carrier only:      Delta v_carrier
style-ID student:  replace E_img(x_s) with style_id
adapter baseline:  g_theta(hard_scfg,t,style_id,prompt_id)
```

Only the last line should be called the adapter baseline. The paper's main
object is an image-conditioned score/velocity field.

## Field Distillation, On-Policy Aggregation, and DMD

The paper must distinguish three objectives clearly.

### Stage A: Online score refinement

At each optimized denoising state, evaluate the same-state native branches and
apply the CA-rendered score correction to the current latent. The current
`ca_only` and `score_carrier` runners implement this route without updating
global network parameters.

### Stage B: Offline field distillation (optional acceleration)

This is what the current code verifies. Teacher and student fields are measured
at identical saved states:

```text
L_field = E[lambda_mse ||r_theta-r*||_2^2
          + lambda_cos (1-cos(r_theta,r*))
          + lambda_hf  ||H(r_theta)-H(r*)||_1].
```

This stage is reproducible, stable, and gives a direct test of whether the
reference-conditioned correction field is learnable.

### Stage C: On-policy field aggregation

Offline regression sees normal/teacher probe states and can suffer
state-distribution shift when the student samples its own trajectory. The first
remedy is to collect student states, evaluate the teacher correction at the
same `(z_t,t,reference)`, append those fields to the dataset, and continue
regression. This is trajectory-aware or on-policy field distillation, not DMD.

```text
z_t ~ student trajectory
r*_t = H_grid(v_ot(z_t,t,x_s) - v_hard(z_t,t,x_s))
L_on_policy = ||r_theta(z_t,t,p,x_s) - stopgrad(r*_t)||
```

### Stage D: Proper DMD, only if required

DMD becomes justified only when a separate fake-distribution score is fitted
dynamically on generated samples and the generator is updated using the target
versus fake score difference. An offline student velocity field is not itself
the score of its generated distribution. If online CA-only already closes the
quality gap, the paper should not promote DMD or student training into the main
claim.

Recommended training order:

```text
1. online CA-only refinement as the main method;
2. add the online fake-score correction for CA+DMD;
3. optionally amortize the online field into an image-conditioned student;
4. measure the remaining distribution gap before making a formal DMD claim;
5. optional human/preference weighting for style-content trade-off.
```

## Reproducibility Contract

### Data unit

Every training sample must record:

```text
reference_image_path and immutable hash
prompt_id and exact prompt text
seed, height, width, scheduler, number of steps, guidance scale
z_t and exact timestep t
v_normal, v_hard, v_ot
hard_scfg, teacher target, prediction mode
image-encoder name/version and preprocessing
model checkpoint hash and code commit
```

### Field invariant

`v_normal`, `v_hard`, `v_ot`, and the student field are evaluated at the same
`(z_t,t)`. The main sampling path calls `scheduler.step` exactly once per
timestep. Teacher profiling does not decode or advance an extra latent.

### Split invariant

Random row splits are diagnostic only because adjacent timesteps and repeated
style IDs leak information. The paper reports:

```text
Split A: unseen prompt/content, seen reference family
Split B: unseen exact reference image
Split C: unseen style family
Split D: cross-dataset reference images
```

No reference image or near duplicate crosses train/test boundaries. Duplicate
checking should use both file hashes and a perceptual embedding threshold.

### Main conditioning invariant

The main student consumes `E_img(x_s)`. A model that only consumes `style_id`
cannot be reported as the Think-with-Image method.

### Configuration invariant

Every image figure is generated from a saved JSON config containing all branch
weights. The current frozen latent120 operating point is:

```text
paired baseline:                  inject_beta=0.50, score_student_scale=1.00
recommended/default experiment:  inject_beta=0.50, score_student_scale=1.25
conservative fallback:            inject_beta=0.45, score_student_scale=1.25
```

The recommendation is supported by a 24-pair same-seed comparison over eight
styles and three prompts: learned refinement increases by `21.03%`, with
`24/24` wins and only `-0.10%` average change in final displacement from the
normal branch. It must be revalidated after replacing style-ID conditioning
with the real reference-image code. Timestep windowing remains a diagnostic
unless it produces a measured safety gain.

### Release surface

The minimum reproducible release contains:

```text
01_dump_teacher_fields.sh
02_build_manifest.sh
03_train_field_student.sh
04_refine_score_function.sh
05_sample_paired_grid.sh
06_evaluate_metrics.sh
configs/main.yaml
configs/ablations/*.yaml
data/splits/*.json
environment lock and checkpoint hashes
```

The existing scripts already cover most of stages 1, 2, 3, and 5. The missing
paper-critical components are real image-code conditioning, layer probes,
on-policy score-function refinement, and held-out-reference evaluation.

## Three Contributions

The paper should use three contributions. Two would undersell the mechanism;
more than three would fragment the story.

### Contribution 1: Goal-Carrier-Executor view of image editing

We identify and causally verify that image-conditioned editing separates into
an early visual goal, a content-derived spatial carrier, and a late conversion
from goal-bearing image code to executable flow. This replaces the progressive
revealing view with a measurable computation model.

Claim gate: layer decodability alone is insufficient. At least one carrier swap
and one late-executor intervention must show causal separation.

### Contribution 2: Decoupled image-CFG teacher

We formulate reference-image editing as an additive conditional score/velocity
correction and build a teacher that composes a structure-preserving carrier with
a reference-specific appearance correction at the same denoising state.

Claim gate: the decoupled teacher must outperform carrier-only and monolithic
reference injection on style fidelity at matched content preservation.

### Contribution 3: Online carrier-anchored reference-image score refinement

We apply the native reference-conditioned score correction online, using the
explicit content carrier to control spatial rendering. The reference is treated
as an image instruction rather than a style ID or ordered reference layout.
Offline image-conditioned residual distillation is reported only as an optional
acceleration path.

Claim gate: the online method must consume real images, improve the
style/content Pareto frontier over the hard carrier on held-out references,
avoid reference-layout leakage, and report latency and teacher-forward cost.
DMD is an optional additional claim only after a dynamic fake-score estimator is
implemented.

## Experiments Organized by the Story

### RQ1: When are goal, carrier, and execution formed?

Experiments:

- per-layer goal decodability and reference retrieval;
- content/reference token patching at early, middle, and late blocks;
- carrier shuffle, carrier swap, and carrier restoration;
- final velocity reconstruction from intermediate states;
- analysis across denoising timestep as a second axis.

Expected figure:

```text
x-axis = MM-DiT depth
left y-axis = goal decodability
right y-axis = contribution to final Delta v
carrier intervention bands = content preservation / leakage
```

The key result is a staggered transition: goal information saturates before the
late velocity contribution, while carrier interventions control spatial
cleanliness in between.

### RQ2: Does decoupling improve the teacher field?

Compare:

```text
normal generation
hard carrier only
OTCFG only
low-pass-only and high-pass-only corrections
Hyperion teacher
decoupled teacher
```

Report style fidelity versus content fidelity as a Pareto curve, residual
spatial coherence, reference-layout leakage, and per-frequency error.

### RQ3: Can online refinement preserve image-conditioned control?

Compare:

```text
normal / hard carrier baselines
online score-only
online CA-only
online CA+DMD
optional full residual student
optional carrier-anchored image-conditioned student
teacher upper bound
```

Evaluate all methods on unseen prompts, unseen reference images, and unseen
style families. Include wall-clock latency, peak memory, number of teacher/model
forwards, and trainable parameter count.

### RQ4: What makes the method robust?

Ablate:

- image code versus style ID;
- frozen image encoder choices and token pooling;
- carrier anchor on/off;
- MSE, cosine, high-frequency, and spatial-coherence losses;
- teacher lambda, injection beta, student scale, and timestep window;
- data scale and style-family diversity;
- offline-only versus student-state field aggregation;
- field aggregation versus proper DMD, only if both exist.

### Data scale

The large real dataset should be used to make generalization, not raw experiment
count, the central evidence. Build splits at reference-image and style-family
level before generating teacher dumps. Report learning curves by number of
reference images and style families so the reader can see whether diversity or
sample count drives the gain.

### Metrics

Use a balanced metric set:

```text
style:    CSD / reference-style embedding similarity / texture coherence
content:  text-image alignment / source-layout similarity / identity retention
leakage:  reference-layout retrieval or segmentation-boundary leakage
field:    residual MSE, cosine, high-frequency error, spatial neighbor cosine
quality:  aesthetic score plus blinded human preference
system:   latency, memory, forward count, trainable parameters
```

No single style metric should select the default. Configuration selection uses
a content-constrained style score or a Pareto frontier.

## Paper Outline and Front-Back Echo

### Introduction

1. Think-with-Image systems must turn visual evidence into a controlled edit.
2. Existing methods assume that injecting reference features deeper or more
   strongly progressively reveals the target appearance.
3. Our layer analysis shows a different computation: goal first, carrier next,
   flow execution last.
4. This explains why strong attribute recognition can coexist with poor edit
   rendering and layout leakage.
5. We formulate the missing action as an image-conditioned score correction,
   construct a decoupled teacher, and distill it.
6. State the three contributions above.

### Method

```text
3.1 Goal-Carrier-Executor analysis
3.2 Image-conditioned score-ratio formulation
3.3 Decoupled image-CFG teacher
3.4 Carrier-anchored field distillation
3.5 Student-state field aggregation
3.6 Optional proper DMD refinement
```

### Experiments

Follow the same order as the motivation:

```text
4.1 Does the goal really appear early?
4.2 Does the carrier causally control clean rendering?
4.3 Do late blocks translate image code into flow?
4.4 Does the proposed teacher improve the Pareto frontier?
4.5 Can the student preserve it on unseen images?
4.6 Efficiency and scaling
```

### Conclusion

Return to the opening claim explicitly:

```text
An image edit is not progressively painted through depth. The network first
forms what should change, carries that intent through the spatial organization
of the current image, and only then compiles it into a flow update. Treating the
edit as a score correction lets us distill this process without collapsing the
reference into a category label.
```

## Figure Plan

1. **Teaser/motivation:** three depth probes showing early goal availability,
   carrier-dependent clean placement, and late flow execution.
2. **Method:** normal flow plus explicit carrier plus learned image-conditioned
   residual; all branches share `(z_t,t)`.
3. **Causal decomposition:** goal patch, carrier swap, and executor patch.
4. **Teacher Pareto frontier:** style fidelity versus content preservation.
5. **Online comparison:** score-only, CA-only, CA+DMD, and the teacher across
   unseen references; report the offline student only as an acceleration study.
6. **Generalization and efficiency:** held-out family results and speed/memory.

## Current Evidence and Honest Boundary

Already verified:

```text
latent120 direct hard-delta MLP:
val MSE          = 0.368352
hard baseline    = 0.406864
improvement      = 0.038513
student cosine   = 0.861675
hard cosine      = 0.841386
```

The direct correction also improves over the hard baseline on a held-out style
slice. In the strict 24-pair image validation, `beta=0.50, scale=1.25` improves
the learned refinement in every pair while leaving average final edit magnitude
essentially unchanged. This is now the latent120 operating point.

Not yet established:

```text
real reference-image conditioning instead of style IDs
early-goal / carrier / late-executor causal decomposition
unseen-reference and unseen-family generalization
student-state field aggregation and remaining on-policy gap
proper DMD with a dynamic fake-score estimator
teacher-to-student quality and efficiency at paper scale
```

These missing items are the next implementation priorities and should remain
visible in all progress reports. The present latent120 result validates the
score-correction direction; it is not yet the final Think-with-Image method.

## Immediate Implementation Order

1. Implement the deterministic reference-KV image code specified in
   `docs/reference_kv_score_distillation_spec.md`; retain style IDs only as
   manifest metadata and diagnostics.
2. Add layer hooks and a reproducible probe format for goal decodability,
   carrier statistics, and late velocity contribution.
3. Train the carrier-anchored student with image-code conditioning and group
   validation by exact reference image and style family.
4. Run teacher Pareto, prompt-collision, reference-noise, and wrong-reference
   gates before freezing the main conditioning design.
5. Add student-state field aggregation only after the image-conditioned field
   student is stable; measure whether a proper DMD stage is still necessary.
6. Scale teacher collection over the real style dataset and generate figures
   from immutable split/config files.
