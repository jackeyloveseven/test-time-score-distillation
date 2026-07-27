# Historical Literature Survey: Rectified-Flow Mechanisms and Evaluation

> Status update (2026-07-23): research note only. It predates the completed
> 40-style benchmark, Agent audits, AAAI 2027 layout, and current submission
> package. Use `AAAI/main.tex` and `docs/PROJECT_STATE.md` for current status.

Status: Phase-1 RESEARCH only. Read-only literature survey, no code/GPU touched.
Scope: (A) rectified-flow/FLUX-specific editing/inversion mechanisms to borrow
for the latent-only score-residual push; (B) the training-free evaluation bar
at recent CVPR/ICCV/NeurIPS/ICLR/AAAI venues, adapted to our constraint of
**no external competitor baselines**.

Cross-check performed: grepped `AAAI/main.tex` + `AAAI/references.bib` to see
what is already cited, so this report only flags genuinely new material.
Already cited: Delta Denoising Score (`hertz2023dds`), Contrastive Denoising
Score (`nam2024cds`), StyleID (`chung2024styleinjection`), StyleAligned
(`hertz2024stylealigned`), InstantStyle (`wang2024instantstyle`), rectified
flow / flow matching foundations (`lipman2023flow`, `liu2023rectified`,
`esser2024sd3`), DMD/DMD2, and the CSD-is-not-calibrated critique
(`frochte2026csd` = Frochte 2026, arXiv 2605.09030, "When Style Similarity
Scores Fail: Diagnosing Raw CSD Cosine in Artist-Style Evaluation" — matches
what this survey independently found). **Not yet cited**: RF-Inversion,
FlowEdit, RF-Solver, FireFlow, Stable Flow, Scheduled Style Injection, Dual
Rectified Flows style transfer, V-Shuffle — these are the new candidates
below.

---

## Part A — Rectified-flow mechanisms to borrow

### A1. RF-Inversion's controlled-ODE blend (η/γ schedule)
**Source**: Rout et al., "Semantic Image Inversion and Editing using Rectified
Stochastic Differential Equations," ICLR 2025 (arXiv 2410.10792).

**Mechanism**: derives, via a linear-quadratic-regulator (LQR) optimal-control
argument, a *provably optimal* blend between the unconditional native
velocity `u_t(Y_t)` and a goal-directed "controller" velocity
`u_t(Y_t|y_1)`:
`dY_t = [u_t(Y_t) + γ(u_t(Y_t|y_1) − u_t(Y_t))] dt`. On the reverse
(editing) pass the same structure appears with a second knob `η`, and both
knobs can be **time-varying**: `η_t = η` for `t ≤ τ`, then `0` — i.e. lock
faithfulness early, release it late. γ=1 has a proven SDE-equivalence
guarantee (output noise matches the true prior exactly); γ=0 recovers the
original image exactly.

**How to apply**: our current carrier-strength / step-count hyperparameters
are tuned ad hoc. Reframe the AdamW latent push as an explicit
controlled-velocity blend with a τ-scheduled knob (strong content lock in
the early ~2 of 5 iterations, release toward the CFG residual in the late
iterations) instead of (or as a principled prior on) a flat learning rate.
This gives a *theoretically motivated* one-parameter Pareto knob rather than
an empirically-fit one, which directly serves the paper's already-planned
P1 Pareto-frontier sweep (`docs/…SUBMISSION_READINESS.md` P1 item 1) with a
citable mechanism, not just a grid search.

**Risk**: RF-Inversion's SDE-equivalence guarantee is proven for full
noise→image inversion+generation; we start from a real content latent with
no formal inversion step, so the guarantee does not transfer as-is — only
the *qualitative* schedule idea (early-lock/late-release) is safely
transferable, not the theorem. Adds one more hyperparameter (τ) that itself
needs an ablation to avoid looking like unexplained tuning.

### A2. FlowEdit's inversion-free source→target velocity difference + averaging
**Source**: Kulikov et al., "FlowEdit: Inversion-Free Text-Based Editing
Using Pre-Trained Flow Models" (arXiv 2412.08629; SD3 + FLUX).

**Mechanism**: never inverts. At each step, samples a noisy source point
`Ẑ_t^src = (1−t)Z_0^src + tN_t`, forms the matched target point, and computes
`V_Δ = V^tar(Ẑ_t^tar,t) − V^src(Ẑ_t^src,t)`, **averaged over `n_avg` random
noise draws** to approximate an expectation before applying the ODE step.
Uses *asymmetric* CFG scales for source vs. target branches (FLUX: 1.5 vs.
5.5). Reports this beats inversion-based editing on MSE (1376 vs 2239) and
LPIPS (0.15 vs 0.25) at matched structure-preservation goals.

**How to apply**: structurally this is close to our "native image-CFG
residual at the same state" signal, but FlowEdit's `n_avg` noise-averaging
is a cheap, literature-tested variance-reduction trick we may not be doing —
average the residual over a few random noise draws per iteration before
handing it to AdamW, which should reduce gradient noise without changing
the mechanism. Also worth ablating **asymmetric** guidance scales for the
with-reference vs. without-reference branches (rather than symmetric CFG),
since FlowEdit found meaningfully different optimal scales per branch.

**Risk**: `n_avg > 1` multiplies native forward-pass cost per iteration —
must be weighed against the existing runtime/NFE table already required by
P0 gate item 3. Also FlowEdit's asymmetric-scale numbers were tuned for
*text*-prompt source/target pairs, not image-style conditioning — must be
retuned, not copied.

### A3. RF-Solver / FireFlow: free second-order ODE accuracy via velocity reuse
**Sources**: "Taming Rectified Flow for Inversion and Editing" (RF-Solver;
arXiv 2411.04746) and "FireFlow: Fast Inversion of Rectified Flow for Image
Semantic Editing" (arXiv 2412.07517). Two independent groups converge on the
same trick, which is reassuring evidence it is safe/well-established.

**Mechanism**: plain Euler stepping of the flow ODE has O(h²) local error;
a midpoint/second-order solver fixes this but naively doubles NFEs. Both
papers show the *previous* step's midpoint velocity can be reused for the
*current* step's half-point, giving second-order accuracy (O(h³) error) at
**first-order (Euler-equivalent) compute cost**. RF-Solver reports LPIPS
0.149 / CLIP 33.66 on FLUX editing, beating RF-Inversion; FireFlow reports
18 NFEs vs. 56 for vanilla, ~3× wall-clock speedup.

**How to apply**: our pipeline runs ~5 iters × 25 steps of flow integration
while optimizing the latent. Swapping the underlying integrator for this
free second-order scheme is a nearly zero-cost, drop-in change (no new
forward passes) that reduces the numerical noise floor the AdamW loop is
correcting against. Cheap ablation: same iters/steps, solver order 1 vs. 2,
measure CSD/ArtFID/Gram delta — if error was a real bottleneck this should
show up as either fewer iterations needed for the same quality, or higher
quality at fixed iterations.

**Risk**: needs a first-derivative (velocity-rate) finite-difference
estimate, adding implementation complexity; interacts non-obviously with
the *outer* AdamW correction loop (the outer optimizer may already be
absorbing whatever error a first-order integrator introduces, making the
gain marginal) — should be profiled before committing engineering time.

### A4. Dual Rectified Flows: velocity subtraction as an alternate style-transport signal
**Source**: "Inversion-Free Style Transfer with Dual Rectified Flows" (arXiv
2511.20986; SD3 ReFlow backbone). Venue not verifiable from search (appears
arXiv-only preprint as of this survey — flag as unverified for peer-review
status).

**Mechanism**: from a *shared* noise seed, runs parallel forward
trajectories from the content image and from the style image, and defines
the style-transport direction as **velocity subtraction**
`v_stylized = v_θ(x_t^style) − v_θ(x_t^content)` (and the mirrored
direction), then fuses both via a midpoint interpolation of the two
resulting stylized estimates. Reports ArtFID 27.792 vs. 29.5–35.0 for
StyleSSP/Zstar, and 5.5s/image inference (vs. 1445s for InST). Ablates
trajectory variants (single-direction vs. midpoint-fused) and shows removing
attention injection degrades texture fidelity.

**How to apply**: this is a genuinely different, and *simpler*, mechanism
for producing a "style direction" than a same-state image-CFG residual —
pure velocity subtraction between independently-run content/style forward
passes, no CFG, no reference/without-reference contrast. Given that this
project's memory already flags "the carrier differentiator is empirically
inert," this is a good **causal control/ablation candidate**: swap the
transport signal for this velocity-subtraction baseline (implementable
in-house, no external repo needed) and show our score-residual signal beats
it, or honestly report if it does not. Either outcome is informative,
non-baseline evidence.

**Risk**: requires an extra full forward pass for the style-image trajectory
at every timestep (compute cost); the paper's own reported numbers are
against trained/attention-injection competitors on SD3, not on FLUX.2
[klein], so nothing here transfers as a validated *number* — only as a
mechanism to reimplement and test in-house.

### A5. Delta/Contrastive Denoising Score (already cited) — sharpen the differentiation
**Sources**: Hertz et al., "Delta Denoising Score," ICCV 2023 (already cited
as `hertz2023dds`); Nam et al., "Contrastive Denoising Score for Text-Guided
Latent Diffusion Image Editing," CVPR 2024 (already cited as `nam2024cds`).

This is the closest academic ancestor of the whole method family: DDS is a
same-latent, paired conditional-minus-unconditional score difference used
directly as an editing gradient (originally text-CFG, diffusion score, SDS
distillation-style). CDS's contribution — adding a **structure-preservation
regularizer** (a CUT/contrastive loss on self-attention features) on top of
the raw score difference, because "the difference between scoring functions
alone is insufficient to preserve structure" — is a documented,
peer-reviewed fix for exactly the style-vs-layout trade-off this project's
P1 gate is trying to characterize.

**How to apply**: since the paper already differentiates itself from DDS/CDS
algebraically (main.tex lines 176–186: image-CFG not text-CFG, transported
by a content-only carrier rather than distilled), the concrete borrow is
CDS's **auxiliary structural loss** as a second regularizer alongside the
carrier — an additional lever to move the style–layout Pareto frontier that
is independently literature-validated (not just carrier tuning). Needs
adaptation from U-Net cross-attention (CDS's setting) to FLUX's DiT joint
self-attention blocks.

**Risk**: adds another loss term and balancing hyperparameter; if it
supersedes the carrier's effect, that would need to be reported honestly
given the carrier's contribution is already fragile.

---

## Part B — AAAI evidence checklist without external baselines

Surveyed papers for "what evidence carries a training-free claim": StyleID
(CVPR 2024 Highlight), Scheduled Style Injection (arXiv 2605.26538, Pareto
frontier framing), Stable Flow (arXiv 2411.14430, vital-layer causal
ablation), Dual Rectified Flows style transfer (arXiv 2511.20986), FlowEdit
(arXiv 2412.08629), RF-Inversion (ICLR 2025). All of these papers *do*
compare against external baselines — I found **no precedent for a
training-free style/editing paper skipping competitor comparison entirely**
at these venues; FlowEdit is the closest (explicitly scopes comparison to
"recently available flow-model methods" rather than exhaustive prior art,
but still compares). This means our "no external baselines" constraint is
unusual and the paper must compensate with *depth* in what it does report.
Below, ✅ = already have per `AAAI/SUBMISSION_READINESS.md` and `main.tex`;
⬜ = still needed.

**Metrics** (bar set by StyleID/Dual-RF/Scheduled-Style-Injection):
- ✅ CSD (`somepalli2024csd`), CLIP-T, DINO, VGG Gram, a leakage metric —
  already reported, already flagged with the CSD-calibration caveat
  (`frochte2026csd`, matches this survey's independent find).
- ⬜ ArtFID = (1+LPIPS)·(1+FID) — the single most common "jointly evaluates
  stylization + content preservation" composite metric across every paper
  surveyed (StyleID, Dual-RF, Scheduled Style Injection); not currently in
  `main.tex`'s metric list. Cheap to add (needs only our own generations +
  the CSD/ArtFID reference toolchain, no competitor code).
- ⬜ Formal LPIPS as its own reported number (currently folded implicitly
  into "leakage"; LPIPS is reported standalone in every surveyed paper).

**Ablations** (bar set by StyleID's γ/temperature ablation, Scheduled Style
Injection's schedule ablation, Dual-RF's trajectory-variant ablation):
- ⬜ P0 item 1 (full matched seed×prompt×reference matrix for kv_match /
  score_match / score_carrier, with mean/SD/CI/win-rate) is the **single
  highest-leverage item** — every surveyed paper's headline table is exactly
  this shape (a matched matrix with variance), just against external methods
  instead of our own variants. In our no-baseline setting, this matrix *is*
  the paper's "baseline comparison," so it must be at full strength, not the
  current 3-pair mechanism study.
- ⬜ P1 item 1 (carrier-strength/temperature Pareto sweep) — directly
  matches Scheduled Style Injection's core contribution shape (a
  style-fidelity vs. content-preservation curve, not a point estimate).
  See A1 above for a citable mechanism (RF-Inversion's η/τ schedule) to
  frame this sweep as principled rather than a raw grid search.
- ⬜ P1 item 2 (identity / shuffled-token / uniform-affinity /
  reference-affinity carrier interventions) — this is the **direct
  substitute for "why it works" that a baseline comparison would otherwise
  provide**. Stable Flow's vital-layer protocol (perturb component, measure
  output distance via a fixed perceptual metric, define a scalar
  "importance") is a ready-made template: define an analogous "carrier
  vitality" metric, perturb each carrier variant, and report the resulting
  metric deltas. Also see A4 (velocity-subtraction as an alternate transport
  signal) as one more concrete intervention to add to this set.
- ⬜ P0 item 3 (runtime/memory/NFE vs. K/V replacement and unmodified native
  sampler) — this table needs no external baseline code (both comparators
  are internal), so there is no excuse to leave it out; every training-free
  paper surveyed reports at least a runtime/steps table (FireFlow, RF-Solver
  lead with exactly this kind of table).

**Human evaluation** (bar set by Stable Flow's MTurk 2AFC, and general
practice across surveyed style-transfer papers):
- ⬜ P0 item 2 (blinded human preference). Concrete, precedented protocol:
  pairwise 2-alternative forced choice (not multi-way — psychophysics
  literature and practice across surveyed papers favor 2AFC for reliability)
  between our own method variants (kv_match vs. score_match vs.
  score_carrier), ≥5 raters per comparison with majority vote (a pattern
  seen in comparable style-transfer human studies), report raw win rates
  **and** an inter-rater agreement statistic. Since we have no external
  competitor to compare against, the 2AFC should be run *between our own
  ablation arms*, which is exactly the same evidentiary role a
  baseline-comparison human study would otherwise play.

**Breadth / coverage**:
- ⬜ P0 item 4 (broader, license-documented reference set). Every surveyed
  competitive paper evaluates on tens–hundreds of style/content pairs; our
  three-pair result is explicitly flagged in `SUBMISSION_READINESS.md` as
  "a mechanism study, not a benchmark" — this is the most direct fix for
  that framing gap and does not require any external code.

**What we already have that meaningfully counts**: qualitative 3-pair grid,
partial matched-metric comparison, CA-only 9-run diagnostic, an explicit
claim–evidence audit document, and — notably — the CSD-calibration caveat is
already ahead of the curve (independently found by this survey as a live,
recent 2026 critique paper, already cited).

---

## Ranked TRY FIRST (top 3)

1. **Full matched multi-seed matrix (P0 item 1), reported with mean/SD/CI/
   win-rate.** In a no-external-baseline paper this table *is* the paper's
   comparison section. Nothing else below is credible until this is solid —
   it is the cheapest, highest-leverage, and lowest-risk item (uses existing
   infra, no new mechanism).
2. **Carrier-strength Pareto sweep (P1 item 1), framed via RF-Inversion's
   η/τ schedule (A1) as a principled rather than raw-grid knob.** Matches
   exactly the evidentiary shape (a style/content Pareto curve) that
   Scheduled Style Injection uses as its entire contribution — cheap to
   generate from the existing sweep infrastructure, and gives a citable
   mechanism instead of an unexplained hyperparameter grid.
3. **Causal carrier interventions (P1 item 2) using Stable Flow's
   perturb-and-measure template, including the velocity-subtraction control
   from Dual Rectified Flows (A4) as one concrete alternate-mechanism arm.**
   This is the direct fix for the paper's current weakest point ("carrier
   differentiator is empirically inert") and is the single strongest
   non-baseline "why it works" evidence available — it either rescues the
   carrier's contribution with a rigorous causal story, or honestly narrows
   the claim, both of which are defensible outcomes for a reviewer.

Bonus / lower priority: A2's noise-averaging (`n_avg`) as a cheap
variance-reduction tweak to the residual signal; A3's free second-order
integrator swap as a compute/quality profiling check; A5's CDS-style
structural regularizer if the Pareto sweep in (2) shows the carrier alone
cannot resolve the style–layout trade-off.
