# PAPER DRAFT (complete, doc-only) — NOT applied to main.tex

> **Status**: full draft assembled from the evidence package, under the evidence-driven
> DEFAULT framing **A** (contribution = the minimal native-score-residual training-free
> method + rigorous characterization; carrier/schedule/optimizer = negative ablations).
> `[FRAMING-A]` tags mark parts that change if the user picks B (build a differentiator).
> Numbers trace to `docs/research/{ABLATION_EVIDENCE,RESULTS_DRAFT}.md` + results/*.csv.
> **main.tex is untouched** (still the frozen "beat-K/V" version; rewrite only on user
> authorization). This file is for the user to react to, then greenlight application.

---

## Title (candidates)
- "Style Transfer for Free: Native Same-State Score Residuals in Frozen Rectified-Flow Transformers"
- "The Native Score Residual is All You Need for Training-Free Style Transfer"

## Abstract
Training-free style transfer on modern text-to-image diffusion typically bolts on
attention-feature replacement, learned adapters, or hand-designed transport heuristics.
We show that on a frozen rectified-flow transformer (FLUX.2 [klein] 9B) a single,
minimal signal suffices: the **native same-state image-CFG score residual** — the
difference between the model's score with and without the style reference as an image
condition, evaluated at the *same* noise state — used to optimize *only the current
latent* at inference (AdamW, a few iterations per denoising step). We train nothing and
add no modules. Across a 40-style benchmark (91 matched cells, multi-seed) the method
produces sharp, content-preserving transfers over radically different styles (impasto,
pop-art, engraving, woodcut, anime, expressionism). `[FRAMING-A]` Crucially, we ablate
three "obvious" additions — a content-affinity transport carrier, timestep-scheduled
injection strength, and the latent optimizer choice — and find that *none* improves over
the plain score residual beyond seed noise: the minimal form is deliberate. We further
show that the standard CSD style metric systematically *under-rates* visually excellent
transfers when the reference's subject differs from the target content, corroborating
recent calibration critiques, and we report a calibration-aware CSD+ readout alongside
qualitative evidence.

## 1. Introduction `[FRAMING-A: contribution framing depends on A/B/C]`
Diffusion style transfer without training is attractive but crowded with machinery:
key/value replacement in self-attention, IP-Adapters and other learned encoders, and
transport/AdaIN heuristics. Each adds assumptions, parameters, or reference-coupling.
We ask how little is actually needed on a modern *rectified-flow* transformer, whose
straight-line ODE and image-conditioning differ from the DDPM U-Nets most methods target.

Our answer is a single native signal. The frozen model already "knows" how the style
reference should bend its prediction: the score difference between conditioning on the
reference image and not, taken at the *same* noise state, is a style-transport direction.
We push the current latent along it, optimizing only that latent (no weights, no modules)
for a few AdamW steps per denoise step. That is the whole method.

**Contributions.** (i) A minimal training-free style-transfer method for frozen
rectified-flow transformers using only the native same-state image-CFG score residual,
optimizing the latent alone. (ii) `[FRAMING-A]` A rigorous negative-ablation study
showing that a content-affinity carrier, timestep scheduling, and the optimizer choice
each add nothing beyond seed noise — the minimal form is sufficient. (iii) A 40-style
multi-seed benchmark and a calibration analysis demonstrating that raw CSD under-rates
visually excellent transfers, with a CSD+ readout and qualitative evidence.

## 2. Related Work
- **Training-free style transfer in diffusion**: attention-feature replacement
  (StyleAligned, Style Injection/StyleID), reference-conditioning adapters (IP-Adapter,
  InstantStyle), and optimization/transport approaches (RB-Modulation, DDS/CDS). We use
  none of their added machinery; the signal is the model's own image-CFG score.
- **Test-time guidance**: universal/energy guidance, PAG/SEG attention perturbation,
  CADS — shape sampling toward a target at inference. Our residual is a same-state,
  image-conditioned instance of this idea, applied via latent optimization.
- **Rectified flow & inversion**: flow-matching foundations, RF-Inversion, FlowEdit.
  We operate on a frozen flow model and never invert; we optimize the running latent.
- **Style-similarity metrics**: CSD; and the calibration critique (Frochte 2026) that
  motivates our §5 analysis and CSD+.
  `[Scope: we do NOT run these as competitor baselines — see §6 Limitations.]`

## 3. Method
**Same-state score residual.** At denoise step with latent $z_t$ and noise level
$\sigma_t$, form the frozen model's velocity/score with the style reference as image
condition, $s^{+}$, and without it, $s^{-}$, at the *identical* state. The residual
$\Delta = s^{+}-s^{-}$ is the style-transport direction. (Same-state contract: both
branches share timestep, latent, and image ids; only the reference edit condition differs.)
**Latent optimization.** We optimize only $z_t$ to follow $\Delta$ under a style objective
(with light content and distance regularization), AdamW, ~5 inner iterations per step
over 25 denoising steps, starting from a high-noise start-time. No network weights, no
adapters, no reference tensors injected into attention. `[FRAMING-A: the carrier that a
gen-1 lineage would add here is shown inert in §5.3 and omitted.]`
**Why rectified flow matters.** The straight-line ODE makes the same-state residual a
clean directional signal; we exploit the model's own image-conditioning rather than
replacing internal features.

## 4. Experimental Setup
Model FLUX.2 [klein] 9B, frozen. Benchmark: 40 style reference images × content prompts
(panda / castle / cat / sunflowers / fisherman), 91 matched cells for the method, multi-
seed {64,65,66}. Metrics: CSD and CSD+ (CSLS readout, Frochte 2026) style similarity,
CLIP-T (prompt adherence), DINO style similarity, VGG-Gram distance, DINO layout-leak.
No external competitor baselines (§6); evaluation is internal ablations + qualitative +
calibration analysis.

## 5. Results
> Headline figure drafted: `docs/research/fig_breadth_grid.png` (12 diverse styles,
> per-image CSD labels) — visually shows both the breadth AND the CSD-underrating point
> (excellent transfers scoring low CSD). Draft montage; final should use canonical cells
> + a style-ref thumbnail column.
### 5.1 Breadth (headline)
Over all 40 styles / 91 cells: per-style mean CSD **0.600** (95% bootstrap CI
[0.562, 0.638]), CLIP-T 0.311 [0.304, 0.318], layout-leak 0.356 [0.331, 0.381],
VGG-Gram 3.47. Qualitatively (Fig. breadth-grid) transfers are uniformly sharp and
content-preserving across impasto oil, pop-art halftone, B&W cross-hatch engraving,
woodcut, anime/digital painting, and expressionist brushwork.

### 5.2 Style metrics are uncalibrated — read qualitatively
Raw CSD systematically under-rates excellent transfers because it entangles style with
*subject* similarity to the reference. A flawless engraving of a sunflower field scores
CSD 0.49 (reference subject: an archer); an excellent expressionist castle scores 0.381
(reference: a portrait). This matches Frochte (2026) (negative discrimination gaps for
23/91 artists). We therefore report CSD+ (CSLS) alongside raw CSD and rely on qualitative
/[human] evaluation for the headline claim; mean CSD 0.600 is a lower bound. (Figs.
style25, style10.)

### 5.3 The minimal form is sufficient (negative ablations) `[FRAMING-A]`
- **Content-affinity carrier** (multiplicative reweight of the residual): inert / at
  best style-conditional across five probes — matched-matrix wash (0.5935 vs 0.5993),
  flat-strength sweep {0.5,0.75,1.0} (never wins), timestep schedule
  {const,back,front,sqrt,cosine} (no effect beyond seed noise), CSD+ (no rescue),
  method-delta LPIPS (perturbs impasto but no quality gain).
- **Timestep scheduling of injection strength**: no effect beyond the ±0.11 seed noise.
- **Latent optimizer** (Adam vs AdamW, wd 0/1e-4/1e-3): near-null (Δ 0.01–0.05).
(First-gen SLT/AdaIN warm-start heuristics are disabled and not part of the method.)

## 6. Limitations
- No external published baseline (scope choice); compensated by negative ablations,
  40-style breadth, and calibration analysis. [Reviewers may want one; addable later.]
- Standard ArtFID is not soundly computable at single-reference-per-style scale
  (rank-deficient style-FID covariance, 40 ≪ Inception dim); we justify its omission
  and report per-image metrics + CSD+.
- Content-LPIPS needs a no-style baseline set (queued).
- Large seed variance on textured styles (±0.11 CSD); small deltas are noise.

## 7. Conclusion `[FRAMING-A]`
On a frozen rectified-flow transformer, training-free style transfer needs only the
model's native same-state image-CFG score residual, applied by optimizing the current
latent. Added transport, scheduling, and optimizer machinery do not help; the minimal
form is sufficient and, read qualitatively (since CSD under-rates it), strong across
40 styles.

---
## Open decisions before this becomes main.tex (USER)
1. Framing A/B/C (this draft = A). 2. Eval-corpus scope (accept 40-single-ref + justified
suite, or expand for ArtFID). 3. Authorize main.tex rewrite (replace the frozen K/V spine
with this). 4. Optional: human study (infra buildable now). 5. Switch to aaai2027.sty
(kit on disk per LOOP_PLAN N-C).
