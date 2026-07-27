# Phase-1 Research Synthesis → Staged Improvement Plan
Date: 2026-07-19. Inputs: `lit_style_transfer.md`, `lit_guidance.md`, `lit_rf_and_eval.md`
(three independent lit-survey sub-agents). Goal (user): 借鉴顶会文献改进方法 → AAAI 水平.
NO external competitor baselines (user directive); improve OUR training-free,
latent-only, native-score-residual method on frozen rectified-flow FLUX.2 [klein].

## Starting diagnosis (corrected 2026-07-19)
The "carrier is dead" verdict was a CSD artifact. Truly-matched eyeball + per-cell
metrics show the carrier is **style-conditional and high-variance**: on impasto/
palette-knife style27 it visibly helps (stronger texture + cleaner subject; CSD &
layout-leak win on 2/3 seeds) but loses seed65; on flat graphic styles (pop-art 00,
woodcut 13) it is a near-exact no-op. So the current carrier is a *weak, unstable*
mechanism, not a wrong idea. The literature explains WHY and how to fix it.

## Why the current carrier is weak (literature diagnosis)
1. **Fixed strength across all 25 steps.** Identical to StyleID's single global blend
   scalar γ — *Scheduled Style Injection* (CVPR-W/NTIRE 2026, arXiv 2605.26538)
   proves a fixed scalar is Pareto-suboptimal; a schedule (protect content early,
   push style late) beats the best fixed γ by ~6% ArtFID *for free*. Our
   carrier-strength and residual weight are both flat scalars → same flaw.
2. **Structure-preservation and appearance-transport forced onto ONE signal.** Our
   carrier does `Transport(goal, Affinity(z_t))` — a *multiplicative* reweighting of
   an appearance residual by a content-affinity kernel. The literature (PnP Diffusion
   Features CVPR 2023; Splicing ViT CVPR 2022) *always* uses self-similarity as an
   *independent additive structure loss*, never as a multiplicative kernel on an
   unrelated signal. We conflated two jobs into one term → neither is done well.

## Convergent signal (what ≥2 independent agents both chose)
- **Timestep scheduling of injection strength** — picked by style-transfer #1,
  guidance #2, and RF #1 (RF-Inversion lock-early/release-late). Cheapest possible
  (zero new forward pass, zero new loss), directly targets flaw #1, and DIAGNOSES
  whether "inert at any strength" was a magnitude/scheduling problem (fixable) vs a
  direction problem (not). → **Stage 1, do first.**

## Staged plan (cheapest → strongest; each stage decides the next)
### Stage 1 — Timestep-scheduled strength  [cheapest, do first]
Re-parameterize the flat `carrier_strength` AND the style-residual weight into a
schedule over the existing `progress` variable (runner already computes
`progress = 1 - t/start_time`, 0 early → 1 late; lines 441/881). Shapes to sweep:
content-early/style-late cosine and √ ramps (per Scheduled Style Injection), plus
the reverse as a control. No new forward pass. On the matched matrix cells +
textured styles. **Decision gate:** if a schedule turns the carrier from wash into a
consistent win (esp. on textured styles, low layout-leak), the differentiator is
alive and cheap → that becomes the paper. If schedules do nothing, the carrier's
problem is *direction* not magnitude → go to Stage 2.

### Stage 2 — Decouple structure from transport  [reuses affinity code]
Stop multiplicatively reweighting. Instead add an *independent additive* content
self-similarity structure loss (à la PnP/Splicing) to the AdamW objective, alongside
the raw native score residual for appearance. This separates "hold content" from
"push style" into two clean gradients. Cheap (reuses existing affinity computation).

### Stage 3 — Attention-perturbation counter-term  [strongest, 2× on one branch]
If Stage 1–2 insufficient: add a PAG/SEG/**SoftPAG-HeadHunter** (NeurIPS 2025,
validated on FLUX.1 for style-specific guidance) counter-term — build a
structure-degraded "bad" prediction of the frozen model and guide away from it,
CFG-style, decaying over steps. Most direct functional replacement for what the
carrier was meant to do (hold content while pushing style); architecture-matched to
FLUX. Costs one extra forward pass on that branch. Also consider TAG (arXiv
2510.04533) tangential/radial decomposition as a zero-forward manifold booster on
the residual (risk: not a style/content disentangler on its own).

### Orthogonal booster (any stage) — RB-Modulation Tweedie CSD loss
Add a genuinely independent gradient: CSD style-sim computed at the Tweedie/x0
estimate, backpropped into the same AdamW loop (RB-Modulation ICLR 2025). Orthogonal
to the residual; moderate cost (extra CSD forward per inner iter).

## Evidence gaps to close for AAAI (from the eval report)
- **No training-free paper skips competitor comparison entirely** → our no-baseline
  constraint is unusual and RAISES the internal-depth bar. Compensate with: causal
  mechanism ablations (Stable Flow vital-layer perturb-and-measure template),
  breadth, and human eval.
- **Missing metrics:** ArtFID + standalone LPIPS (we have CSD/CLIP-T/DINO/Gram/Leak).
- **Verify** the CSD-calibration citation already in references.bib
  ("Frochte 2026, arXiv 2605.09030") is real, not a placeholder.

## Unverified claims flagged by the agents (verify before building)
- Dual-RF (arXiv 2511.20986) & Scheduled Style Injection venue/status = arXiv-only,
  unconfirmed peer review. Attention-Distillation (arXiv 2502.20235) exact loss not
  verified past project page. HeadHunter/SoftPAG & CFG-OEC exact formulas not
  extracted past abstract — read source before implementing.

## Immediate next action
Implement Stage 1 (schedule option + sweep driver), syntax-check now (GPU is
saturated by external jobs), fire a 1-cell smoke run the moment GPU frees, then the
matched sweep. Eyeball before trusting any metric.
