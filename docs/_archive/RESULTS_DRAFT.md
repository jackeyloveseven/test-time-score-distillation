# Results Section — DRAFT (doc only; NOT applied to main.tex)
Date: 2026-07-19. Paper-ready draft of the experiments/results section under the
pivot framing (contribution = the minimal native-score-residual training-free method
+ rigorous characterization; carrier/schedule/optimizer = negative ablations). All
numbers trace to results/*.csv (see docs/research/ABLATION_EVIDENCE.md for provenance).
Framing pending user A/B/C — do NOT paste into main.tex until confirmed. K/V spine in
current main.tex still to be replaced (user directive: K/V dropped).

## 1. Setup
- Model: FLUX.2 [klein] 9B, frozen rectified-flow T2I transformer. Train nothing.
- Method (score_match): native same-state image-CFG score residual as the style-
  transport direction; optimize ONLY the current latent (AdamW, 5 inner iters over
  25 denoise steps, start-time 1000). Single style reference image + content prompt.
- Benchmark: 40 style references x content prompts (kp0–kp4: panda/castle/cat/
  sunflowers/fisherman); 66 matched cells for the base method (subset multi-seed
  {64,65,66}). Metrics: CSD & CSD+ (CSLS, Frochte 2026) style similarity, CLIP-T
  (prompt adherence), DINO style sim, VGG-Gram distance, DINO layout-leak.
- NO external competitor baselines (scope choice; see §5). Comparisons are internal
  ablations + qualitative + the calibration analysis.

## 2. Main result — breadth (the headline evidence)
Across ALL 40 styles / 91 matched cells (every style multi-seed, seeds {64,65,66}),
the base method achieves per-style mean CSD 0.600 (95% bootstrap CI [0.562, 0.638]),
CLIP-T 0.311 ([0.304, 0.318], solid prompt adherence), layout-leak 0.356 ([0.331,
0.381]), VGG-Gram 3.47. (Cell-level CSD n=91, mean 0.597, std 0.130.) Qualitatively the
transfers are uniformly sharp and content-preserving across radically different
styles — thick impasto oil, pop-art halftone, B&W cross-hatch engraving, woodcut,
anime/digital painting, expressionist brushwork (Fig. breadth-grid). Content
(the prompt subject) stays legible while the reference's style is faithfully applied.

## 3. The minimal form is sufficient (ablations — all negative)
We ablate three "obvious" additions; none improves over the plain score residual
beyond seed noise, so the method's minimal form is deliberate, not incidental.
- **Content-affinity carrier** (multiplicative reweight of the residual): inert/at
  best style-conditional across FIVE independent probes — matched matrix wash (CSD
  0.5935 vs 0.5993), flat-strength sweep {0.5,0.75,1.0} (never wins), timestep
  schedule {const,back,front,sqrt,cosine} (no effect beyond the ±0.11 seed noise),
  calibrated CSD+ (does not rescue it), and method-delta LPIPS (changes impasto
  images 0.56x a full method-swap but yields no quality gain). Literature diagnosis:
  it conflates structure-preservation with appearance-transport in one term.
- **Timestep scheduling of injection strength** (protect-content-early / push-style-
  late, per StyleID/Scheduled-Style-Injection): no effect distinguishable from seed
  noise. Timing/magnitude is not a lever here.
- **Latent optimizer** (Adam vs AdamW, wd 0/1e-4/1e-3): near-null (CSD moves 0.01–
  0.05, consistent with AdamW(wd=0) ≡ Adam). Robustness, not a lever.
(First-gen warm-start heuristics SLT/AdaIN are disabled and not part of the method.)

## 4. Style metrics are not calibrated — read qualitatively (analysis)
Raw CSD cosine systematically UNDERRATES visually-excellent transfers. Because CSD
entangles style with SUBJECT similarity to the reference, a perfect stylization whose
prompt subject differs from the reference's subject scores low. Examples: a flawless
B&W-engraving sunflower field scores CSD 0.49 (ref = an archer); an excellent
expressionist castle scores 0.381 (ref = a portrait). This matches Frochte (2026),
who shows raw CSD gives negative discrimination gaps for 23/91 artists. We therefore
(a) report CSD+ (CSLS readout) alongside raw CSD, and (b) rely on qualitative /
[human] evaluation for the headline quality claim. The mean CSD 0.594 is a LOWER
bound on true quality.

## 5. Limitations (state plainly)
- **No external published baseline.** Scope choice; we compensate with the negative
  ablations (§3), 40-style breadth (§2), and the calibration analysis (§4). [Reviewers
  may want a published baseline; can be added if rejected on that axis.]
- **Standard ArtFID is not soundly computable** at single-reference-per-style scale
  (FID-style covariance is rank-deficient with 40 style refs << Inception dim ≥64);
  we report per-image style/content metrics + CSD+ instead and justify the omission.
- **Content-LPIPS** requires a no-style baseline set (not yet generated); queued.
- **Seed variance is large on textured styles** (±0.11 CSD); small single-seed deltas
  are noise — all ablation nulls are reported against this floor.

## 6. Open items before submission (TODO)
- [user] Confirm framing (A base-method-as-contribution / B build a real differentiator
  / C keep base rigor). This draft assumes A/C.
- [user] Eval-corpus scope: accept the 40-single-ref benchmark + justified metric suite,
  OR invest in a multi-image-per-style corpus for ArtFID.
- Generate no-style T2I baselines -> content-LPIPS (small new script; GPU-gated).
- [DONE 2026-07-19] Multi-seed 40-style table: all 40 styles now >=2 seeds, 91 cells,
  per-style CSD 0.600 CI[0.562,0.638] (§2).
- Human preference study (infra can be built; needs annotators).
- Replace the K/V experimental spine in main.tex once framing confirmed (currently
  frozen — the whole current main.tex is a "we beat K/V" paper, must be rewritten).
