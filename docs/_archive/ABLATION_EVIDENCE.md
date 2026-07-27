# Ablation Evidence Ledger (framing-agnostic; facts + provenance)
Date: 2026-07-19. Source-of-truth for the paper's ablation/results section. This is
an EVIDENCE LEDGER, not paper prose — every row cites the artifact it came from so
the results section can be written (under any framing) without re-deriving. Numbers
are authoritative here; cross-check `docs/LOOP_PLAN.md` loop log for context.

## Method under test
Native same-state image-CFG **score residual** ("goal"): run the frozen FLUX.2
[klein] 9B rectified-flow model with vs without the style reference as an image
condition at the SAME noise state; the score difference is the style-transport
direction. Optimize ONLY the current latent (AdamW, ~5 inner iters over 25 steps).
Train nothing. The "score_match" mode = this base method, no carrier.

## What WORKS (keep — the contribution candidate)
- **Base method (score_match) produces strong, sharp, content-preserving transfer.**
  Eyeball: style27 impasto panda, style02 anime panda, style07 ink-wash castle,
  style13 woodcut sunflowers, style00 pop-art castle — all sharp, style strongly
  applied, prompt subject legible. Cleared the user's explicit bar ("五个style优秀
  就够了"). Content-preservation is NOT a deficiency (advisor discriminating check).
- Quant (41 score_match cells now, matrix+breadth+breadthext, 15 styles): CSD 0.574
  mean, CLIP-T 0.313 mean (solid prompt adherence), layout_leak 0.354 mean. Artifacts:
  results/matrix_metrics.csv, results/breadth_metrics.csv, results/breadthext_metrics
  .csv, results/breadth_aggregate.json.
- **FULL 40-STYLE BENCHMARK (2026-07-19, bench40)**: base method now covers ALL 40
  available style refs (66 score_match cells). Mean CSD 0.594 (std 0.119), CLIP-T
  0.312, layout_leak 0.352. Eyeball across styles = uniformly strong-to-excellent
  (styles 10/18/21/25 + earlier 00/02/07/13/27 all sharp, content-preserved, style
  faithfully applied). This is the paper's breadth evidence. Artifacts: results/
  bench40_metrics.csv + matrix/breadth/breadthext.
- **CSD-UNDERRATING is systematic (key paper finding + Frochte-2026 support)**: the
  low-CSD tail (<0.45) is NOT method failure — every case eyeballed is visually
  good/excellent. Mechanism: raw CSD entangles style + SUBJECT similarity to the ref,
  so when the style ref's subject differs from the prompt subject, CSD penalizes a
  perfect style transfer. Two clean figure-ready examples:
  * style25 (B&W cross-hatch engraving ref, archer) x sunflowers-at-sunset: flawless
    engraving transfer, CSD only 0.49/0.40.
  * style10 (expressionist thick-impasto portrait ref) x castle-by-sea: excellent
    bold-brushstroke castle, CSD only 0.381.
  => CSD mean 0.594 UNDERSTATES true visual quality. Directly motivates CSD+
  (Frochte 2026) + qualitative/human eval, and is itself a contribution-worthy
  observation in a no-baseline paper. [results/bench40_score_match_style{25,10}_*.png]

## What DOESN'T help (ablations — "the minimal form is sufficient")
### A1. Carrier (content-affinity MULTIPLICATIVE reweight of the residual) — INERT / at
best STYLE-CONDITIONAL. Five independent probes:
1. Matched 3x3 matrix (9 cells/method): carrier mean CSD 0.5935 vs score_match
   0.5993 — wash; Gram/leak slightly worse. [results/matrix_*]
2. Flat carrier-strength sweep {0.5,0.75,1.0}: never beats score_match at any
   strength (0.5912/0.5935/0.5994 vs 0.5993). [results/sweep_*]
3. Timestep schedule of carrier strength {const,back,front,sqrt,cosine} x 3 cells:
   no effect beyond seed noise (all |Δ vs const|<0.02, single-seed, inside the
   ±0.11 seed variance). [results/sched_*_metrics.csv]
4. CSD+ (CSLS, calibration-aware, Frochte 2026): does not rescue a carrier win;
   two justified gallery configs disagree in direction (style27 −0.011 / −0.029).
   [results/matrix_manifest_metrics_csdplus_fullgallery.csv]
5. Method-delta LPIPS: carrier changes the image 0.158 mean (~28% of a full
   method-swap 0.574) but STYLE-CONDITIONAL — style00 0.099 / style13 0.178 /
   style27 0.560. Changes impasto images substantially, near-nothing on flat
   styles; but the change is NOT a robust quality win (probes 1–4). [results/
   lpips_ablation_*]
   => Honest claim: "the content-affinity carrier is mostly inert; on heavily
   textured styles it perturbs the image but yields no robust quality gain on any
   measured axis." Literature diagnosis (docs/research/SYNTHESIS.md): it conflates
   structure-preservation with appearance-transport in one multiplicative term.

### A2. Timestep scheduling of injection strength — INERT.
Stage-1 deciding experiment (above, probe A1.3). Scheduling the dose (protect
content early / push style late, and the reverse control) produces no effect beyond
seed noise. => timing/magnitude is not a lever; the base flat form is fine.

### A3. Latent optimizer (Adam vs AdamW, wd 0/1e-4/1e-3) — NEAR-NULL.
adamw_three_pair: CSD moves 0.01–0.05 across optimizer/wd (consistent with
AdamW(wd=0) ≡ Adam). Robustness ablation, not a lever. [results/adamw_three_pair_*]

### A4. SLT / AdaIN (first-gen warm-start carryovers) — DISABLED, not part of method.
Formal runs use --disable-slt --adain-alpha 0 --disable-final-adain. LRS absent
(no-op) in gen-2. These are lineage from sec/Hyperion (gen-1), never the mechanism.

## Honest limitations (must appear in the paper)
- **Eval corpus is thin**: 11 unique style refs, 1 image per style, prompt-based
  content. Consequence: **ArtFID is NOT soundly computable** (FID-style covariance
  rank-deficient at n=11; content factor needs a content image we don't have) and
  **content-LPIPS needs a no-style baseline set we haven't generated**. Options:
  expand the style corpus (multi-image per style) OR justify an alternative suite
  (CSD+/CLIP-T/DINO/Gram + user study). UNRESOLVED — needs user scope call.
- **Seed variance is large on textured styles** (±0.11 CSD on style27) — single-seed
  deltas below ~0.02 are noise. Multi-seed needed for any small claim.
- **No external published baseline** (user directive: K/V = unpublished gen-1,
  dropped). Raises the internal-rigor bar (per lit_rf_and_eval.md): compensate with
  ablations (this ledger), breadth, causal analysis, human eval.

## The minimal-sufficient narrative (for the ablation section, under pivot framing)
"We ablate the content-affinity carrier, timestep scheduling, and latent-optimizer
choice; none improves over the plain native-score-residual form beyond seed noise.
The contribution is the minimal training-free native-score-residual latent
optimization itself; the negative ablations show its additions are unnecessary."
(This framing is pending user confirmation A/B/C in LOOP_PLAN.)
