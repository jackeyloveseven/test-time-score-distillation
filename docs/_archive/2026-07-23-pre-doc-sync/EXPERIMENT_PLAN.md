# Experiment plan — the complete set for the TSD paper (2026-07-20)
Organized by the paper's four contributions (see docs/research/PAPER_PITCH.md).
Each item: **What · Why · Intent(claim+reviewer question) · Status · Cost.**
Priority: **P0** = must-have for submission · **P1** = strengthens · **P2** = extension.

Contributions: **C1** TSD method · **C2** extrapolation-vs-interpolation mechanism ·
**C3** reverse-process-is-the-shield (DM unnecessary+harmful) · **C4** multi-dim VLM benchmark.

## ★ SCOPE & COMPLETENESS DECISIONS (2026-07-20, user-confirmed)
- **COMPLETENESS VERDICT: enough to fully finish an AAAI paper with exactly ONE required new run
  = E1 (breadth main-results table).** Everything else assembles from existing material.
  Minimal complete set = **E1 (must)** + assemble E2/E3/E5/E8/E12 + **recommended E6 + E9(a)**.
- **The negatives are a CONTRIBUTION, not a patch** → promote the "why every alternative fails" study
  to a **full paper section** ("A systematic study of what doesn't work, and why"): six alternatives
  (score_match / re-noise / DM / student / kv / carrier) all converge on CA-alone, each with a
  mechanism. This is a memorable strength — tell it well.
- **Mechanism + theory (C2/C3) are the CENTERPIECE** ("原理当主角"); evidence mostly exists.
  → **E6 (reverse-shield ablation) strongly recommended** — the only experiment that turns the
  reverse=shield theory from *argued* to *proven*; cheap (small code switch), highest ROI after E1.
- **E11 (layout-leak) DROPPED** — we do NOT reuse the first-gen's leak/LRS, so a leak metric is
  off-narrative. Removed below.
- **No external competitor baseline** is a deliberate scope choice; the six-way negative study + the
  40-style breadth ARE the comparison, and the core theory (DMD recast at test time) stands
  independent of baselines. Defensible; state plainly in Limitations.
- **Benchmark validity risk** (two same-vendor Gemini judges = circular): mitigate cheaply with
  **E9(a)** — one OPEN local 3rd judge for cross-judge agreement. Human study optional.

---

## C1 — the method works (breadth)

### E1 [P0] Breadth main-results table — TSD across all 40 styles × 3 seeds, StyleBench-quantified
- **What:** run TSD (`ca_only --cfg-scale 7`) on all 40 styles × seeds {64,65,66} × rotating
  prompts; StyleBench-score every cell; aggregate into the main results table (fidelity,
  style_capture, content_pres, + per-art-style/technique breakdown).
- **Why:** the "it works" claim currently rests on ~6 styles = anecdotal. A method paper needs a
  quantified result over the *full* benchmark, and the same run doubles as the **regression check**
  (does TSD fix hard failures *without* breaking styles score_match already handles?).
- **Intent:** the headline Results table + breadth figure. Answers the reviewer's first question —
  "does this work broadly or is it cherry-picked, and does the fix cost you the easy cases?"
- **Status (2026-07-23):** seed 65 is complete for TSD Semantic, TSD Global,
  native keep-composition, and native new-composition. Seeds 64/66 are queued
  for all four methods under a paired, resumable protocol; 320 cells remain.
  The current table is explicitly labeled fixed-seed cross-style dispersion.
  **Cost: queued, headroom-gated.**

### E2 [P0] Headline qualitative figure — score_match → TSD, diverse styles
- **What:** paired grid, same (style,prompt,seed): score_match (palette-only) vs TSD (texture+content),
  across 8–12 diverse styles.
- **Why:** the single most convincing evidence is visual; this is the "money figure."
- **Intent:** Figure 1 / teaser — the core improvement at a glance.
- **Status:**素材 mostly exist (caprobe/kv_vs_sm montages). **Cost: cheap (assemble).**

---

## C2 — mechanism: extrapolation vs interpolation

### E3 [P0] α-invariance sweep — cfg {1,4,7,10,15,20} on ~4 styles
- **What:** sweep guidance scale, quantify + show outputs are near-identical (flat).
- **Why:** proves α is *washed out by the rms-norm* → the mechanism is **extrapolation, not
  amplification**. Distinguishes us from "just crank CFG."
- **Intent:** mechanism figure/table; pre-empts "you just used a high guidance scale."
- **Status:** cfg 4/7/10/15/20 on 04/16 done; add cfg1 + 2 more styles. **Cost: cheap-ish (few runs).**

### E4 [P1] Saturation curve — style/texture metric vs inner-iteration, score_match vs TSD
- **What:** log a texture proxy (e.g., VGG-Gram distance or StyleBench fidelity of the intermediate
  x0) across inner iters for both losses.
- **Why:** to *show* score_match **saturates** (interpolation to the weak fixed point) while TSD keeps
  **accumulating** style (extrapolation) — the mechanistic story made empirical.
- **Intent:** mechanism figure; converts the code-level claim into a curve a reviewer can see.
- **Status:** not run. **Cost: moderate (instrument + a few runs).**

---

## C3 — reverse process is the free shield (DM unnecessary AND harmful) — the novel contribution

### E5 [P0] DM-harmful across styles — TSD vs TSD+DM (ca_dmd) on ~6 styles
- **What:** paired TSD vs ca_dmd (real fake critic) on styles where CA wins *and* marginal ones;
  StyleBench-quantify the style loss.
- **Why:** one style (04, mosaic lost) is anecdotal; the "DM harmful at inference" claim needs several
  styles + numbers.
- **Intent:** core theory experiment — adding DMD's distribution-matching term *degrades* style.
- **Status:** 04 done (harmful); need ~5 more + quant. **Cost: NEW (ca_dmd is heavy; ~6 runs).**

### E6 [P0] ★ The reverse-shield ablation (killer experiment) — remove/weaken the interleaved denoise
- **What:** a code variant that suppresses the frozen reverse step's manifold-contraction (e.g. run
  the CA optimization *without* interleaving `scheduler.step`, or freeze the state / over-iterate at
  one noise level), then push α up; show the sample **drifts off-manifold (over-saturation, high-freq
  artifacts)** — whereas *with* the reverse step it stays clean at the same α.
- **Why:** this **directly proves** the theoretical claim. Without it, "the reverse process is the
  shield" is an assertion. With it, we exhibit *what plays the shield role* and *what happens without
  it* — turning C3 from hand-wave into a contribution.
- **Intent:** the theory-nailing figure. Answers the skeptic of "DM unnecessary": here is the free
  regularizer, and here is the collapse when it's gone.
- **Status (2026-07-23):** code switch and full protocol are implemented:
  `--reverse-step-strength` defaults to 1 and tests `{1,.5,0}` under standard
  and stress settings on five styles. Generation is queued behind E1; metric,
  image-diagnostic, and local-VLM aggregation scripts are ready. **Cost:
  queued.**

### E7 [P1] No-collapse-at-high-α vs training-collapse framing
- **What:** show TSD stays artifact-free to cfg20 (extends E3); contrast with decoupled-DMD's reported
  CA-alone *training* collapse (cited).
- **Why:** supports "the collapse DM prevents is a *training* phenomenon (parameter drift), absent in
  single-sample inference."
- **Intent:** the inference-vs-training argument that grounds why DM is redundant here.
- **Status:** cfg20 evidence exists; formalize + write. **Cost: cheap (from E3).**

---

## C4 — multi-dimensional VLM benchmark

### E8 [P0] Full StyleBench breakdown across the benchmark
- **What:** report the 8-dim CoT results (art-style/technique/color/lighting/realism/complexity +
  fidelity/style_capture/content_pres/weakest_dim) aggregated + stratified.
- **Why:** the benchmark is a contribution; show its diagnostic value — e.g. `weakest_dim` localizes
  *where* a style fails (texture vs structure vs color).
- **Intent:** the Evaluation section; the benchmark's diagnostic power + the texture+content claim.
- **Status:** small subset scored; scale with E1. **Cost: cheap once E1 outputs exist (API).**

### E9 [P1] Benchmark validity — open 3rd judge + small human study
- **What:** (a) add an OPEN local VLM (sensenova, GPU idle) as a 3rd, non-Gemini judge → cross-judge
  agreement; (b) a small blind human preference study (kit in `human_study/`) aligning with VLM scores.
- **Why:** a benchmark validated only by two same-vendor (Gemini) judges is circular; reviewers demand
  human correlation and a reproducible (open) judge.
- **Intent:** makes the benchmark defensible ("human-aligned, reproducible"), not a black box.
- **Status (2026-07-23):** the third local judge is complete on 95/95 cells and
  passes the wrong-reference control, with moderate cross-judge agreement.
  Its exact checkpoint is private; a public-open-weight Qwen3-VL replication
  remains blocked by GPU occupancy. Human evaluation is intentionally excluded.

---

## Ablations & secondary metrics

### E10 [P1] Component ablations — optimizer / inner-iters / step-size / start-time / num-steps
- **What / Why:** vary each; show which are robust (no-op) vs which are real levers (e.g. iters/steps).
- **Intent:** the ablation table — demonstrates the method isn't a fragile hyperparameter stack.
- **Status:** partial (adamw). **Cost: moderate.**

### E11 — DROPPED (2026-07-20)
~~Content / layout-leak quantification~~. **Cut**: we do not reuse the first-gen's leak/LRS, so a
layout-leak metric is off-narrative. The "content preserved" claim is carried by StyleBench
`content_preservation` (E8) instead.

### E12 [P1] Aggregate CSD companion (de-emphasized)
- **What:** pooled-over-all-styles CSD as a coarse companion, with the calibration caveat (CSD can't
  rank quality within successes; validated against the VLM panel).
- **Why:** reviewers may want a "standard" metric; provide it honestly.
- **Intent:** completeness + the CSD-calibration analysis.
- **Status:** per-style CSD exists; re-pool + caveat. **Cost: cheap.**

---

## Extension (P2, days, user-gated)

### E13 [P2] Fast feed-forward student (TSD distilled)
- **What:** Tier-2 distillation — regress the CA-driver's *converged* outputs (or trajectory-force) into
  a fast generator.
- **Why:** answers "why not amortize the test-time optimization?" with a fast version.
- **Intent:** an extension / future-work section; honestly days-scale (B1 Tier-1 amortization was null).
- **Status:** infra mapped (train_score_residual_student.py, DiffSynth DMD2); Tier-2 is the days build.

---

## Recommended execution order
1. **E1 (breadth) — start first** (longest run; the headline main table + regression check).
2. **E2 / E3 / E5(assemble) / E12 — cheap, from existing** while E1 runs.
3. **E6 (reverse-shield ablation) — the theory-nailing new experiment** (small code switch).
4. **E8 (full StyleBench) — after E1 outputs.**
5. **E9(a) open 3rd judge / E10 — strengthen.** (E11 dropped.)
6. Then **write `AAAI/main.tex`**; E13 as optional extension.

**Two most valuable NEW experiments: E1 (breadth main table) + E6 (reverse-shield ablation).**
Everything else is either cheap-from-existing or a strengthener.
**Completeness: E1 is the only *required* new run; the paper finishes on E1 + assembled existing +
recommended E6/E9(a).**
