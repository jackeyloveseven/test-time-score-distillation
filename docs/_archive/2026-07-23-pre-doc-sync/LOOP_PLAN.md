# Loop Plan (historical log)

> ⚠️ **CURRENT STATE / ENTRY POINTS (updated 2026-07-21)** — read these first, this file below is the trail:
> - `docs/HANDOFF.md` — the handoff (method=TSD, findings, taboos, file map, next steps).
> - `docs/research/PAPER_PITCH.md` — live paper pitch (title/abstract/topics/why-others-fail).
> - `docs/research/LOSS_DECOMPOSITION.md` top "WHAT'S TRUE NOW" — findings summary.
> The method EVOLVED from "GoalCarrier"/score-residual to **Test-Time Score Distillation (TSD)** =
> `ca_only` CFG-augmentation extrapolation **(updated 07-21: "strong-cfg" is stale — `cfg_scale`/
> `lambda_ca` are algebraically washed to ZERO effect by the rms-norm; the live knobs are num-steps/
> iters × step-size × normalize-mode)**; core theory = frozen reverse process is a free distribution
> shield so DMD's fake-score is unnecessary+harmful at inference. Taboos: never K/V (unpublished
> first-gen), never headline "klein 9b" (show generality). The "GoalCarrier"/carrier framing below
> is SUPERSEDED.
> **07-21 addendum:** paper now organizes around ONE axis, **Controllable Style Locality** —
> Semantic Style Editing (composition kept, medium transferred) ↔ Global Style Transfer (composition
> dissolved via the "step-knob" = fewer denoise steps, medium kept). Global-Transfer campaign
> resolved (step-knob num-steps 8–10 wins; a layout-free primitive `v_cR_scram`, NOT K/V, parked at
> 3/5). A klein-native-edit baseline was ADDED this session (value-add proof, not a competitor
> table) — we win ~2× CSD, 5/5 styles. A novel metric SRSD is proposed, validation pending. Full
> detail: `docs/HANDOFF.md` 07-21 session block.

## (historical) Loop Plan: drive GoalCarrier to a submission-ready AAAI paper

Started 2026-07-19 (user directive: "设立 loop 全力向成正式论文努力,尽量不要让显卡空下来").
This file is the durable source of truth for the autonomous loop. Each loop
iteration: (1) check GPU / running jobs, (2) if GPU idle or under-used, launch
the next unstarted GPU batch, (3) advance one paper/non-GPU task, (4) eyeball &
record any finished results, (5) update statuses here. Never let the H200 sit
idle while any queued GPU batch remains.

## ====== VLM-JUDGE FALSIFICATION (2026-07-20, MATERIAL — read first) ======
Built an independent 2-judge VLM panel (gemini-pro+flash) rating style-match
blind-to-subject over all 95 cells (scripts/vlm_judge_calibration.py; verdicts in
results/vlm_judge/). Full writeup: docs/research/VLM_JUDGE_FINDINGS.md. Pre-committed
falsification test per advisor. OUTCOME — the paper's two big claims are hit:
- Instrument VALID: discrimination Δ=2.66 (correct 3.76 vs wrong-ref 1.09), inter-
  judge r=0.74, MY EYEBALL confirms worst-failure calls (style16/04 = only palette
  copied). (First read "r=0.61 → thesis falsified" was an OVER-SWING; corrected by
  4 cuts — see docs/research/VLM_JUDGE_FINDINGS.md.)
- Calibration contribution (iii) SURVIVES, SHARPENED + validated: among panel-
  EXCELLENT cells (n=63) CSD spans 0.36–0.86 → CSD cannot RANK quality within
  successes (Fig2 style25/style10 = genuine under-rated successes). Stronger than the
  original 2-example claim, now panel-backed.
- "uniformly sharp across 40 styles" = FALSE: 7 style TYPES fail on every seed
  (04,08,14,21,31,34,37); mechanism = content-reg dominates → palette-copy only
  (all 31 failures = content 5.0 + style 1.65). Content ALWAYS preserved (5.0 all).
- 54% (51/95) rated indistinguishable-medium by independent panel = a headline.
=> HELD (advisor): user chose A on premise "works excellently across 40 styles" =
now FALSE → basis of A decision invalidated + reshapes baseline question. Do NOT
rewrite main.tex, do NOT launch method-fix, until user re-decides with corrected
picture. Improve-lead (weaken content-reg rescues hard styles?) queued, not started.
## ================================================================================

## ====== FRAMING DECIDED + APPLIED (2026-07-20): USER CHOSE **A** ======
User answered the A/B/C framing question: **"我选择a"** → framing A is final =
base-method-as-contribution + honest negative ablations (carrier/schedule/optimizer
as negatives); differentiation = the mechanism itself (native same-state score
residual), NOT the carrier. Executed:
- AAAI/main.tex ← promoted from AAAI/main_baseA.tex (framing-A paper, aaai2027.sty,
  6pp letter, ZERO K/V, carrier only in ablation §). Reversible backup of the old
  K/V spine at AAAI/main_v1_kv_frozen.tex (33751 B). main_baseA.tex REMOVED (was a
  byte-identical dup → drift hazard); **AAAI/main.tex is the single canonical file**.
- Rebuilt clean: pdflatex×3 + bibtex all rc=0, 6 pages letter, NO undefined
  refs/citations, NO overfull>15pt, NO bibtex warnings.
Remaining OPEN (user-gated, none blocking): human study run (kit ready in
human_study/), eval-corpus scope (accept 40-single-ref + justified suite vs expand
for ArtFID), optional content-LPIPS no-style baseline set.

AUDIT (2026-07-20) of promoted main.tex — PASSED, no factual errors:
- Compiles clean; all 5 figs present; all 31 \cite resolve (33 defined).
- Headline numbers trace to results/*metrics*.csv & RESULTS_DRAFT (CSD 0.600
  CI[.562,.638], cell-level n=91 mean .597 sd .130, CLIP-T .311, leak .356, Gram 3.47).
- Under-rating figs VERIFIED by eyeball + data: style25 ref=B&W engraving archer,
  gen=engraved sunflower field, CSD=0.4901→"0.49" ✓ (kp3=sunflowers); style10
  ref=impasto portrait, gen=impasto castle-by-sea, CSD=0.38118→"0.381" ✓ (kp1=castle).
- Ablation table numbers match ABLATION_EVIDENCE (base .5993 / carrier .5935 / sweep
  .5912/.5935/.5994 / schedule |Δ|<.02 / optimizer Δ.01–.05).

KNOWN STALE ARTIFACTS (dormant, NOT in the built 6pp PDF — fix coupled to the
pending baseline decision + whether a supplement is wanted; do NOT half-fix):
- AAAI/supplement.tex = ENTIRELY pre-pivot: title "Carrier-Anchored Image CFG",
  presents GoalCarrier as MAIN method vs **K/V** baseline (lines 111/113/115 =
  taboo violation, but dormant), claims "main paper claim remains ... score-carrier".
  Needs full rewrite for framing A or drop. Referenced by 6 docs (README/BUILD/
  SUBMISSION_READINESS/AGENT_HANDOFF/full_paper_draft) → renaming cascades.
- AAAI/reproducibility.tex = stale wording ("baselines", "optional student and DMD
  extensions", "small matched benchmark", "three-pair results"). AAAI checklist is
  submission-required → must be updated to framing A (after baseline decision).
- Page budget: at 6pp; AAAI ~7 content pp + refs (UNCONFIRMED for 2027 specifically)
  → likely room for a baseline table/fig if the user adds one.
## ================================================================================

## ====== STRATEGIC PIVOT (2026-07-19 ~15:40, advisor-driven, evidence-forced) ======
STOP chasing a carrier/differentiator revival. The data forced it:
- Carrier is inert/dead FOUR independent ways: flat-strength sweep, timestep
  schedule (Stage-1, no effect beyond seed noise), CSD+ calibrated metric, eyeball.
- The BASE method (native same-state image-CFG score residual, optimize latent only,
  NO carrier = "score_match") already produces STRONG, sharp, content-preserving
  5-style+ transfer that cleared the user's explicit bar ("五个style优秀就够了").
  Discriminating check (advisor): content-preservation is NOT a deficiency of the
  base method — eyeball of score_match style02 (panda, anime+flame) = crisp,
  legible, prompt-faithful; CLIP-T mean 0.311 over 33 cells. => the carrier and the
  planned Stage-2 (additive self-sim) both target content-preservation, which is a
  NON-PROBLEM. Building S2 would be mechanism #5 chasing a dead differentiator.
- AAAI gaps (ArtFID/LPIPS, breadth, causal ablations, internal rigor — because no
  external baselines) need NONE of them a working carrier. The 4 carrier negatives
  ARE honest ablation content that strengthens a no-baseline paper.

NEW LOOP DIRECTION = RIGOR ON THE WORKING BASE METHOD (not carrier revival):
  1. ArtFID + LPIPS on images we already have (non-GPU-heavy, direction-independent).
  2. Breadth: extend the ~11-style grid; solidify the multi-style + multi-seed table.
  3. Ablation/analysis section: carrier=inert, schedule=inert, optimizer(adamw)=inert
     => "the minimal native-score-residual form is sufficient" is the honest story.
  4. Frame the CONTRIBUTION = the base method (native score-residual training-free
     latent optimization on rectified flow) + rigorous characterization; carrier etc.
     are negative ablations. Differentiation = the mechanism itself, not the carrier.
Stage 2 (decouple) DOWNGRADED to at-most a single-seed ablation, run only if a
content deficiency ever appears — NOT the mission.
NOTE: this changes the paper's identity vs the user's earlier "主打差异化(carrier)"
framing — surfaced to the user via AskUserQuestion (A accept pivot / B keep hunting
differentiator / C do base rigor first, decide framing later). USER AWAY / no
response. Proceeding with best judgment = option C: do the direction-INDEPENDENT
rigor now (ArtFID/LPIPS running bg ab99ff99; then breadth + ablation consolidation),
DEFER the final framing commit until hard metrics land + user weighs in. Option B
(build Stage-2/3 differentiator) stays open if user redirects. Do NOT rewrite
main.tex or delete the K/V spine until framing is user-confirmed.
## ================================================================================

## CURRENT GOAL (2026-07-19, user directive — supersedes prior sub-goals)
"调研顶会相关文献,借鉴想法,改进实验,直到得到更好结果,足以达到 AAAI 水平."
=> This is a RESEARCH→BORROW→IMPROVE loop, NOT a comparison-table loop. We do
NOT run competitor methods as baselines (explicit user directive). We survey
top-venue literature, borrow mechanisms, and improve OUR training-free
latent-only method until the RESULTS are AAAI-grade. The dead carrier means the
central research question is now: *what stronger training-free mechanism replaces
the carrier as the differentiator?* Phase-1 RESEARCH (ml-intern three-phase)
is running (3 lit-survey sub-agents, 2026-07-19).

## Fixed decisions (do not relitigate)
- Main method: native same-state image-CFG score residual, optimize latent only,
  frozen rectified-flow FLUX.2 [klein] 9B, train nothing. Formal flags = README cmd.
- Gen-2 positioning: self-contained, lead with differentiation. NO head-to-head
  vs first-generation (sec/Hyperion). SLT/AdaIN are disabled first-gen carryovers,
  not part of the method. LRS absent (no-op).
- **K/V baseline: DROPPED ENTIRELY (user directive, 2026-07-19).** K/V is the
  user's UNPUBLISHED first-gen baseline — it must not appear as a comparison
  anywhere (paper, tables, evidence). This REVERSES the earlier "keep+reframe"
  decision. Remove kv_match from the results narrative and DELETE the main.tex
  "K/V as prior-art baseline" reframe (do not soften — delete).
- **"对比adamw" resolved:** = the Adam-vs-AdamW(wd0/1e3/1e4) latent-optimizer
  ablation. Evidence (adamw_three_pair + single-cell wd scan): near-NULL — CSD
  moves 0.01–0.05 across optimizer/wd (consistent with adamw_wd0 ≡ adam). It is
  an honest robustness ABLATION, not a headline result. Keep for the ablation
  section only.
- **Carrier: NOT yet declared "dead" in print.** Inert on CSD (0.5935 vs 0.5993)
  + strength sweep, but that rests on one uncalibrated metric with no qualitative
  confirmation (advisor flag, 先定性后定量). Before printing it as a negative
  result: eyeball carrier-on/off pairs + check Gram/Leak deltas. Meanwhile the
  research is hunting for a mechanism that actually fires.
- Guardrails: see `PROJECT_STATE.md` (same-state branches, carrier gets no
  reference tensors, original Hyperion read-only, no force-add of result trees).

## Environment / cost
- Python: /home/ubuntu/chengyanli/flux/bin/python3 ; one H200 (143771 MiB).
- Peak mem per run: kv_match ~34 GB (no teacher), score_match/score_carrier ~63 GB.
- Safe parallelism: 2-wide for score-type (2x63=126<140); kv can go 3-wide.
- One run ~ 20-80 s of sampling + model load; batches run in background drivers
  or sonnet subagents so verbose logs stay out of the main context.

## GPU work queue (ordered by evidence value)
- [RUNNING] Q-A Matched multi-seed matrix. kv_match/score_match/score_carrier x
  {27/kp0, 00/kp1, 13/kp3} x seeds {64,65,66} = 27 runs. Driver:
  `scripts/run_matched_seed_matrix.sh` (bg, prefix `matrix`). Outputs:
  results/matrix_metrics.csv, matrix_metrics_summary.json, matrix_aggregate.json.
  Deliverable: main results table with mean/std/95%CI/win-rate. (P0 #1)
- [TODO] Q-B Carrier mechanism ablations on >=2 pairs: identity (carrier-strength
  0 / lambda_A=0), uniform affinity, shuffled content tokens, reference-derived
  affinity (leakage-positive control). Deliverable: mechanism table. (P1)
- [TODO] Q-C Carrier temperature x strength Pareto sweep on the matrix pairs.
  Deliverable: style-vs-layout frontier figure. (P1)
- [TODO] Q-D Broader reference set: ~10-12 additional styles x varied prompts x
  seed 64, quantitative + qualitative grid. Deliverable: benchmark-scale table +
  breadth figure (extends the 5-style 2026-07-19 eyeball check). (P0 #4)
- [TODO] Q-E Runtime / peak-mem / native-forward / latent-backward counts vs
  kv_match and the unmodified native edit sampler. Deliverable: cost table. (P0 #3)

## Non-GPU (paper) work queue
- [BLOCKED-ON-RESULTS] N-A/N-B **The current main.tex is a "we beat K/V" paper end
  to end** — K/V is the whole experimental spine: intro contribution (l.47,100),
  teaser fig (l.146), an entire "Why Not K/V Replacement?" subsec (l.323), the main
  results subsec+table (l.367,386), qualitative fig (l.399), three-pair table
  (l.456), audit table (l.506), properties (l.590). Per user directive this ENTIRE
  spine must be replaced, not reframed. DO NOT surgically delete K/V lines now — it
  would gut the paper and be redone once the new mechanism lands. GROUND-UP results
  rewrite is the LAST phase, AFTER: (1) research picks a live mechanism to replace
  the inert carrier, (2) it produces AAAI-grade results. Until then main.tex is
  frozen. (The earlier one-line "K/V as prior-art baseline" reframe is moot — the
  whole section goes.)
- [TODO] N-B2 New method/contributions/results section, written around the improved
  mechanism + its evidence (ablations incl. adamw robustness + carrier negative
  result, breadth, analysis). Baselines: none external (user directive); one-line
  tradeoff note only.
- [TODO] N-C Format gate: switch preamble to the official AAAI style. Kit is
  ALREADY on disk -- `aaai2027.sty` + `aaai2027.bst` at
  `/home/ubuntu/chengyanli/items/cherry/paper/` (and image-edit-lens/paper/).
  NOTE: target venue is **AAAI-2027**, not 2026 (2026 deadline long past as of
  2026-07-19); the "aaai2026" in the current preamble comment is a stale
  placeholder. Copy sty+bst into AAAI/, switch documentclass, preserve
  anonymity, rerun page/font/render audit. No external fetch needed.
- [TODO] N-D Keep PROJECT_STATE.md / SUBMISSION_READINESS.md gate statuses in
  sync as each item closes.
- [BLOCKED-HUMAN] Human preference study (P0 #2): can build the blinded A/B
  protocol + image pairs, but scoring needs human annotators. Prepare infra only.

## Verify-before-trust
- Every new results table traces to a committed metrics CSV + summary JSON.
- Eyeball a sample of every GPU batch before believing the metric (a high CSD on
  a mushy image is a red flag). The 2026-07-19 eyeball bar: sharp, non-blurry.
- Do not commit until a batch converges into a paper-ready artifact (Commit
  Pacing).落盘 + mirror is the default; commit is a milestone.

## GPU coexistence note
GPU state is EXTERNAL-JOB DEPENDENT — always re-check `nvidia-smi` before launching.
- 2026-07-19 ~14:2x: H200 SATURATED by 3 external jobs (~138/144 GB: deepcompressor
  PTQ 63 GB + a 61 GB job + a 14 GB job). Only ~5.5 GB free -> CANNOT launch any of
  our runs (kv 34 GB / score 63 GB both OOM). None are ours; NEVER kill. GPU
  sitting idle-to-us is due to external saturation, not slack. Wait for headroom.
- Earlier (~08-09h): probe/41_holdout_eval.py (26 GB) then worker.py (56 GB).
Rule: score-type = 63 GB, kv = 34 GB. Only launch when free mem clearly exceeds the
run's peak; go 1-wide while any large external job is present.

## Research synthesis inbox (Phase-1 reports -> pick mechanism to build)
- [IN] lit_rf_and_eval.md (RF mechanisms + AAAI eval bar). Key actionables:
  * Candidate carrier REPLACEMENTS: Dual-RF velocity subtraction (content/style
    twin-trajectory velocity diff as the style-transport signal); RF-Inversion
    faithfulness schedule (lock-early/release-late) to tame carrier's high seed
    variance; FlowEdit noise-averaging as cheap residual variance reduction.
    Carrier causal ablation template = Stable Flow vital-layer perturb-and-measure.
  * HARD: no training-free paper skips competitor comparison -> our no-baseline
    constraint raises the internal-depth bar. Missing metrics we likely need:
    ArtFID + standalone LPIPS. main.tex already cites DDS/CDS/StyleID/StyleAligned/
    InstantStyle + a CSD-calibration critique (VERIFY that cite is real, not a
    placeholder: "Frochte 2026, arXiv 2605.09030" looks suspicious).
- [IN] lit_style_transfer.md, lit_guidance.md — ALL 3 reports in.
- [VERIFIED-REAL] The CSD-critique cite `frochte2026csd` (arXiv 2605.09030, "When
  Style Similarity Scores Fail", Frochte, Bochum UAS, 2026-05) is GENUINE (WebFetch
  + WebSearch confirmed) — NOT a placeholder. It reports raw CSD cosine gives
  NEGATIVE discrimination gaps for 23/91 artists pairwise, 15/91 pooled => published
  backing for our empirical finding that CSD's 0.006 carrier "wash" is within the
  metric's own unreliability band ("carrier isn't dead, CSD can't see it"). ACTIONABLE
  for AAAI eval bar: adopt its **CSD+ readout (CSLS + positional-embedding interp)
  and/or the discrimination-gap diagnostic** instead of / alongside raw CSD cosine —
  a literature-grounded metric upgrade that also props the carrier argument.
- [DONE] Synthesis -> `docs/research/SYNTHESIS.md`. CONVERGENT top pick (chosen by
  all 3 agents independently): **timestep-scheduled injection strength** = Stage 1.
  Staged plan: S1 schedule the flat carrier-strength/residual-weight over the
  existing `progress` var (zero new forward; diagnoses magnitude-vs-direction) →
  S2 decouple structure (additive self-sim loss, not multiplicative reweight) →
  S3 PAG/SoftPAG attention-perturbation counter-term (validated on FLUX). Evidence
  gaps: add ArtFID+LPIPS; verify the CSD-crit cite is real; causal ablation via
  Stable Flow vital-layer template.

## Phase-2 PLAN (executing)
- [IN PROGRESS] S1 code: add a schedule option to carrier-strength + style-residual
  weight keyed on `progress` (runner l.441/881), shapes {cosine, sqrt, linear,
  reverse-control} × base strength; + a sweep driver over matched cells + textured
  styles. Delegated to sonnet; syntax-check now (GPU saturated), smoke + sweep when
  GPU frees. Gate: schedule turns wash→consistent win (esp. textured, low leak)?
  yes→carrier alive/cheap = paper; no→S2 (direction problem).

## ====== DECISION RESOLVED (2026-07-19) ======
User did NOT pick a "which baseline" option. User reframed the whole loop:
"调研顶会文献,借鉴想法,改进实验,直到更好结果,足以达到 AAAI 水平." + "别再跟
K/V 比了(它没发表)". So the strategic call is made:
- DROP the K/V comparison line entirely (it was never the point).
- Do NOT lean on "score beats K/V" as the headline — it beats an unpublished
  baseline, so it can't anchor the paper.
- The real work = borrow mechanisms from top-venue literature and IMPROVE the
  method's results (esp. a live replacement for the inert carrier). Phase-1
  RESEARCH sub-agents launched; synthesize their reports, pick 1–2 high-leverage
  mechanisms, implement, run, eyeball, measure, iterate.
Note: advisor confirmed do NOT run external competitor baselines (user redirected
away from that); one-line tradeoff in paper is enough ("reviewers may want a
published baseline; can add if rejected on that axis").
## ===========================================================

## !!! CORRECTION (2026-07-19 ~14:20) -- "carrier is dead" was a CSD artifact
Eyeballed 3 truly-matched carrier-vs-score_match pairs (matrix_ prefix, same config):
- style27 (impasto oil / palette-knife): carrier VISIBLY stronger — heavier knife
  texture, more faithful vivid yellow/orange palette, AND a cleaner/more legible
  panda subject (content preserved while context gets heavy style).
- style00 (pop-art halftone): carrier ~= score_match, near-identical.
- style13 (woodcut): carrier ~= score_match, near-identical.
=> Carrier is NOT inert; it is STYLE-CONDITIONAL. It bites only when the style has
strong texture that competes with content structure (impasto/palette-knife), acting
as a content-preservation knob; it is a no-op on flat graphic styles (pop-art,
woodcut). Global CSD averages the 1/3 hit to ~zero (0.006) AND cannot see "cleaner
subject." The prior "carrier dead" verdict was metric-driven and premature (advisor
was right; 先定性后定量). Implications for the improve-the-method loop: the live
research direction is (a) strengthen/broaden the content-preservation effect, and/or
(b) measure it with the RIGHT metric (content-preservation / layout-leak / perceptual
/ human), not global CSD. Feeds the lit-survey synthesis.

## (superseded) HEADLINE FINDING (2026-07-19 ~08:57) -- carrier differentiation did not survive
Q-A matched matrix (9 cells/method, 9/9 coverage, no OOM) mean CSD:
  kv_match 0.4586  |  score_match 0.5993  |  score_carrier 0.5935
- Native score-matching robustly BEATS K/V (0.60 vs 0.46, win 0.89, CIs above kv). SOLID.
- score_carrier does NOT beat score_match: mean is a wash (carrier slightly lower),
  Gram 4.08 vs 4.00 and Leak 0.3516 vs 0.3512 both slightly WORSE, CIs fully overlap.
  Per-cell carrier-minus-score deltas are noise (mostly +/-0.003) except one big loss
  (s27/kp0/seed65 -0.109) and two small wins (s27 seed64/66). The representative-pair
  advantage (0.6882 vs 0.6770 + better Gram/Leak) was a within-noise sweet spot, NOT
  representative. Eyeball: carrier images are sharp/good, just rebalance style->content
  (pull more content e.g. bamboo); no clear qualitative win over score_match either.
=> The paper's central differentiation (the carrier is what makes GoalCarrier work)
   is UNSUPPORTED on the matched matrix. This is a strategic fork for the user
   (pivot to score-vs-KV / re-tune carrier / demote carrier). Running the carrier
   strength sweep {0.5,1.0} as the decisive diagnostic before deciding.

## Loop log (newest first)
- 2026-07-19 ~18:2x: Built the last endorsed autonomous deliverable = human-study kit
  (human_study/rate.html self-contained blind Likert rater over 14 curated (ref,gen)
  pairs, blind to CSD, exports JSON; + pairs.json + README). Verified main_baseA.pdf/
  fig/main.tex all intact (main.tex still 08:29). ==> AUTONOMOUS SCOPE FULLY EXHAUSTED.
  Complete deliverable set: (1) AAAI/main_baseA.pdf (6pp compiling AAAI paper, framing
  A) + main_baseA.tex; (2) docs/research/{PAPER_DRAFT,RESULTS_DRAFT,ABLATION_EVIDENCE,
  SYNTHESIS}.md; (3) fig_breadth_grid.png; (4) human_study/ kit; (5) 40-style benchmark
  in results/. ALL remaining steps are USER-ONLY: confirm framing A/B/C, authorize
  main_baseA->main.tex swap, run human study, decide eval-corpus scope. Nothing left to
  build autonomously; holding for user.
- 2026-07-19 ~18:15: COMPLETE AAAI PAPER BUILT + VERIFIED. `AAAI/main_baseA.tex` ->
  `main_baseA.pdf`, 6 pages, aaai2027 double-column, anonymous, compiles clean
  (pdflatex+bibtex, rc=0, no undef refs). Independently verified: main.tex UNTOUCHED
  (mtime 08:29, still has 25 K/V mentions); main_baseA.tex has ZERO residual K/V;
  bib 26->33 (7 new verified entries); PDF page 1 renders as a real paper (title
  "Style Transfer for Free: Native Same-State Score Residuals..."). Framing A, carrier
  only in ablation, numbers all sourced. Honest flags for human: 6pp may read short
  (honest no-baseline scope), one "attention-feature replacement" reworded line, no
  inline reproducibility section, limitations carried over. ==> A submission-ready
  AAAI paper now EXISTS (separate file, reversible). Only remaining step = user
  authorizes rename main_baseA -> main.tex (+ optional framing tweak / eval-scope /
  human study). Autonomous work is COMPLETE to the appropriate boundary.
- 2026-07-19 ~18:1x: To satisfy the persistent goal-hook ("complete AAAI paper") WITHOUT
  over-stepping, chose the reversible path: build a COMPLETE COMPILING AAAI paper in a
  SEPARATE file `AAAI/main_baseA.tex` (main.tex UNTOUCHED; user greenlights the swap).
  Toolchain verified: pdflatex + xelatex available, aaai2027.sty/.bst on disk at
  items/cherry/paper/. Delegated to sonnet subagent (a59185469c0efe17c): render
  PAPER_DRAFT.md into aaai2027 LaTeX under framing A (K/V dropped, carrier as negative
  ablation), breadth figure + benchmark/ablation tables, compile-iterate to a clean PDF,
  report page count. On completion: verify it compiled + review, present PDF to user.
  This produces the submission-ready artifact reversibly; the final rename to main.tex
  still needs user authorization + framing confirm.
- 2026-07-19 ~18:0x: Drafted the headline BREADTH FIGURE -> docs/research/
  fig_breadth_grid.png (12 diverse styles, per-image CSD labels; visually shows breadth
  + the CSD-underrating point). Fixed labels to per-image CSD (were per-style min).
  Referenced in PAPER_DRAFT §5. Deliverables now: evidence package + complete paper
  draft + headline figure — all doc-only, main.tex UNTOUCHED. This is the appropriate
  autonomous ceiling; the final step (assemble into main.tex) is deliberately NOT taken
  without user authorization (hard-to-reverse, no git backup, framing-dependent, and
  the user has 3x corrected unconfirmed moves this session). Awaiting user: framing /
  eval-scope / main.tex authorization.
- 2026-07-19 ~17:55: Wrote a COMPLETE paper draft -> docs/research/PAPER_DRAFT.md
  (title/abstract/intro+contributions/related-work/method/setup/results/breadth/
  CSD-calibration/ablations/limitations/conclusion), under evidence-default framing A,
  [FRAMING-A] tags on the parts that change if user picks B. Doc-only; main.tex still
  UNTOUCHED (rewrite needs user authorization + framing confirm). This is the maximal
  reversible progress toward an AAAI-ready deliverable. User now has a full paper to
  react to + greenlight. Remaining = the 3 user-gated decisions (framing / eval scope /
  authorize main.tex rewrite) + optional human study. Logged 40-style exp to
  experiment-log. TRUE autonomous ceiling — awaiting user.
- 2026-07-19 ~17:50: seed65 DONE (25/25). 40-style benchmark now MULTI-SEED complete:
  91 cells, all 40 styles >=2 seeds, per-style CSD 0.600 (95% CI [0.562,0.638]),
  CLIP-T 0.311 [0.304,0.318], layout-leak 0.356 [0.331,0.381]. CIs added to
  RESULTS_DRAFT §2. ==> BASE-METHOD EVIDENCE PACKAGE COMPLETE: 40-style multi-seed
  breadth table w/ CIs + 5-signal carrier-null + schedule-null + optimizer-null
  ablations + CSD-uncalibrated analysis (style25/style10 figure examples) + CSD+
  metric + honest limitations + paper-ready RESULTS_DRAFT.md. AUTONOMOUS CEILING
  REACHED. Remaining path is USER-GATED: (1) framing A/B/C, (2) eval-corpus scope
  (accept 40-single-ref + justified suite vs invest in multi-image-per-style for
  ArtFID), (3) authorize main.tex rewrite (still the frozen "beat K/V" paper),
  (4) human study (needs annotators). STOPPING generation (marginal value low);
  handing off to user.
- 2026-07-19 ~17:2x: Drafted the AAAI-grade RESULTS section -> docs/research/
  RESULTS_DRAFT.md (setup / 40-style breadth headline / negative ablations /
  CSD-uncalibrated analysis / limitations / open items). Closes the Stop-hook's "no
  results presentation" flag (doc only, main.tex still frozen pending framing).
  Then launched multi-seed CIs: `bash scripts/run_bench40.sh 65` (bg b2cezpq3w) =
  seed-65 pass over the 25 new styles for 2-seed coverage -> confidence intervals on
  the breadth table (makes it AAAI-credible). On completion: recompute 40-style
  aggregate WITH CIs, add to RESULTS_DRAFT §2. AFTER THIS the base-method evidence
  package is essentially COMPLETE; remaining path is USER-GATED (framing A/B/C,
  eval-corpus scope, main.tex rewrite, human study) -> hand off.
- 2026-07-19 ~16:5x (wakeup + Stop-hook push): realized I'd been OVER-WAITING on the
  A/B/C framing — A and C both = base-method rigor, and evidence overwhelmingly kills
  B, so the rigor work needs NO user gate (only the final main.tex commit does).
  GPU FULLY FREE now (143 GB, deepcompressor finished). Discovered the style folder
  has 40 refs (grid only used 15) => the "thin corpus" gap is self-fixable. Launched
  `scripts/run_bench40.sh` (bg bapc2gd28): score_match on the 25 REMAINING styles x
  rotating prompts x seed64, 1-wide headroom-gated, to complete a 40-STYLE base-method
  benchmark (closes the hook's "eval corpus too thin" gap). Config pools with all
  score_match cells. On completion: eyeball a sample + full 40-style aggregate + note
  any weak styles honestly. NOTE: no-style/plain-T2I mode does NOT exist in the runner
  (only style-transfer modes); content-LPIPS needs a small new T2I script — deferred
  behind the higher-value 40-style expansion. ArtFID stays hard even at 40 (style-FID
  covariance needs >=64 samples/block; 40 refs still short) -> likely justify-and-omit
  rather than force.
  DONE 17:14 (25/25, no OOM): score_match now covers ALL 40 styles / 66 cells, CSD
  0.594±0.119, CLIP-T 0.312, leak 0.352. Eyeball uniformly strong. KEY FINDING: the
  low-CSD tail (<0.45) is NOT failure — it's SYSTEMATIC CSD-UNDERRATING (ref subject
  != prompt subject -> CSD penalizes a perfect style transfer). Clean figure examples
  style25 (engraving, CSD 0.49) + style10 (expressionist portrait ref -> excellent
  castle, CSD 0.381). => CSD mean understates true quality; strong support for both
  the base-method contribution AND the CSD-unreliability (Frochte) angle. Logged in
  ABLATION_EVIDENCE.md. Breadth gap CLOSED. Next autonomous (GPU-free): draft the
  AAAI results/ablation section in a DOC (not main.tex) from ABLATION_EVIDENCE.
- 2026-07-19 ~16:36 (wakeup): user still away (no A/B/C answer). GPU freed to 82 GB
  (one 60GB external job left). Per option-C best-judgment, launched direction-
  INDEPENDENT breadth extension `scripts/run_breadth_ext.sh` (bg b9zu8wzh0):
  score_match ONLY (base method), 4 NEW styles 05/11/18/25 x rotating prompts x
  seeds 64/65 = 8 runs, headroom-gated (>=68GB) + OOM retry, config identical to
  existing breadth grid (pools with results/breadth_*). Strengthens the multi-style
  base-method table (evidence needed under any framing). On completion: eyeball the
  8 outputs (sharp? content preserved?) + add to the score_match breadth aggregate.
  Batch kept to 8 (<=10, pacing). NOT running kv (dropped) or carrier (dead).
  DONE 16:47 (8/8, no OOM): mean CSD 0.581 / CLIP-T 0.320 / leak 0.341, in line
  with the grid. Eyeball: all strong. NOTABLE = style25 (B&W engraving ref) sunflower
  scene: visually one of the BEST transfers yet but CSD only 0.49/0.40 (ref subject
  = archer) -> clean Frochte CSD-unreliability example, logged in ABLATION_EVIDENCE.
  score_match now 41 cells / 15 styles, CSD 0.574 mean. Base-method table solidified.
- 2026-07-19 ~15:5x: ArtFID/LPIPS attempt -> surfaced a REAL AAAI eval-corpus gap.
  * ArtFID (arXiv 2207.12280, formula verified) = NOT computable here: FID-style
    factor needs Inception covariance over the style set, but we have only 11 unique
    style refs -> rank-deficient/degenerate at every Inception block (64-2048 dim >>
    11); pooling can't fix (only grows the Xg side, and per-method Xg sizes 9 vs 33
    confound relative ranking). Content-LPIPS factor needs a content IMAGE; ours is a
    prompt + NO no-style baseline exists in results/. => ArtFID deferred with a
    documented linear-algebra reason (honest, not lazy).
  * method-delta LPIPS (implemented, scripts/eval_artfid_lpips.py, CPU): carrier vs
    score_match = 0.158 mean but STYLE-CONDITIONAL (style00 0.099 / style13 0.178 /
    style27 0.560 as ratio of a full method-swap kv-vs-score=0.574). = carrier's 5th
    inert/conditional signal; changes the image on impasto but not a quality win
    (schedule+CSD already showed that). Advisor caught + fixed an apples/oranges
    over-read (cross-seed floor vs same-seed delta). Artifacts:
    results/lpips_ablation_pairs.csv, lpips_ablation_summary.json.
  * IMPLICATION (important): reaching AAAI eval rigor needs EITHER (a) a larger style
    corpus (multiple images per style) to make ArtFID/FID meaningful + a no-style
    baseline set (~9-33 plain-T2I imgs, same prompt/seed grid) for content-LPIPS, OR
    (b) an honestly-justified alternative metric suite (CSD+/CLIP-T/DINO/Gram + user
    study) that explains why ArtFID is inapplicable at single-ref-per-style scale.
    This is a genuine gap, not a formatting item. Queued: no-style baseline gen
    (GPU-gated, ~34GB plain-T2I is cheaper than 63GB score-type — verify runner
    supports a no-style mode first); decision on corpus expansion needs user framing
    (A/B/C) + is a real scope call.
- 2026-07-19 ~15:35: STAGE-1 GATE = NO ROBUST WIN (deciding experiment complete,
  all 5 modes x 3 cells seed64). Mean Δ vs const(flat): CSD back +0.000 / front
  +0.019 / sqrt -0.006 / cosine +0.002; leak cosine -0.011(best) / back -0.004;
  Gram back/front/cosine slightly better, sqrt worse. VERDICT: scheduling produced
  NO effect distinguishable from seed noise (all |Δ|<0.02, single-seed, inside the
  ±0.11 seed variance measured earlier). [Corrected over-read, advisor: do NOT read
  front's +0.019 as "falsifying style-late" — if it's noise it falsifies nothing;
  the honest statement is timing/magnitude is simply not a lever here.] Eyeball const vs
  cosine on style27 impasto: both good, cosine bg slightly more textured, but content
  NOT visibly better. => Per SYNTHESIS.md Stage-1 gate, scheduling (magnitude/timing
  redistribution) does NOT fix the carrier -> the weakness is DIRECTION, not
  magnitude. Carrier now tested 4 ways (flat-strength sweep, timestep schedule, CSD+
  metric, eyeball) all negative-to-inert. NEXT = Stage 2 (decouple structure from
  transport: additive self-similarity structure loss, NOT multiplicative reweight).
  Deferred: multi-seed confirm of the schedule (low value — effect already too small
  to be a headline). Data: results/sched_*_metrics.csv.
- 2026-07-19 ~15:2x: Sweep PARTIAL. const + cosine succeeded (3 cells each,
  full_cuda); back/front/sqrt ALL OOM'd during model load — root cause = a new
  external job (3019787, 21.6GB) started mid-sweep + deepcompressor grew 61.7->65GB,
  leaving ~89 MiB free exactly in the back/front/sqrt window (const/cosine + the
  manual smoke all ran fine same config => NOT a code bug, pure external contention;
  the mid-sweep-OOM risk I'd flagged). PARTIAL SIGNAL const vs cosine (1 seed):
  style27 impasto cosine = better content (leak 0.310->0.285) + texture (Gram
  2.45->2.27), CSD ~tie; style00 cosine +0.025 CSD +better leak; style13 cosine
  -0.015 CSD. => weak POSITIVE (scheduling helps content-preservation on textured
  style, mild on flat) but single-seed + mixed, NOT decisive. Launched
  `scripts/rerun_sched_failed.sh` (bg bv1dyd47e): headroom-GATED (>=68GB) full_cuda
  re-run of the 9 OOM'd cells, byte-identical config (no confound vs const/cosine),
  idempotent. GPU env is CHURNY (external jobs cycle 20-65GB every few min) so the
  gated wrapper slots into windows. On completion: analyze all 5 modes (CSD + leak)
  + EYEBALL back/sqrt/cosine vs const on style27, then the gate decision.
- 2026-07-19 ~15:1x: GPU freed (21GB ext job cycled out -> 76 GB free, deepcompressor
  66.6GB remains). Fired Stage-1 DECIDING experiment. Smoke (score_carrier
  --carrier-schedule back, style27/kp0/seed64) PASSED both gates: exit 0, 25 steps
  ~28s, and EYEBALL = sharp/strong-impasto/clean panda, no degradation vs flat
  carrier. => code path valid. Launched full sweep `run_carrier_schedule_sweep.sh`
  (bg bt4p1x21e): 5 schedules {const,back,front,sqrt,cosine} x {27/kp0,00/kp1,
  13/kp3} x seed64 = 15 runs + per-mode eval + combined analyze. GATE on completion:
  does any schedule turn flat carrier's wash into a robust win on CSD (style) OR
  dino_layout_leak (content)? Eyeball across modes REQUIRED (visible effect is off
  the metric axes). const mode = in-sweep flat-carrier baseline for direct compare.
- 2026-07-19 ~14:5x: CSD+ (CSLS, Frochte 2026) IMPLEMENTED + run on the real "0.006
  wash" data. Formulas verified verbatim from the PDF (k=15; singleton-anchor
  collapse handled); discrimination-gap HONESTLY DEFERRED (corpus too small, coded
  as status=deferred not faked). RESULT: CSD+ does NOT rescue a carrier style win at
  style27 — the two justified gallery configs DISAGREE in direction (N=3/k=2:
  −0.011; N=40/k=15: −0.029), i.e. inconclusive-by-construction at our N. KEY: the
  eyeballed carrier effect (cleaner subject on impasto) is a CONTENT/LAYOUT-axis
  effect that NO style-similarity metric (CSD or CSD+) can see. => Refined carrier
  state: STYLE axis = neutral-to-negative (CSD & CSD+ agree); CONTENT/LAYOUT axis =
  possible but high-variance + not cleanly measured. Stage-1 schedule sweep is the
  DECIDING test: can scheduling produce a robust win on ANY measurable axis? Eval
  must read CSD (style) + dino_layout_leak (content) AND eyeball (visible effect is
  off the metric axes). Files: results/matrix_manifest_metrics*.csv. GPU: freed
  19->59 GB (one big external job cycled out) but 59<63 peak => still waiting for
  >70 GB clean headroom; scheduled poll at ~15:04 will fire the sweep.
- 2026-07-19 ~11:xx: GOAL RESET by user -> research→borrow→improve→AAAI. Closed
  the "对比adamw" question from artifacts (near-null robustness ablation, not a
  headline). K/V dropped entirely (reverses keep+reframe). advisor consulted:
  stop chasing external baselines, resolve adamw from files not the phrase, delete
  (not soften) the K/V reframe in main.tex, don't print "carrier dead" on one
  uncalibrated metric w/o an eyeball. Launched 3 lit-survey sub-agents (sonnet):
  (1) training-free style transfer, (2) test-time score/CFG guidance, (3) RF
  editing + AAAI eval bar -> reports to docs/research/lit_*.md. External GPU job
  now = deepcompressor PTQ (`...app.diffusion.ptq`, ~62 GB) — NOT ours, never
  kill; ~81 GB free, 1-wide safe. Next: synthesize reports -> pick mechanism(s) ->
  implement -> run -> eyeball/measure -> iterate. Also pending: delete K/V reframe
  in main.tex; eyeball carrier on/off before any "negative result" wording.
- 2026-07-19 ~10:25: multi-seed breadth COMPLETE (24/24 per method, no OOM). 8
  fresh styles x 3 seeds, score_match vs kv: CSD 0.5612 vs 0.4661, win/kv 0.71,
  CIs [0.523,0.597] vs [0.418,0.520] (barely disjoint). SOBERING vs the seed-64
  slice (was win 0.88, gap +0.19): adding seeds 65/66 raised kv to 0.466 and cut
  win to 0.71, gap to +0.095. Leak still favors kv (0.360 vs 0.328). => the
  score-vs-kv headline is real but MODEST (win ~0.71-0.89 depending on style set,
  +0.10 CSD pooled), not a blowout. Both research lines now fully characterized:
  carrier = inert (dead), score-vs-kv = solid-but-modest. Stopping breadth
  expansion (diminishing returns + direction unresolved). Next: Q-E cost table
  from EXISTING summaries/logs (no new GPU). Still HOLDING paper rewrite for user.
- 2026-07-19 ~09:45: breadth batch COMPLETE (8/8 per method, no OOM). 8 fresh
  styles (02,07,15,22,29,34,36,39) x rotating prompts x seed 64, score_match vs
  kv: CSD 0.5688 vs 0.3801 (win/kv 0.88, CIs disjoint), Gram 3.89 vs 4.18. Honest
  costs: kv better CLIP-T (.326 vs .310) and Leak (.305 vs .363). Eyeball style07
  (ink-wash castle) + style29 (pencil fisherman) score_match = sharp, no mush.
  => "native score-matching beats K/V" now spans 11 styles, robust. External GPU
  job changed: probe(26GB) gone, now `worker.py`(56GB) -> only ~87GB free, 1-wide.
  Next: extend breadth to seeds 65/66 for a true multi-seed benchmark (prefix
  breadth, --expect-cells 24). User has NOT answered the paper-direction fork yet;
  still HOLDING the paper rewrite.
- 2026-07-19 ~09:28: carrier strength sweep COMPLETE and DECISIVE. Mean CSD:
  carrier@0.5=0.5912, @0.75=0.5935, @1.0=0.5994 vs score_match=0.5993. Carrier
  never beats score_match at any strength (best case = tie at strength 1.0, worse
  below). => salvage-the-carrier (option 2) is empirically dead. Realistic paper
  paths: option 1 (pivot to score-vs-KV, DEFAULT/recommended) or option 3 (carrier
  as a control knob, weak). Held paper rewrite for user (AskUserQuestion unanswered,
  user away). Launching direction-INDEPENDENT breadth work (score_match vs kv over
  more styles = P0 #4) to keep GPU busy on the one solid result.
- 2026-07-19 ~08:57: Q-A matrix COMPLETE (see headline finding above). Launched
  `scripts/run_carrier_strength_sweep.sh` (bg, harness-tracked) = does carrier beat
  score_match at any strength? Surfaced the fork to the user. HOLD on Q-B/N-B until
  the user steers the paper direction.
- 2026-07-19 ~08:30: Q-A OOM incident -- 2-wide score_match hit CUDA OOM (7/9
  score_match failed) because an external 26 GB job left only ~117 GB. Killed the
  2-wide driver (kept 9/9 kv results), relaunched `scripts/run_score_serial.sh`
  (1-wide) for the 18 score-type cells. `analyze_matched_matrix.py` upgraded:
  dual baseline (win vs kv AND vs score_match) + 9/9 coverage gate.
- 2026-07-19 ~08:2x: N-A done -- reframed K/V as generic prior-art baseline
  (cite stylealigned+styleinjection), stripped SLT/AdaIN lineage in main.tex
  protocol, softened "Hyperion" -> "external read-only source". Numbers untouched.
  N-C unblocked: aaai2027.sty is on disk (target venue = AAAI-2027, not 2026).
- 2026-07-19 ~08:20: launched Q-A matrix driver in background (2-wide). Started
  N-A. Loop established.
