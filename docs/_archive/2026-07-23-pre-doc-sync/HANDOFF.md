# HANDOFF — read this first (updated 2026-07-22)

## 2026-07-23 benchmark hardening update

- Baseline scope is now explicit: the controlled baseline is the same native
  MMDiT image+text edit path that TSD modifies. Published methods appear only
  in Related Work; no external-method experiment table is required.
- The existing 40-style table was audited. All 160 rows are seed 65; its
  `mean +/- SD` is cross-style dispersion, not multi-seed uncertainty. The
  manuscript and figure package now state this exactly.
- Added resumable paired protocol for four methods x 40 styles x seeds
  `{64,65,66}`:
  - `scripts/run_benchmark_multiseed.sh`
  - `scripts/benchmark_multiseed.py`
  - current inventory: 40/120 present per method, 80 missing per method.
  - aggregation reports style-bootstrap 95% CIs, seed-level SD, cell/style win
    rates, wall time, peak memory, forward calls, and backward calls.
- Added default-preserving `--reverse-step-strength` to the main runner, plus
  inference/process wall time and true DiT forward-hook accounting. Added the
  standard/stress reverse-shield protocol in
  `scripts/run_reverse_shield_ablation.sh` and
  `scripts/reverse_shield_ablation.py` (`rho={1,.5,0}`, five styles).
- The native MMDiT baseline runner now accepts `--seed` and writes per-image
  runtime/memory/forward-call summaries without changing seed-65 filenames.
- Completed a third local VLM judge over all 95 calibration cells. Results:
  local vs Gemini-Pro Pearson 0.694 / Spearman 0.657 / within-one 82.1%; local
  vs Gemini-Flash 0.588 / 0.507 / 84.2%. Wrong-reference control: 4.21 vs 1.33,
  correct higher on 89.5%. This is a private Gemma-4-family checkpoint, so it
  is independent/local validation, not yet public-open-weight validation.
  Files: `scripts/vlm_open_judge.py`, `results/vlm_judge/open_local_*`, and
  `AAAI/figures/open_judge_validation/`.
- Literature search completed with 77 unique 2023--2026 records and recorded
  source failures/429s in project-root `allinone.md`. Related Work now cites
  representative attention, sampling, score-residual, and rectified-flow
  editing routes without adding external experimental baselines.
- Current PDFs after visual QA: `AAAI/main.pdf` = 6 pages;
  `AAAI/supplement.pdf` = 7 pages. No overfull boxes or undefined references.
- GPU blocker at update time: only about 7 GiB free; three foreign processes
  occupy about 136 GiB. A 5-GiB offload smoke still OOMed while loading the 8B
  text encoder, so changing precision/encoder is rejected as benchmark drift.
- Persistent serial queue: `scripts/run_remaining_validation.sh`. It runs the
  multi-seed benchmark first, then reverse-shield ablation, and never kills
  foreign jobs. Check the `tsd-validation` tmux session and
  `agent_runs/remaining_validation_driver.log`.
- Remaining non-human gap after that queue: rerun the third-judge protocol on
  a publicly downloadable open-weight VLM (the local Qwen3-VL-8B checkpoint is
  available, but cannot be loaded under current GPU occupancy).

> # NEXT AGENT — CURRENT FIGURE STATE (user-confirmed 2026-07-22)
> The old v4 generation task is complete and superseded. **Do not regenerate it as the next action.**
> The approved mechanism figure is `AAAI/figures/tsd_framework/tsd_framework.png`; it is wired into
> `AAAI/main.tex`. Its exact Prompt, generation intent, top-conference design research, caption, Kaon
> GPT Image 2 script, and prohibited drift are self-contained in that directory. The separate
> Semantic Style Editing draft lives in `AAAI/figures/semantic_style_editing/`; its concept is retained
> but its layout is explicitly rejected. Every active paper figure now has its own self-contained
> package. Read `AAAI/figures/FIGURE_WORKFLOW.md` before creating or modifying any figure.

A fresh agent should be able to continue from this file + the three it points to.

## Historical 2026-07-22 session update — original framework campaign
Session was cut short by Claude quota; below is the exact resumable state. No experiments run this session.

**A. Framework / method figure (the active work) → full state in `AAAI/figures/pipeline/README.md`.**
- Route DECIDED (user override, do not re-litigate): **raster via gpt-image-2**, ref-guided by the user's
  exemplar `/home/ubuntu/chengyanli/20260722-111414.jpg` (the decoupled-DMD paper's own method figure).
  User rejected matplotlib/mermaid vector as amateur; chose gpt-image even knowing raster garbles dense math.
- **APPROVED look:** `AAAI/figures/pipeline/gptimg_arch_ref_v3.png` (user: "非常非常好"). Muted pastels,
  real-photo thumbnails, minus-nodes, curly brace, dashed (a)DMD/(b)Ours view-boxes, the CA+DM vs CA-only
  loss algebra. v1/v2/v3 all at 1675×939.
- **NEXT ACTION (queued, prompt ready, NOT generated — do this first):** the user asked for an improved v4
  with 3 additions — (1) integrate both methods' formulas ((a) `∇L=CA+DM`, (b) `∇L=CA` only), (2) left column
  ends in our two IDEAL outputs = the two modes (Semantic T=25 kept / Global T=8 dissolved) on a `local↔global`
  axis, (3) the DM branch GREYED (not hard-X) + `DM ≈ frozen reverse process (free shield)`. The verbatim v4
  prompt + the regenerate recipe are in `AAAI/figures/pipeline/gptimg_academic_prompt.md`. Run it → v4a/b/c →
  eyeball each vs the exemplar (human's eye is the gate) → show user. **Do NOT fabricate a second loss** (only
  CA exists; two modes = one CA + step-knob T; derived 2nd losses all failed = the K/V wall).
- After look is locked: composite 3 real bronze thumbnails onto the LEFT (paths in the figure README), wire the
  chosen PNG into `AAAI/main.tex` replacing the old `pipeline.png`, write caption. Optional: thin vector formula overlay.
- Pitfall logged: `experiment-log/2026-07/log.jsonl` (raster can't render crisp derivation math — tool ceiling, not a prompt bug).

**B. Title + abstract UPDATED to current truth → canonical `docs/research/ABSTRACT_TITLE_v3.md` (supersedes v2).**
- Also mirrored into `PAPER_PITCH.md` (primary title line + Abstract block). **New primary title (user-picked "A" 2026-07-22):**
  *"Test-Time Score Distillation: Mask-Free Semantic Style Editing where the Reverse Process Replaces the Fake Score."*
  Reason: headline the concrete/defensible strong mode (mask-free Semantic Style Editing) not the abstract axis;
  **Controllable Style Locality stays as the internal §3 organizing axis**, Global Transfer = second mode / free-knob bonus.
- Abstract now folds in the two-mode Controllable Style Locality axis, the ~2× CSD win over the backbone's
  native edit, and the honest texture-reproduction (K/V) wall. `ABSTRACT_TITLE_v2.md` is now OUTDATED (trail only).

## What this project is
Second-generation **training-free style transfer** paper, target **AAAI-2027**.
Root: `/home/ubuntu/chengyanli/items/style_residual_distill_20260709`.

## 2026-07-21 session update — Controllable Style Locality (read this first; extends 07-20 below, doesn't replace it)
The paper is now organized around **one axis, "Controllable Style Locality,"** with two modes:
- **Semantic Style Editing** (main mode, mask-free, works well): subject stays in the reference's
  composition, only the medium transfers. Composition **KEPT** (`dino_layout_leak` 0.45–0.82).
- **Global Style Transfer**: reference composition dissolves, medium is kept. Composition
  **DISSOLVED** (`dino_layout_leak` 0.37–0.46). `dino_layout_leak` (patch self-similarity) is the
  metric axis that separates the two modes.

**Global Transfer campaign — RESOLVED** (this session's E3–E6 = a NEW campaign series; distinct
numbering from the 07-20 EXPERIMENT_PLAN's own E1/E5/E6/E8/E9/E12/E13 below — don't conflate):
- Velocity-space *derived* losses ALL fail: A (scalar content-repel `d_R−b·d_C`), B (projection),
  D (multi-view marginalization) — each collapses to "semantic-edit or lose-medium," i.e. the K/V
  wall the paper's mechanism predicts, now confirmed empirically.
- A taboo-clean **layout-free** primitive `v_cR_scram` (position-scramble of `edit_image_ids`,
  `src/native_edit_teacher.py:build_layout_free_edit_condition`; **NOT K/V** — it perturbs only
  input positional ids, never touches K/V) = **E3: 3/5 grid styles work** (bronze/pop-art/
  stained-glass — composition dissolved + medium kept + subject present) but **2/5 destroy the
  subject** (mosaic-portrait/oil-portrait). **PARKED as a niche potential** (strong-frame media the
  step-knob can't dissolve) — not adopted as the Global-Transfer method.
- The WORKING Global-Transfer lever = the **"step-knob"**: `ca_only` with FEWER denoise steps.
  **E4: usable band = num-steps 8–10.** `iters` is NOT an escape hatch — it couples to structure,
  doesn't break the dissolve↔sharpness tradeoff.

**Key mechanism fact (verified in code, sharpens the "α washed out" finding below):**
`pseudo_grad = lambda_ca·(cfg_scale−1)·delta_ca_score` is then rms-normalized in
`build_pseudo_target_loss` (default `--pseudo-grad-normalization rms`, scale-invariant) →
**`cfg_scale` and `lambda_ca` have ZERO effect on `ca_only` output** — not just a flat sweep, they
are algebraically washed out. **The run command below (`--cfg-scale 7 --lambda-ca 1.0`) is
harmless-but-pointless now; drop those two flags.** The live tunable knobs are: **num-steps / iters
× pseudo-step-size × normalize-mode {rms, rms_token}.**

**Recommended params (mode-specific — do NOT collapse to one recipe):**
- **Global transfer:** num-steps 8, step-size 0.12 (E5 best: CSD≈0.63 rms / 0.64 rms_token vs. the
  s08 anchor's CSD 0.53; layout_leak still ~0.45 = composition stays dissolved).
- **Semantic editing:** iters 5, step-size 0.08, rms — **still 25 outer steps, this did NOT drop to
  8** (E6 balanced pick: CSD 0.74 / CLIP-T 0.28, vs. anchor q25s CSD 0.70 / CLIP-T 0.285). The
  CSD-max cell (`i8/s08/rms_token`) over-stylizes: CSD 0.82 but CLIP-T crashes to 0.21, layout_leak
  0.82 (subject degrading) — **do not use the CSD-max cell.**

**Baseline comparison ADDED this session** (updates TABOO #4 below — see that item): "directly
prompt klein (native image+text edit)" = DiffSynth's native `Flux2ImagePipeline` edit — the base
model's own honest "what happens without TSD" (klein's frozen-reverse trajectory does NOT consume
the edit image at `--lambda-ca 0`, so this native-edit path is the real baseline; cfg_scale=4, 25
steps; scripts `diagnostic_kleinedit_baseline.py` / `diagnostic_kleinedit_sysprompt_baseline.py`).
Klein exposes no system-prompt channel via public API; a strong expert style-transfer instruction
was folded into prompt + negative_prompt. **Result: even at its best (strong) prompt, klein native
edit either fails to transfer style OR couples style with composition (can't dissolve layout). CSD:
klein-strong sem 0.36 / global 0.29 vs OURS 0.70 / 0.53 — we win ~2×, 5/5 styles.** klein CLIP-T is
marginally higher (0.31 vs 0.29) = an honest subject↔style tradeoff. Klein's tell: reverts to a
generic "engraving/lineart" look, only nails relief media (bronze09).

**Metric stack** (unchanged: CSD / CLIP-T / `dino_layout_leak` via
`scripts/evaluate_matched_teacher_outputs.py`) **+ one NEW proposed metric, validation PENDING:**
**SRSD** (Score-Residual Style Distance) — single-shot, generator-native = ‖v_cR −
v_c0(full_style_prompt)‖ on the generated image (lower = more style-faithful, no external network).
SRSD↔CSD validation **DONE 07-21 → REJECTED** (correlation POSITIVE +0.79/+0.88 via content-collapse, not style; within-config r=0.20 n.s.; scale-corrected unchanged). NOT a valid style metric — do not use/flip. Kept as an honest null in paper §4; keep the 3 external axes.
Reinforced lesson: CSD alone is misleading — must pair with CLIP-T, else it rewards subject-
destroying over-stylization (see the CSD-max cell above).

**Figures published to R2** (`https://yata-image-publics.flowgpt.com/felix/`):
gst_e5_paramgrid_{rms,rms_token}_kp0.png, se_e6_paramgrid_{rms,rms_token}_kp0.png,
klein_fair_compare_kp0.png, mainline_semedit_multiprompt_montage.png,
paper_q1_method_csd_bar.png, paper_q2_paramgrid_csd_heatmap.png, paper_q3_csd_clipt_scatter.png,
gst_e3_layoutfree_montage_kp0.png, gst_e4_stepknob_opt_montage_kp0.png.

**Remaining work (updates "Next steps" below):** SRSD↔CSD validation; main-line semantic
multi-prompt breadth — kp1 castle / kp2 cat DONE & strong, **kp3 sunflowers / kp4 fisherman + full
38-style breadth still TODO.**

## The method — it EVOLVED; don't use the old framing
**Winner = Test-Time Score Distillation (TSD)** = `style_mode=ca_only` at strong `--cfg-scale`
(~7), **same-state** (no re-noise). The CFG-augmentation direction
`g=(α−1)(v_with − v_without)` is applied as a **fixed-magnitude, rms-normalized pseudo-guidance
step** on the current latent (`x0 ← x0 + s·rmsnorm(g)`), ~5 AdamW inner iters × 25 outer
denoise steps, on a frozen rectified-flow transformer. **Optimize the latent only — train no
weights.**
- `score_match` (the old "native same-state score residual") is the **weaker predecessor** —
  palette-only, misses texture. **Superseded by the CA-driver (TSD).** Don't re-pitch it as the method.
- Run it: `run_tpami_with_style_teacher.py --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0
  --pseudo-step-size 0.05 --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw
  --latent-weight-decay 0 --disable-slt --adain-alpha 0 --disable-final-adain --style-id NN
  --prompt-id kpN --seed S --out-prefix ...`

## What's proven (authoritative: docs/research/LOSS_DECOMPOSITION.md → "WHAT'S TRUE NOW")
1. **Mechanism (code-verified):** CA = *extrapolation-walk past* the weak fixed point; score_match
   = *interpolation to* it (saturates). α is washed out by the rms-norm → **flat α-sweep cfg4≈7≈20**.
2. **DM/fake-score unnecessary AND harmful at inference:** the frozen reverse process (one
   `scheduler.step` per outer step) is a free distribution "shield"; explicit DM (ca_dmd) pulls
   toward the naturalistic manifold and erases style (mosaic lost). Scheduled DM (`--dm-schedule
   front`) gives no gain on the marginal styles either. → the paper's theoretical hook.
3. **Quant (StyleBench-CoT, the user's benchmark in `stylebench/`):** CA moves style_capture
   **partial→full AND content_pres full** on hard styles (04/16/14) — texture gained *and* content
   preserved (the anti-K/V-leak claim). Judge verified to discriminate.
4. **Student smokes:** B1 supervised distill = null (student learns ~40% of the direction,
   delta_cosine 0.40); B2 ca_dmd = DM harmful. A real fast student = **Tier-2 = days** (regress the
   CA-driver's converged outputs / trajectory-forcing), not a smoke.

## Paper framing (docs/research/PAPER_PITCH.md is the live pitch)
Title: **"Test-Time Score Distillation: Style Transfer where the Reverse Process Replaces the
Fake Score."** Positioning = SDS/DDS lineage (single-sample score distillation) × DMD's
CFG-augmentation(spear)/distribution-matching(shield) decomposition × CFG-as-extrapolation.
Novelty = DMD recast at test time; the shield is free.

## TABOOS / constraints — do not violate
1. **K/V is off-limits entirely.** It's the user's UNPUBLISHED first-gen (reserved for a stronger
   venue). Never a comparison, never the method's mechanism (kv/attention-matching collapses content
   AND = self-plagiarism). SLT/AdaIN are first-gen carryovers, not part of TSD.
2. **Don't name "FLUX.2 [klein] 9B" in the title/abstract/contributions.** Show generality —
   architecture-agnostic by construction (needs only the model's native image-conditioning + standard
   reverse process). Mention the specific model minimally, in experimental setup only.
3. **Be honest:** TSD is *training-free of weights* but *optimization-based*. Marginal (already-graphic)
   styles 14/21/34 staying marginal = an honest **limitation**, not a bug to kill.
4. **No external competitor-baseline table** (advisor-confirmed, 07-20). Six *internal* alternatives
   (score_match, re-noise τ<t, DM/fake, kv-match, student, carrier) all tested → all converge on
   CA-alone. Don't reopen that internal-negatives study.
   **(updated 07-21): this taboo is now SCOPED, not absolute.** We added ONE comparison this
   session — klein's own native image+text edit capability (`Flux2ImagePipeline`, the base model's
   honest "what happens without TSD") — as the value-add proof, NOT a table of competing published
   methods. See the 07-21 session block above for numbers (we win ~2× CSD, 5/5 styles). Still do
   NOT add a table of external published competitor methods.

## Next steps — completeness verdict (2026-07-20, user-confirmed)
**(updated 07-21): see the session block near the top of this file for the now-current remaining
items — SRSD↔CSD validation + kp3/kp4 + full 38-style breadth. The E-numbers immediately below are
the OLD 07-20 EXPERIMENT_PLAN series, distinct from the NEW 07-21 E3–E6 Controllable-Style-Locality
campaign described above.)**
Full plan with What/Why/Intent: **`docs/EXPERIMENT_PLAN.md`** (has the SCOPE & COMPLETENESS block).
**The paper finishes on ONE required new run + assembled existing material:**
1. **E1 (REQUIRED, the only must-run): breadth main-results table** — TSD across 40 styles × seeds,
   StyleBench-quantified → main table + breadth figure + the fix/no-regression check. Longest run;
   start first. **Use the Bash tool's `run_in_background`, headroom-gate, skip-existing.**
2. **Assemble from existing (cheap):** E2 headline score_match→TSD figure · E3 α-invariance · E5
   negatives · E8 full StyleBench · E12 pooled-CSD companion.
3. **Recommended NEW (high ROI):** **E6 reverse-shield ablation** (small code switch: remove the
   interleaved `scheduler.step`, show drift = *proves* reverse=shield) · **E9(a)** one OPEN 3rd VLM
   judge (benchmark validity).
4. **Then write `AAAI/main.tex` from PAPER_PITCH.md** (aaai2027 template in `AAAI/`; figs from
   scratchpad/{caprobe,cabatch_alpha,b2_cmp,dmsched_cmp}.png → AAAI/figs).
- **Framing decisions:** negatives = a **full section** (contribution, not a patch); **layout-leak
  DROPPED** (no LRS reuse); mechanism+theory = centerpiece; no external baseline (scope choice, the
  negatives + breadth are the comparison).
- E13 Tier-2 fast student = P2 extension, days, user-gated.

## File map
- **(new 07-21)** Layout-free primitive: `src/native_edit_teacher.py:build_layout_free_edit_condition`
  (NOT K/V — position-scramble of `edit_image_ids` only; parked, niche). Klein baseline:
  `diagnostic_kleinedit_baseline.py`, `diagnostic_kleinedit_sysprompt_baseline.py`. Evaluation:
  `scripts/evaluate_matched_teacher_outputs.py` (manifest cols: method, output_path, style_path,
  prompt, style_id, prompt_id, seed). Metric libs: CSD = `/home/ubuntu/chengyanli/items/CSD_Score`,
  CLIP-T = `/home/ubuntu/chengyanli/hfmodels/clip`. Figures on R2 — see the 07-21 session block above.
- Method: `scripts/run_tpami_with_style_teacher.py` (ca_only path; env-gated `--dm-schedule`,
  `--ca-renoise-tau-frac` added this session — both negative/optional). Loss: `src/style_objectives.py`.
- Benchmark: `stylebench/` (user's StyleBench-CoT, 8-dim CoT VLM judge), `scripts/run_stylebench.py`,
  `scripts/vlm_judge_calibration.py`, `results/stylebench/`.
- Key result PNGs: `results/caprobe_ca_only_*` (CA win), `results/b2dmd_*` (DM harmful),
  `results/dmsched_*` (scheduled-DM null), `results/bench40_score_match_*` (score_match baselines).
- Docs: `docs/research/PAPER_PITCH.md`, `docs/research/LOSS_DECOMPOSITION.md` (WHAT'S TRUE NOW),
  `docs/research/VLM_JUDGE_FINDINGS.md`, `docs/LOOP_PLAN.md`, `experiment-log/2026-07/log.jsonl`.

## Gotchas
- **`setsid nohup ... &` silently fails in this Bash tool shell.** Use the Bash tool's
  `run_in_background: true` for long jobs (worked reliably for training/eval).
- GPU shares with **external jobs — never kill them**; headroom-gate launches (NEED ~64GB for
  score/CA modes; ca_dmd loads an extra fake DiT).
- The student trainer (`train_score_residual_student.py`) reloads all dumps per epoch (I/O-bound) →
  keep epochs small for smokes; use `--model-type token_mlp` (grid-agnostic; token_mixer needs the
  32×32 latent grid, not 512).
