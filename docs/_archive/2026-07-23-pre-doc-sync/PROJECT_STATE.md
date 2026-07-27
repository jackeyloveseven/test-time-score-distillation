> ⚠️ **STALE — CURRENT-TRUTH POINTER (2026-07-21):** this file is dated 2026-07-14 and predates
> both the TSD pivot (07-20) and the Controllable Style Locality campaign (07-21); its "Current
> Thesis" / "Main Method" (`score_carrier`/GoalCarrier) below is **SUPERSEDED**. Authoritative
> entry point: `docs/HANDOFF.md` (+ `docs/research/PAPER_PITCH.md`,
> `docs/research/LOSS_DECOMPOSITION.md` top "WHAT'S TRUE NOW"). One-line current status: method =
> **Test-Time Score Distillation** (`ca_only` CFG-augmentation extrapolation, no weights trained),
> organized around ONE axis — **Controllable Style Locality**: Semantic Style Editing (composition
> kept, medium transferred) ↔ Global Style Transfer (composition dissolved via a "step-knob" =
> fewer denoise steps, medium kept). A baseline comparison vs. klein's own native edit was ADDED
> this session (we win ~2× CSD, 5/5 styles). Recommended params: global = num-steps 8/step-size
> 0.12; semantic = iters 5/step-size 0.08/rms. A novel metric SRSD is PROPOSED (validation
> pending). Everything below (score_carrier/GoalCarrier as "Main Method", the K/V comparison-table
> language) is historical trail only — kept for record, not current.

---

# Project State

Status date: 2026-07-14.

This file is the shortest authoritative description of the project. If an old
plan, result log, or handoff conflicts with this file, use this file, current
source code, and `AAAI/main.tex`.
**(SUPERSEDED 07-21 — see pointer block at the top of this file; use `docs/HANDOFF.md` instead.)**

## Current Thesis

The paper studies reference-based style transfer as a goal--carrier--flow
problem. The native image condition exposes a useful style direction, but that
direction needs content-aligned spatial support before it is executed by the
rectified-flow sampler.

## Main Method (SUPERSEDED 07-21 — current main method = TSD `ca_only`; see `docs/HANDOFF.md`)

The paper-facing method is `score_carrier`, named **GoalCarrier**:

1. evaluate native with-reference and without-reference branches on the same
   latent, timestep, prompt metadata, and image-token IDs;
2. convert their velocity difference to a score difference after analytically
   removing the shared state term;
3. construct a detached token affinity from the current content state only;
4. transport the native goal through that affinity;
5. optimize the current latent through differentiable base clean-state
   prediction and a detached carrier pseudo-target.

No global network parameters are trained during the main method.

## Current Paper

Title:

> From Visual Goal to Denoising Flow: Carrier-Anchored Image CFG for
> Training-Free Style Transfer

The working PDF is seven technical pages, two reference pages, and one AAAI
reproducibility-checklist page. The supplement is three pages. Paper compilation
uses result images available on `ubuntu-107`; the Mac is documentation-only.

The prose is conference-level, but broad empirical readiness is not complete.
The working preamble is also not the official AAAI author-kit style.

## What Is Validated

- Score/velocity conversion and detach boundaries are implemented and tested.
- The strict representative GoalCarrier rerun reproduces CSD `0.689619`.
- On the representative historical comparison, K/V, score match, and
  GoalCarrier obtain CSD `0.5614`, `0.6770`, and `0.6882`.
- On three matched pairs, GoalCarrier improves mean CSD from `0.4027` to
  `0.5662` over K/V and keeps CLIP-T approximately unchanged.
- The three-pair result does not dominate Gram distance or layout leakage.
- AdamW is the default latent optimizer; zero decay is the formal setting.
- A real-image VAE code can condition an optional field student, but held-out
  style evidence is nearly tied with style-ID conditioning.
- Local noisy-state CSD reward, including GRPO and a same-state local oracle,
  does not beat the formal GoalCarrier CSD baseline.
- Regression verification on 2026-07-14: `53 tests, OK`.
- Qualitative held-out generalization (2026-07-19): GoalCarrier on five unseen
  styles (05, 11, 20, 31, 38) with a held-out prompt (cat on a windowsill),
  seed 64, SLT and AdaIN disabled -- 5/5 sharp and non-blurry, 4/5 with clear
  style fidelity (pen/ink engraving, black-yellow comic, woodcut, low-poly);
  only the pixel/voxel style under-transferred its discrete texture while still
  staying clean. Artifacts: `results/qual5_score_carrier_style{05,11,20,31,38}_kp2_seed64.png`.

## What Is Not Established

- broad superiority over many styles, prompts, and seeds;
- human preference gains;
- a complete runtime/memory comparison;
- a winning persistent CA+DMD result;
- a particular transformer layer where style first appears;
- universal generalization of the optional image-code student;
- equivalence between intermediate-image reward and final-image best-of-N.

## Locked Decisions

- Main method: `score_carrier` / GoalCarrier. **(SUPERSEDED 07-21 — current main method = TSD
  `ca_only`; see `docs/HANDOFF.md`.)**
- `ca_only`: online ablation, not the paper headline.
- `ca_dmd`: extension requiring a real persistent fake-score backend.
- `kv_match`: baseline, never the native teacher.
- Adapter/student: optional amortization only.
- Native branches must be same-state.
- Carrier affinity must not receive reference tensors.
- Original Hyperion directory is read-only.
- Paper prose lives in `AAAI/main.tex`; Markdown paper notes are summaries.

## Gen-2 Positioning (2026-07-19, user directive)

This is the second-generation style-transfer paper. It is presented as a
self-contained goal--carrier--flow contribution and leads with differentiation.

- Do NOT frame GoalCarrier as an increment over the first-generation
  (sec / Hyperion TPAMI) method, and do NOT add a head-to-head benchmark
  against it. The paper already contains no such comparison; keep it that way.
- SLT and AdaIN warm-start are first-generation carryovers and are NOT part of
  the method. The formal setting already disables them
  (`--disable-slt --adain-alpha 0 --disable-final-adain`); the 2026-07-19
  five-style qualitative check ran in exactly this configuration and stayed
  sharp, so removing them costs nothing on quality. Strip SLT/AdaIN lineage
  framing from the paper and code surface; do not present the method as
  "first-generation minus warm-start."
- LRS does not exist in the gen-2 code or paper; it was a first-generation
  component and was never imported here. "Removing LRS" is a no-op for gen-2.
- OPEN (user Q1, asked 2026-07-19, awaiting answer): the single quantitative
  baseline `kv_match` (K/V replacement) is essentially the first-generation
  attention-replacement mechanism. Decision pending on whether to (a) keep it
  but reframe it as generic training-free prior art (StyleAligned /
  Cross-Image-Attention / InstantStyle family), or (b) drop the head-to-head
  entirely for a pure-differentiation paper. Until resolved, do NOT alter the
  K/V evidence table or the method-framing prose that depends on it.

## Next Gates

1. Complete matched multi-seed K/V, score match, GoalCarrier, and CA+DMD.
2. Run carrier identity, shuffled, uniform, and reference-affinity interventions.
3. Add blinded human style/content preference.
4. Report wall time, forward/backward counts, and peak memory.
5. Add a larger license-documented reference set.
6. Switch to official AAAI style and rerun page/render/font checks.
