# Local-edit is the MECHANISM, not a bug — code-grounded analysis (2026-07-20)

> **NOMENCLATURE (2026-07-21, user-decided): do NOT call this "local-edit" in the paper.**
> The edit-mode capability is named **Semantic Style Editing (语义风格编辑)**. Its headline
> property is **mask-free** — subject preservation is an *emergent* property of the method, with
> NO mask/region/bbox/segmentation/blend anywhere (grep-clean, see "Not a bug" below). Frame the
> method as one training-free **Controllable Style Locality** axis spanning
> **Semantic Style Editing (subject preserved) ↔ Global Style Transfer (whole-image)**.

---
## ⚠️ RETRACTION (2026-07-21) — read this first
The "CONVERGED METHOD … substantially fixed" section below (and any "~85% clean" figure)
is **RETRACTED as self-deception**. Honest re-audit after adversarial review:
1. **Method does NOT solve local-edit where it occurs.** The overnight "wins" were
   cherry-picked twice: (a) sty04 — a **permanently-excluded pathological scene image**
   (see attention_20260426 CLAUDE.md) whose "success" was actually **content leakage**
   (output copied the reference's musicians, not the prompt); (b) styles 00/25/12 —
   **graphic/medium styles (pop-art/scratchboard/neon) that never had a local-edit problem**;
   baseline≈method there, so my delta ≈ 0.
2. **On the honest hard set** (photoreal-subject + concrete texture: mosaic/crystalline/
   stained-glass), the method **leaks (sty04), stalls (crystalline sty16), or is marginal**.
   Contribution toward "subject genuinely takes the reference texture" ≈ **0-for-hard-set**.
3. **This night is a BOUNDARY result, not a fix.** Faithful texture-on-subject with content
   preserved needs the reference's *spatial structure* = attention **K/V = reserved first-gen
   (taboo)**. Every latent-space route (Sinkhorn / content_only / SW / Gram / VGG /
   styled-denoise) cannot cross that boundary. `content_pres=full` and "subject stylized" are
   **opposites** — removing local-edit spends the headline.
4. **The one real, small keeper:** global-rms-norm crushes a coherent subject signal
   (cosine 0.92–0.99) down to a 4–16% step share — a legit **ablation/knob**, not a fix.

**Open decision (user's, not self-cleared):** (A) keep the complete TSD paper, frame
local-edit honestly as *content-preservation* [safest, submittable]; (B) adopt styled-denoise
as a **different, weaker-thesis** method; (C) unlock K/V [only true route, taboo]. Do NOT
declare the autonomous "good style transfer" goal *met* on the non-problem graphic styles.
---

## TL;DR
TSD's "local-edit" look (style fills the background; the prompt-named subject stays
photoreal) is **not a bug**. No mask, region, blend, warm-start, or anchor-freeze localizes
it (audit grep-clean). It is the faithful, **over-determined** consequence of one root:

> the subject's pixels are jointly pinned to the frozen model's photoreal prior by the
> **content prompt**, and style is injected only as a **same-prompt, same-state contrast**
> that is ≈0 exactly where that pinning is strongest — the subject.

The keystone localizer is the reverse (denoise) process, which is the **same mechanism** we
credit as the C3 "free shield." So de-localizing is not a fix — it removes the shield and the
`content_pres=full` claim with it. **Full-subject stylization is a different method, not a
patch to this one.**

## What actually runs (verified, line refs)
1. **Main denoise, every outer step** (`run_tpami_with_style_teacher.py:1671-1683`):
   `noise_pred = main_dit(latents, encoder_hidden_states=main_prompt_embeds, <NO edit_condition>)`
   then `latents = scheduler.step(noise_pred, t, latents)`. → the whole latent is advanced
   toward the **no-style, content-prompt** prediction (photoreal). 25 steps.
2. **Then** inner_optimize nudges that latent toward the CA target. CA signal
   (`native_edit_teacher.py:145-214`): `g = v_with − v_without`; in the **default**
   `full_style_prompt` mode BOTH branches use the same prompt
   (`--prompt-template:175` = `"Create {content}. Use the visual style of the reference image…"`),
   differing ONLY by `edit_condition` (the style-ref image). → **g = pure "reference provided
   vs not" effect.**
3. Step applied as `x0 ← x0 + s·rms_normalize(g)`; `rms_normalize_field`
   (`flux2_score.py:181-185`) divides by ONE **global** scalar → per-token magnitudes of g are
   **preserved**. Each token moves ∝ its own |g|.
4. `content_anchor` (`run:1724`) re-anchored each outer step to the current (post-denoise)
   latent; uniform L1; weight `content_weight=0.2 × decay 1.0→0.5` (`run:145-146, 636`).

## Why local-edit — one root, ≥4 manifestations
- **(1) CA signal ≈0 on the subject** [forward]: adding the reference image barely changes the
  prompt-defined subject → g small on subject, large on the underspecified background. The
  global rms-norm (step 3) preserves that spatial concentration, so the update follows it.
- **(2) KEYSTONE — the reverse process re-photorealizes the subject every step** [between
  steps]: the denoise (`:1683`) predicts velocity from the **no-style** content prompt → pulls
  the whole latent toward photoreal; on the subject that pull is strong (the prompt defines it)
  with no counteracting style push, on the background it is weak. **Whatever style ANY signal
  source pushes onto the subject in inner_optimize, the next denoise predicts a correction back
  toward photoreal-subject.** This is why single-point fixes (1/3) are defeated regardless of
  tuning.
- **(3) content_anchor re-locks** the re-photorealized subject each step [regularizer] —
  amplifies, does not cause.
- **(4) edit-conditioning delivery is edit-biased** [conditioning channel]: style enters via
  klein's native edit-concat (Kontext-style: change context, keep subject).

## The keystone is welded to our headline (the real cost)
Mechanism (2) is the **same `scheduler.step`** we credit as C3's "free shield":
- **Shield framing:** contracts to the manifold → no trained DM/fake needed (why cfg20 doesn't
  collapse).
- **Localizer framing:** contracts to the **content-prompt** manifold → re-photorealizes the
  subject.

One call, two consequences. Therefore **removing the localization removes the free shield.**
The fork is deeper than "lose `content_pres=full`": full-subject stylization costs BOTH
`content_pres=full` AND the C3 contribution, because they are the same mechanism.
⇒ **Full-subject stylization = a different method, not a fix.**
(Writing note: the paper must not both praise and blame the reverse process without naming this
coupling explicitly, or a reviewer will read a contradiction.)

## Is there a non-destructive lever? Exactly one — and (2) still fights it
- Sinkhorn attention / per-token norm / lower anchor: each either finds **no signal** on the
  subject (1) or **imports reference structure** (content collapse — the kv-match failure mode);
  i.e. amplifies noise, not a hidden signal. All are undone by (2).
- The ONE lever that injects a **real, non-noise, non-structure-import** subject signal:
  **`--without-prompt-mode content_only`** — the without branch drops to `"Create {content}."`
  (`--content-prompt-template:183`) while the with branch keeps `"…Use the visual style of the
  reference image."` So `g` now also carries the **text** style instruction, which is GLOBAL
  conditioning that reaches the subject (unlike edit_condition, which mostly moves the
  background) = stylize-by-text. **But even with a subject signal, keystone (2) pulls it back
  every step;** whether the CA push overcomes the no-style denoise pull is a **magnitude**
  question, untested. This is the single test worth running — not a menu of prongs.

### ⚠️ RETRACTED (2026-07-21) — this "content_only reversal" was also self-deception
> User verdict: content_only is "**多视角的一个偏移**" (merely a multi-view offset), NOT genuine
> subject stylization. Same self-deception class as the CONVERGED section: a different-looking
> output (generic global rendering shift) read as a mechanism win. It does NOT put the reference's
> real texture on the subject. Kept below only as a record of what was tried. Same category we must
> reject for v-shuffle (which also yields generic/global stylization by destroying reference structure).

### UPDATE (2026-07-20, TESTED — content_only partially works; earlier "futile" was too strong)
Ran `--without-prompt-mode content_only` on style04 (cat) + style16 (old man) vs baseline
(`results/skprobe_conly_*`, `agent_runs/contentonly_montage.png`). **Result: the subject IS
stylized** — cat → sculptural/mosaic, old man → digital-painting — with **content/composition
preserved and no collapse** (opposite of Sinkhorn). So the "no non-destructive lever works"
claim was **overstated**; content_only is a real de-localization knob. Mechanism confirmed: the
default `full_style_prompt` puts "use the reference style" in BOTH branches, so it **cancels out
of `g = v_with − v_without`**, leaving only the (background-biased) edit-image effect; content_only
keeps the global text-style signal in `g`, so it reaches the subject. This is the closest thing to
the "bug" the user insisted on — not a crash, a **signal-construction** issue, one-flag fix.
**Limits still hold:** it is *partial* — the subject gets the overall rendering style/palette, not
the full texture, because keystone (2) still caps it; and it softens `content_pres=full`
(fix-method). Next (untun): content_only breadth (6–8 styles) + StyleBench to quantify
subject-stylization↑ vs content_pres↓.

## Not a bug (audit, grep-clean)
- No `mask/region/bbox/blend/paste/torch.where` anywhere (only the Sinkhorn probe's own
  `attn_mask`).
- `warm_start_latents` is **identity** under `--disable-slt --adain-alpha 0` (`run:589-618`).
- `content_anchor` is a per-step re-anchor, uniform, modest weight — **not** a subject freeze.
- `rms` and `l2` normalize modes are **both global** (`flux2_score.py:174-185`) — no per-token
  option exists.

## MEASURED (2026-07-20): the subject signal is REAL; the global rms-norm crushes it
Instrumented dump of `g = v_with − v_without` per inner-iter on style16 (`scripts/analyze_g_dump.py`,
`agent_runs/g_analysis_gprobe_style16_kp4.png`):
- **Subject-region `g` is highly coherent across inner iters (cosine 0.92–0.99)** = a *real* style
  signal, NOT noise. (My earlier "no signal on the subject" was wrong.)
- It is just **3–12× weaker in magnitude** than the background.
- `rms_normalize_field` divides by ONE global scalar set by the (bright) background, so the subject's
  real signal is normalized down to a **4–16% share of the total step** → subject barely moves →
  local-edit. **This is a genuine loss-normalization defect** ("loss选取"), independent of keystone (2).

**Fix implemented** (`src/flux2_score.py`): `rms_normalize_field_pertoken` (each token stepped along
its own coherent direction) + `blend[mix]`; flag `--pseudo-grad-normalization rms_token|blend0.5`.
**Result (style16 old man / style04 cat, `agent_runs/normfix_montage.png`):** per-token *does* stylize
the subject more (clear on the cat → sculptural; subtle on the strongly-prompt-pinned old-man face) —
**partial de-localization, capped by keystone (2)** (the no-style reverse denoise re-photorealizes the
subject each step). Confirms both: the crushed-signal diagnosis is correct, AND #2 is the remaining cap.

**Paper-asset framing (the off-ramp):** the per-token↔global norm is a clean **knob/ablation** — it
interpolates *background-concentrated ↔ more-global* stylization, with `content_pres` tunable against
it. Strengthens the mechanism story, needs no new loss, no K/V, keeps the headline. Recommended to bank.

## Why "a better loss" is a different paper (not a fix)
CA `= (α−1)(v_with − v_without)` is **CFG on the edit condition** — a score-difference, first-order
"move toward the conditioned sample" signal; behaving like klein-native edit is *structural*, confirmed
by the g-measurement. Real *texture transfer* needs a **feature-/patch-statistics matching** objective
(Gram / sliced-Wasserstein / entropic-OT), a different kind of object. Two hard consequences:
1. **K/V collision (project taboo #1):** texture-statistics matching is the conceptual neighborhood of
   the unpublished first-gen K/V method. A Gram/OT loss on *attention* features = that method in a hat.
   Only statistics on **decoded pixels (VGG/Gatys)** or on the **x0/latent** are clearly distinct.
   Needs an explicit user ruling before any code.
2. **It inverts this paper:** kills the "a single native CFG-aug signal suffices" thesis, guts the
   negatives study, trades away `content_pres=full`. That is a *new-method paper*, multi-week, against
   the ~07-28 deadline — vs the current paper assessed complete on existing material + E1.

## ⚠️ RETRACTED (see top) — CONVERGED METHOD (overnight R1–R6, 2026-07-20): styled-denoise + per-token
> RETRACTED 2026-07-21: "substantially fixed" is false — evidence was cherry-picked (excluded
> sty04 + graphic styles with no local-edit to fix). Kept only as a record of what was tried.
> The styled-denoise config below is real code and a *candidate* for option (B), NOT a solved fix.

**Result:** the local-edit failure mode is substantially fixed by attacking the *mechanism*, not by
swapping the loss. Winning config:
```
STYLED_DENOISE=4  --pseudo-grad-normalization rms_token   (optionally STATS_LOSS=gram STATS_WEIGHT=10)
--style-mode ca_only --cfg-scale 7 --lambda-ca 1 (rest = standard TSD)
```
- **styled-denoise** (`STYLED_DENOISE=g`, in `run_tpami:~1671`): main reverse step becomes
  `noise_pred = v_without + g·(v_with − v_without)` (CFG on the native edit condition IN the denoise)
  → the trajectory itself carries style, so it stops re-photorealizing the subject each step (fixes
  keystone #2). This is the core of the win.
- **per-token norm** (`rms_normalize_field_pertoken`): surfaces the real-but-crushed subject CA signal.
- **+gram (optional)**: adds subject texture; style-dependent knob (mosaic→ceramic ✓, crystalline→
  generic oil-painting ✗).
- Strongest evidence: style04 cat → glazed ceramic figure integrated into the mosaic (real transfer,
  content preserved). style16 old man integrates into crystalline scene, specific style kept.

**What did NOT work (ruled out):** replacing CA with a latent statistics loss — SW/Gram on x0/latent
give only palette/generic texture + content drift (R1/R2); Sinkhorn attention collapses; content_only
= still local-edit; aggressive per-token push = content collapse.

**Scope (breadth R6, honest):** targets the *photoreal-subject-in-textured-style* failure mode
(mosaic/crystalline) — clear help there; neutral on already-global graphic styles (woodcut/comic, no
local-edit to fix); can drift content on some styles at high guidance (lower `g`).

**Wall:** exact reproduction of the reference's *structured* texture on the subject (facets/tiles)
needs importing its spatial structure = attention K/V = reserved first-gen (taboo) + content collapse.
So this is "cohesive/integrative style transfer," not "subject = the reference's exact texture."

**Artifacts:** report https://yata-image-publics.flowgpt.com/felix/style-transfer-report-v3.html ·
`src/style_stats.py` · `agent_runs/{stats_r*_montage,r*_*,g_analysis_*}.png` · algo-log R1–R6.

## Status / decision
- Sinkhorn probe queued (headroom-gated) as **one confirmatory negative**. No further prongs
  added — this analysis predicts they are futile or destructive.
- The decision is the user's and is now sharper: **`content_pres=full` AND the free-shield
  contribution are both local-edit.** Fixing one spends both. The honest alternative
  (evaluate subject-stylization separately, keep the win) needs no method change.
