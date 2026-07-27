# Loss Decomposition Research Record

> **Status update (2026-07-23):** mechanism-development history. Current paper
> wording and bounded claims live in `AAAI/main.tex`, `AAAI/supplement.tex`, and
> `docs/research/PAPER_PITCH.md`. The reverse process is an empirical implicit
> regularizer, not an exact distribution-matching theorem; the direct
> reverse-shield ablation remains pending.

Original record started 2026-07-20.

## ══════ WHAT'S TRUE NOW (consolidated 2026-07-20, extended 2026-07-21; read THIS, sections below are the trail) ══════
THE WIN — the "改loss" that works = **CA driver: `ca_only` at strong α, SAME-STATE (τ=t)**.
CA = (α−1)(v_with − v_without) applied as an rms-normalized pseudo-guidance STEP in x0-space
(build_pseudo_target_loss), every inner iter. Transfers texture score_match misses (mosaic
tesserae, crystalline facets) while PRESERVING content; no K/V, no fake net, no re-noise.

MECHANISM (code-verified, advisor): **extrapolation-walk vs interpolation-to-fixed-point.**
- score_match minimizes ‖N[Δs]‖² → a DIMINISHING pull toward the fixed point v_without=v_with,
  which for hard styles (weak edit-conditioned texture) is a WEAKLY-textured target → it
  interpolates to it and SATURATES before texture emerges.
- ca_only applies a FIXED-magnitude step along (v_with−v_without) regardless of convergence →
  walks PAST the fixed point → EXTRAPOLATION (CFG α>1 = extrapolation beyond the conditional);
  iterated, it accumulates style the single styled prediction only weakly contains.
- α is INSIDE the direction and build_pseudo_target_loss rms-normalizes it away ⇒ **α washed out**;
  EVIDENCE = the flat α-sweep cfg4≈7≈10. (This also explains the retracted "ceiling": x̂0^s was
  the UN-extrapolated single prediction, so of course it looked textureless.)

WHY THE OTHERS FAIL:
- score_match: interpolation-to-weak-fixed-point (above).
- re-noise τ<t (CDM L_CA, tested): NULL. Not magnitude (normalized) — the DIRECTION degrades:
  at low σ the image is determined, so (v_with−v_without) points at fine local content
  corrections, not the global style-transport direction. Same-state τ=t is the operative sweet
  spot. (Don't chase τ>t.)
- kv_match: gets texture but COLLAPSES content (matches raw attention output → imports the
  reference's structure/layout). = first-gen mechanism, OFF-LIMITS (reserved for a stronger venue).
- carrier / score_carrier: multiplicative reweight of the residual, inert (5 signals).

QUANT (StyleBench, flash judge — verified it DISCRIMINATES here, not just lenient):
- style_capture partial→FULL and fidelity 2→4/5 on hard styles (04,16,14); marginal where
  score_match already full (21: 4→4). renoise worse than CA (04:3, 16:2). Content-preservation
  axis (quant2): CA_cfg7 = style_capture FULL + content_pres FULL on all 3 hard styles (04/16/14);
  score_match = partial style. => texture-GAINED AND content-PRESERVED (the anti-K/V-leak claim, quantified).

DM / fake-score (user asked): the literal DM shield needs a TRAINED fake net (all 7 papers; no
frozen genuine-DM). BUT DM-shield exists to stop CA-alone collapse over TRAINING; we do 25-step
single-latent INFERENCE (no training) → that collapse may never arise. **GATE RESULT (cfg15/20 on 04/16): CLEAN — cfg7≈15≈20 near-identical, NO over-saturation/
checkerboard/collapse even at cfg20 => DM UNNECESSARY.** The collapse DM prevents is a TRAINING
phenomenon (generator-distillation); our 25-step single-latent INFERENCE never enters it. FINDING:
CA-alone suffices; the fake-net half of decoupled-DMD is not needed in the training-free regime. If it collapses → add a shield, cost order:
(1) frozen stats-KL (Decoupled-DMD Eq7, recalibrate to FLUX latents) or NFSD negative-branch
(both frozen, cheap) → (2) offline-warmed frozen critic (last resort, multi-hour, only "real" DM).

B1/B2 STUDENT SMOKES (2026-07-20): B1 (amortize CA per-step direction → token_mlp student) NULL
(delta_cosine 0.40, best inject scale=0 — the CA win is iterated extrapolation, not a one-shot
direction). B2 (ca_dmd inference, CA+DM w/ real klein4b fake critic): DM HURTS — style04 mosaic
LOST, pulled to naturalism (fake pulls toward data manifold, undoes CA extrapolation). => DM at
inference is counterproductive, not merely unnecessary. Real amortized student = Tier-2/days.

07-21 UPDATE — Controllable Style Locality axis (extends, does not replace, the above): the paper
now organizes around ONE axis with two modes: **Semantic Style Editing** (subject kept in
reference's composition, medium transferred; dino_layout_leak 0.45–0.82 = composition KEPT) vs.
**Global Style Transfer** (reference composition dissolved, medium kept; dino_layout_leak 0.37–0.46
= composition DISSOLVED).

GLOBAL TRANSFER CAMPAIGN (new E3–E6 series, 07-21 — distinct numbering from the 07-20
EXPERIMENT_PLAN's own E1/E5/E6, do not conflate) — RESOLVED: velocity-space *derived* losses ALL
fail (A scalar content-repel `d_R−b·d_C`, B projection, D multi-view marginalization) — each
collapses to semantic-edit-or-lose-medium, i.e. THE K/V WALL THIS PAPER PREDICTS, now confirmed
empirically. A taboo-clean **layout-free** primitive `v_cR_scram` (position-scramble of
`edit_image_ids`, `src/native_edit_teacher.py:build_layout_free_edit_condition` — NOT K/V, only
input positional ids) = E3 grid 3/5 works (bronze/pop-art/stained-glass: dissolved composition +
kept medium + subject present) but 2/5 destroys the subject (mosaic-portrait/oil-portrait) —
PARKED as a niche potential, not adopted. The WORKING Global-Transfer lever = the **step-knob**:
`ca_only` with FEWER denoise steps (E4: usable band num-steps 8–10; `iters` does NOT substitute —
it couples to structure, doesn't break the dissolve↔sharpness tradeoff).

MECHANISM FACT SHARPENED (updates, does not contradict, the α-washed-out finding above): `pseudo_
grad = lambda_ca·(cfg_scale−1)·delta_ca_score` is then rms-normalized in `build_pseudo_target_loss`
(default `--pseudo-grad-normalization rms`) → **cfg_scale AND lambda_ca have ZERO effect on
ca_only output** (not just a flat sweep — algebraically washed out). Live tunable knobs: num-steps
/ iters × pseudo-step-size × normalize-mode {rms, rms_token}. Recommended: Global = num-steps 8,
step-size 0.12 (E5 best CSD≈0.63 rms/0.64 rms_token vs s08-anchor 0.53, layout_leak still ~0.45).
Semantic = iters 5, step-size 0.08, rms, 25 steps unchanged (E6 balanced pick CSD 0.74/CLIP-T 0.28
vs anchor q25s CSD 0.70/CLIP-T 0.285; the CSD-max cell i8/s08/rms_token over-stylizes: CSD 0.82 but
CLIP-T crashes to 0.21, layout_leak 0.82 — do not use it).

BASELINE COMPARISON ADDED (klein's own native `Flux2ImagePipeline` edit, cfg_scale=4/25 steps —
the honest "what happens without TSD," not a competitor table): CSD klein-strong sem 0.36 / global
0.29 vs OURS 0.70 / 0.53 — ~2× win, 5/5 styles; klein CLIP-T marginally higher (0.31 vs 0.29) =
honest subject↔style tradeoff; klein's tell = reverts to generic engraving/lineart, only nails
relief media (bronze09).

NOVEL METRIC PROPOSED, PENDING: **SRSD** (Score-Residual Style Distance) = ‖v_cR −
v_c0(full_style_prompt)‖ on the generated image, single-shot generator-native, no external network.
SRSD↔CSD validation DONE (2026-07-21) → **REJECTED**: correlation is POSITIVE not negative (Pearson +0.79, Spearman +0.88), driven by over-stylized content-collapse samples (CSD rewards collapse, SRSD spikes there); within-config residual r=0.20 n.s.; scale-corrected unchanged. Spear magnitude marks content-collapse/composition-copying, NOT style faithfulness — do NOT use, do NOT flip sign. Kept only as an honest null in the paper (§4). Keep the 3 external axes (CSD/CLIP-T/dino_layout_leak).
## ════════════════════════════════════════════════════════════════════════════════════════════



## ★ "Decoupled DMD" IS A REAL PAPER (arXiv 2511.22677) — maps to our ca_only/ca_dmd
"Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield".
- CA (CFG-Augmentation) = (α−1)(s_cond − s_uncond) = the DRIVER ("spear"), **frozen-teacher
  only, NO fake model** (resolves my earlier "fake-score ill-posed" blocker!).
- DM = s_cond − s_fake = REGULARIZER ("shield") vs artifacts/oversaturation; needs fake.
- Finding: CA ALONE ≈ full DMD; DM only stabilizes. + decoupled RE-NOISING: τ_CA > t
  (CA evaluated at HIGHER noise than current step), τ_DM ∈ [0,1] global.
OUR CODE ALREADY HAS THIS: pseudo_guidance_from_velocities → ca_only computes
`lambda_ca*(cfg_scale−1)*delta_ca_score` (= CA term, α=cfg_scale), ca_dmd adds
`lambda_dm*(v_with−v_fake)` (= DM term). cfg_scale default 3.0 (so CA≠0). What we DON'T
have = the decoupled RE-NOISING schedule τ_CA > t (our ca_only is same-state τ=t).
=> user directive ("参考各类 decoupled-dmd loss 更新的尝试"): (a) probe ca_only at strong
α (caprobe cfg=7 on style04/16, running), (b) subagent surveying DMD-loss-update family
(SiD/DMD2/R_dm/DMDR/CDM + re-noising) for the best adaptable form. Small-sample eyeball
gate per user before rollout. Config leads eliminated: guidance_embeds=false (guidance=0
correct), auto_resize=True does NOT add texture (small-sample). So the lever is the LOSS.


User directive: don't bolt on a Gram loss / don't tune weights empirically — DERIVE
extra loss terms by decomposing the objective, borrowing **decoupled DMD**'s method.
Frequency decomposition REJECTED (user: "大概率不行"; correct — reweighting can't
conjure texture information not present in term A).

## The decomposition (user-confirmed)
Same-state predictions: with-style velocity v_s, without-style v_0. In x0-space
(s = -(z-(1-σ)x̂0)/σ²) every score-difference is a prediction-difference. DMD moves a
sample toward target distribution p_s via real−fake score. Split s_s via s_0:

    s_s - s_fake  =  (s_s - s_0)   +   (s_0 - s_fake)
    [full DMD]       (A) OUR CURRENT g   (B) real−fake distribution match  ← MISSING

>>> ⚠️ CORRECTION (2026-07-20): the telescoping below is NOT decoupled-DMD's decomposition.
>>> Decoupled-DMD (arXiv 2511.22677 Eq.6, re-verified) is an ADDITIVE sum of two terms, both
>>> anchored on s_cond:  ∇ ∝ (s_cond − s_fake)[DM shield, needs fake] + (α−1)(s_cond − s_uncond)[CA spear].
>>> NOT (s_s−s_0)+(s_0−s_fake). See the "★ Decoupled DMD IS A REAL PAPER" block above (lines ~56-58)
>>> for the correct form. TSD = keep CA = (α−1)(v_+−v_-), drop DM (reverse process = free shield).
>>> The point-mass exploration below was superseded (advisor: vacuous). Trail only.

- (A) = s_s - s_0 = our score residual = a same-state CONDITIONAL contrast (CFG-type).
  In x0-space ∝ (x̂0^s - x̂0^0): styled vs STATIC content prediction.
- (B) = s_0 - s_fake, needs a fake score of the CURRENT output distribution. We have
  NONE of it. score_carrier/ca_only are all still term-A variants → why they're inert.

## Why (B), not frequency, is the right lever
(A) is two frozen-teacher conditionals on the SAME current latent; once palette matches,
their difference carries little texture (shared structure cancels). (B) pulls toward the
REAL data manifold (textured) relative to where the current (textureless) output sits —
it injects NEW information. Frequency only reweights (A)'s output.

## CRITICAL implementation correctness (advisor)
Do NOT implement as "A + β·B" with tuned β. Term B ALONE = s_0 - s_fake ∝ (x̂0^0 - x̂0^cur)
is a spring toward CONTENT — it FIGHTS style. Only the COMBINED form is right:
    full = s_s - s_fake  ∝ (x̂0^s - x̂0^cur)   [styled vs CURRENT, x̂0^cur detached]
i.e. the point-mass fake just SWAPS the static content baseline x̂0^0 for the moving
current prediction x̂0^cur. One change, training-free, NOT collinear with A.

## ⚠️ CEILING CONCLUSION RETRACTED (2026-07-20, user was right)
The "frozen-model ceiling" conclusion below is WRONG. Empirical test: running the
first-gen's kv_match loss (self-attention K/V OT-matching, scripts kvtest) on style04
(mosaic) and style16 (crystal) BRINGS THE TEXTURE BACK — mosaic tesserae / crystalline
facets both render. So the model CAN transfer texture; x̂0^s was textureless only because
score_match's channel (velocity residual off the EDIT-conditioning token) is weak — the
texture is NOT in the edit-path velocity. Evidence: scratchpad/kv_vs_sm.png.
=> The failure is a LOSS-CHANNEL problem, not a model ceiling. GATE 0 result stands only
as "texture isn't in the EDIT-PATH velocity" — NOT "isn't in the model."

## KEY REFRAME: score_match & kv_match = same op, different space
Both match without-style → with-style prediction:
- score_match: in VELOCITY (edit-conditioning) space → content-safe, texture-blind.
- kv_match: in SELF-ATTENTION-OUTPUT space → texture-rich (texture lives here), but
  content-LEAKY (imports the style's structure; kvtest with GRAG off = heavy layout leak).
Subagent map of first-gen (items/save): texture channel = optimize latent so deep
single-stream self-attn output matches OT/Sinkhorn attention against style K/V; content
via native-edit; GRAG strips style layout; dist(mean/std)=palette, OT-attn=brushwork.

## DERIVATION TARGET (user: "只用公式推导改loss，不开GRAG等旋钮")
Operate in the attention-feature space (where texture is) but DECOUPLE appearance
(transfer) from structure (preserve) → get kv_match's texture + score_match's content,
purely by formula, no GRAG/content-loss knobs.
CANDIDATE A (structure/appearance decoupled attention): self-attn out = A·V, A=softmax(QKᵀ/√d)
="where"(structure/content), V="what"(appearance/texture). kv_match replaces BOTH A→OT(Q_gen,K_sty),
V→V_sty ⇒ structure leaks. Derived target: KEEP gen's A_gen, restyle ONLY values via a
statistics transform Γ(V_gen; V_sty) (whitening-coloring / AdaIN of V to style's mean+cov):
  out* = A_gen · Γ(V_gen; V_sty);   L = Σ_layers ‖ self_out − sg[out*] ‖².
Style-as-statistics (Gatys/AdaIN) at the attention-VALUE level with gen's attention map
preserved ⇒ texture without layout leak. Formula-derived, differentiates from raw kv_match.
CANDIDATE B (feature-statistics matching): match permutation-invariant stats of deep self-attn
outputs: L = Σ_layers[‖μ(out_gen)−μ(out_sty)‖² + ‖σ(...)‖² + λ‖Gram(out_gen)−Gram(out_sty)‖²].
Layout-agnostic by construction (stats over tokens). But λ·Gram may still be a "bolt-on".
CANDIDATE C (decoupled-DMD in attention-output space): apply the earlier A+B DMD split to the
attention output instead of velocity; needs a "fake" attention-output — likely overkill.
=> PENDING advisor pressure-test before implementing (my last 2 derivations had flaws).

## GATE 0 — ceiling check (running, headroom-gated)
A+B = s_s - s_fake; texture can ONLY come from s_s (frozen teacher styled prediction).
So decode x̂0^s at several σ on robust-fail styles and LOOK:
  - x̂0^s textureless at operative σ → NO fake score helps → STOP the whole route.
  - texture in x̂0^s but lost in final output → term B worth building.
Instrumented run_tpami_with_style_teacher.py (env DUMP_XHAT_STEPS, dumps x̂0^s/x̂0^0
tokens mid-loop, decodes at end when VAE resident; env-gated no-op by default).
Driver scripts/run_xhat_ceiling.sh (style04/16/21, headroom-gated — external GPU
jobs present, NOT killed).

### GATE 0 RESULT — style04 (mosaic): NEGATIVE (route likely dead)
Decoded x̂0^s at σ={0.992,0.945,0.859,0.676}. The styled teacher's OWN prediction is a
smooth warm-palette cat at EVERY level — the mosaic/tesserae texture is NEVER present,
including at o06 (σ0.945, latent still ~noise, barely optimized → not a
circularity artifact of our optimization). It adopts only the golden PALETTE.
=> texture is NOT in s_s. Per the gate, NO fake score / DMD term / frequency reweight
can conjure it. The failure is a **frozen-model image-conditioning CEILING** (FLUX.2
shown a mosaic ref does not produce mosaic texture), NOT a loss/optimization problem.
Strip: scratchpad/ceil04.png.

### GATE 0 — CONFIRMED NEGATIVE across style04/16/21 (route CLOSED)
Same pattern on all three, every σ (incl. early high-noise o06):
- style04 (mosaic): x̂0^s = smooth warm cat; tesserae/gold-leaf NEVER appear.
- style16 (crystal 3D): x̂0^s = plain photographic old man on orange gradient; glassy
  crystalline faceting NEVER appears.
- style21 (3D low-poly geometric): x̂0^s = vivid digital sunflowers; low-poly angular
  facets NEVER appear.
In every case the styled teacher adopts the PALETTE/mood but not the distinctive
TEXTURE/STRUCTURE. Since A+B = s_s − s_fake and s_s is the ONLY teacher-derived target,
and the fake score only changes the BASELINE (not s_s), texture MUST come from s_s —
and it is absent. => decoupled-DMD (and any inference-time loss reweight on this
conditioning path: frequency, Gram-on-latent, carrier) CANNOT rescue these failures.
DECISION: **STOP the DMD route** (advisor pre-commit: "ceiling is zero, stop here").
Strips: scratchpad/ceil{04,16,21}.png.

### What this MEANS (the real, better finding)
The hard-style failures are a **capability ceiling of the frozen model's image-
conditioning path** — FLUX.2 [klein], shown a mosaic/crystal/low-poly ref as an image
condition, does not *encode* that texture in its styled prediction; it encodes palette.
No training-free method operating on this signal (ours, DMD, Gram, frequency) can
transcend it. Transcending it would require training / a different conditioning
mechanism / a different base model — all outside the免训练 score-residual premise
(and K/V is taboo). => the honest-scope paper is confirmed as THE paper, now with the
failure mechanism proven at the deepest level (model ceiling, filmstrip evidence), not
hand-waved as "regularization dominance." Caveat: circularity ruled out by early-step
(o06, latent≈noise) evidence; tests the score-residual conditioning path specifically.

## Fake-score arms (gated on ceiling PASS), in order
1. **Point-mass moving-baseline** (training-free, FIRST real experiment): full = x̂0^s -
   x̂0^cur (detach x̂0^cur). One code change. Advisor: clean ablation, not crude.
2. **Offline-warmed critic** (arm 3, only if arm 1 helps-but-not-enough): warm the
   ca_dmd_backend KleinFullFakeBackend OFFLINE on our 95 existing stylized outputs,
   FREEZE, use its score. Avoids the "not-validated" online alternating DMD2 loop.
   A distribution over our outputs is what makes s_fake mean something — we have it.
2b. Arm 2 (frozen teacher on re-noised current) DEPRIORITIZED: s_fake≈s_0 → B≈0, null
   by construction.

## Pre-committed success test (over-read guard)
On robust-fail styles (04,08,14,21,31,34,37): StyleBench weakest_dim moves OFF
texture/structure AND eyeball shows real texture, on BOTH seeds, WITHOUT regressing
3–4 already-working styles. One cell improving = noise. A null from the CHEAP arms is
NOT "decoupled-DMD fails" — only a null from the offline-warmed critic speaks to the route.

## Scope discipline (advisor)
Additive methods expedition, real chance of not panning out. Bounded: ceiling → point-
mass → DECISION GATE before the critic build. Honest-scope rewrite + StyleBench
integration remain the closer-to-shippable AAAI value; don't let this go open-ended.

## ★★ DMD-loss-update SURVEY (subagent, 2026-07-20) — the principled path
KEY FACT (all 7 papers): the DM term is ALWAYS s_real − s_fake with s_fake from a
TRAINED online fake net. **No frozen-teacher DM exists.** Degenerate frozen DM
(v_fake=v_without) collapses into CA → adds nothing. So "CA+DM" literally needs a
trained fake net (impractical single-latent). BUT DM is only a SHIELD (paper's own
finding: CA drives, DM stabilizes), and its role has FROZEN substitutes.
FREQUENCY SPINE: high τ resolves low freq (palette), low τ resolves texture. Our
"palette not texture" = the mid-freq band (tesselation/facet) is either never probed
(H1: wrong τ) or swamped by DC magnitude (H2). Decoupled-DMD's τ_CA>t is the WRONG
direction for texture (adds palette); we want τ_CA<t.
RANKED frozen-only loss updates (all need NO v_fake):
 Rec1★ CDM L_CA (arXiv 2605.06376, RF-native): eval CA contrast at RE-NOISED LOW τ<t
   (band) + inverse-L1 norm w_τ=1/‖v_with(z_τ)−v_without(z_τ)‖₁. Hits H1+H2. Strongest bet.
   z_τ=α_τ x̂0+σ_τ ε (REUSE same ε). g=Σ_τ w_τ(α−1)(v_with(z_τ,τ)−v_without(z_τ,τ)).
 Rec2 High-pass CA (cheapest, 0 extra passes): g=(α−1)(r−LowPass(r)), r=v_with−v_without. Attacks H2.
 Rec3 = the re-noising mechanism of Rec1 stated alone (τ_CA<t, sweep band).
 Rec4 Frozen stats-shield (Decoupled-DMD Eq.7): KL on per-image latent mean/var toward
   FLUX.2 latent stats (RECALIBRATE μ_tgt/σ_tgt) → lets us push α harder w/o artifact collapse.
   This IS the frozen substitute for the DM shield the user asked about.
 Rec5 Adaptive-norm CA (Pseudo-Huber g=(α−1)r/√(‖r‖²+c²); SiD magnitude weight; NFSD neg-branch).
OUT (need trained fake/RL/discriminator): DMD2, SiD/SiM DM, R_dm, DMDR(2511.13649), AMD.
PLAN: FFT diagnostic (H1 vs H2) → implement Rec1 (CDM L_CA low-τ + inv-L1 norm) → +Rec4 shield.
Current win (ca_only cfg7): CA driver dramatic on mosaic/crystal, no regression, marginal on
graphic styles → Rec1 should extract texture more uniformly (the marginal cases = mid-freq unprobed).
