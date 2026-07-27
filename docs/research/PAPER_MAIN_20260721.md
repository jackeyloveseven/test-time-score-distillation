# Historical Markdown Paper Draft - 2026-07-21

> **Status update (2026-07-23):** historical snapshot only. The canonical title,
> wording, results, figures, and limitations are in `AAAI/main.tex` and
> `AAAI/supplement.tex`. Do not port TODOs or claims from this file without
> checking the current LaTeX sources and `docs/PROJECT_STATE.md`.

## Original Draft Title

Controllable Style Locality: One Frozen-Model Score Spear for Semantic Style Editing and Global Style Transfer

> Draft v1 — 2026-07-21. Written in the decoupled-DMD exposition style (principled derivation → one operator → honest ablation). Numbers marked ⟨bench⟩ are filled from the full 40-style benchmark (`benchmark_full_summary.csv`); the 5-style param-study numbers are final. Taboos honored: no K/V matching (reserved first-gen baseline; never a mechanism, never a comparison); no backbone model named in title/abstract (architecture-agnostic framing).

---

## Abstract

Training-free style transfer with a frozen text-to-image transformer is usually framed as a single behavior: paint the prompt's subject in a reference image's style. We show it is really a **one-dimensional controllable axis** — *style locality* — whose two ends are distinct, useful tasks: **Semantic Style Editing**, where the subject is rendered in the reference's medium *and* keeps the reference's composition (mask-free), and **Global Style Transfer**, where the reference's composition is dissolved while its medium and palette are kept. We reach both ends with a *single* frozen-model operator. Starting from the affine map between rectified-flow velocity and conditional score, we recast style guidance as the **CFG-augmentation "spear"** of decoupled distribution matching, $(\alpha-1)(v_{\text{with}}-v_{\text{without}})$, applied as a fixed-magnitude, rms-normalized pseudo-target step on the latent. Our central claim is a **regularization substitution**: in the single-sample, test-time, multi-step reverse-process regime, the frozen reverse process *is* the distribution-matching regularizer, so the trained fake-score "shield" of distribution-matching distillation is both **unnecessary** (the spear alone never collapses, even at large guidance) and **harmful** (a trained critic pulls samples back to the natural-image manifold and erases structured media). Dropping the shield and keeping only the spear yields **Test-time Score Distillation (TSD)**. Locality is then a free knob: the full spear gives semantic editing; fewer reverse steps give global transfer. We benchmark with a **three separable-axis protocol** — CSD (style fidelity), CLIP-T (subject fidelity), and a layout-leak measure (composition transfer) — and show that any single style number is misleading: pushing style fidelity alone past a point silently destroys the subject, which only the joint CSD–CLIP-T reading exposes. Against directly instructing the same frozen model to edit the image (its native image+text edit, given its strongest style-transfer prompt), TSD delivers **2.5–2.7× the CSD style fidelity** of the native edit at matched subject fidelity across 40 styles (0.72 vs 0.27 semantic; 0.61 vs 0.25 global), and at roughly **half the variance** — it is not only stronger but far more reliable across styles.

---

## 1. Introduction

**Style transfer on a frozen generator is treated as one task, but users want two.** Given a content prompt and a reference image, a frozen text-to-image transformer can render the prompt's subject "in the style of" the reference. In practice two very different outcomes are desirable and are conflated under one name. In one, an artist wants the reference's *scene* preserved — its composition, its arrangement — with only the subject re-imagined in the reference's medium; this is **Semantic Style Editing**. In the other, the artist wants only the reference's *look* — its medium, texture, and palette — carried onto a freshly composed scene, with the reference's specific layout thrown away; this is **Global Style Transfer**. Conflating them yields methods that silently do one and fail the other.

**These two outcomes are the endpoints of a single controllable axis, which we call style locality.** Style locality asks *how much of the reference is transferred*: at the local end, everything is transferred (medium and composition), giving semantic editing; at the global end, only the transferable, position-free part (medium, palette, texture statistics) is transferred, giving global transfer. Framing the problem as one axis, rather than two disjoint tasks, is what lets a single operator serve both — and lets a user dial between them.

**We drive this axis with one frozen-model operator, derived rather than engineered.** Recent distribution-matching distillation separates its update into a *spear* (a classifier-free-guidance-augmentation term that sharpens toward the conditioning) and a *shield* (a distribution-matching term, requiring a separately trained fake-score network, that keeps samples on the data manifold). We show that for *test-time, single-sample* generation over a multi-step frozen reverse process, the shield is redundant with the reverse process itself and, when instantiated, actively destroys structured style. Removing it leaves a pure, frozen-teacher spear — no auxiliary network, no training — which we call **Test-time Score Distillation (TSD)**.

**Contributions.**
1. **A reframing.** Training-free style transfer is a single *style-locality* axis whose two ends (Semantic Style Editing, Global Style Transfer) are both first-class tasks; we serve both with one operator and a single free knob (reverse-step count).
2. **A regularization substitution, derived and verified.** We show the frozen reverse process replaces the trained distribution-matching shield of decoupled-DMD; the shield is unnecessary (no collapse even at large guidance) and harmful (a trained critic erases structured media). TSD keeps only the spear.
3. **A separable-axis evaluation protocol.** We evaluate style fidelity (CSD), subject fidelity (CLIP-T), and composition transfer (layout-leak) as three *separable* axes, and show that a single style number rewards subject-destroying over-stylization — a failure only the joint reading exposes. (We also tested, and honestly reject, a generator-native single-forward style metric read from the spear; §4.)
4. **A fair, strong baseline and a decisive result.** Against the same frozen model *directly instructed* to edit the image with its strongest style-transfer prompt, TSD reaches **2.5–2.7× its CSD** at matched subject fidelity across all 40 styles (with half the variance), and — unlike the native edit — can dissolve composition while keeping medium.

---

## 2. Related Work

**Distribution-matching distillation and its decoupling.** Distribution-matching distillation (DMD) trains a fast generator by matching its output distribution to a teacher's via a real-minus-fake score difference, which requires an auxiliary *fake-score* network tracking the student's current output distribution. Decoupled-DMD reorganizes the update into two additive, teacher-anchored terms: a distribution-matching *shield* $(s_{\text{cond}}-s_{\text{fake}})$ and a CFG-augmentation *spear* $(\alpha-1)(s_{\text{cond}}-s_{\text{uncond}})$, and observes that the spear does most of the sharpening while the shield stabilizes. Our method is the *test-time* specialization of this view: we keep the spear, and we replace the shield — not with a cheaper trained critic, but with the frozen reverse process itself. We use decoupled-DMD's decomposition as the exposition backbone; our departure is the claim that the shield's *network* is dispensable in the single-sample regime.

**Training-free style transfer and image editing on frozen transformers.** A large body of work steers frozen diffusion/rectified-flow models toward a reference style without weight updates, via guidance, latent optimization, or conditioning tricks. Our operator is a latent optimization driven by a score residual; it is *training-free of weights* but *optimization-based at test time*, which we state plainly rather than claiming a feed-forward free lunch. Distinct from editing methods that require masks, segmentation, or region blending, our Semantic Style Editing preserves the subject as an *emergent* property of the spear, with no mask — a property we quantify.

**What we deliberately do not use.** Faithful reproduction of a reference's *exact* structured texture — the specific tessellation of a mosaic, the precise leading of a stained-glass panel — is known to require manipulating self-attention key/value statistics. We do not use that mechanism, as either a component or a comparison; it is out of scope by design. Our claim is narrower and, we argue, more useful: the *transferable* part of style (medium, palette, texture statistics, not the exact motif layout) is exactly the part a frozen-model velocity spear can carry, and it is exactly the part the style-locality axis controls.

---

## 3. Method

### 3.1 Velocity differences are score differences

A rectified-flow model predicts a velocity $v_\theta(z_t,t,c)$ at latent $z_t$, time $t$, conditioning $c$. For the flow-matching schedule the velocity is an affine function of the conditional score, $s_\theta(z_t,t,c)=-\,(z_t-(1-\sigma)\hat x_0)/\sigma^2$, so a *difference* of velocities under two conditionings equals, up to one positive schedule factor, a difference of conditional scores:
$$
v_\theta(z,c_1)-v_\theta(z,c_2)\;\propto\;\nabla_z\log p_t(z\mid c_1)-\nabla_z\log p_t(z\mid c_2)\;=\;\nabla_z\log\frac{p_t(z\mid c_1)}{p_t(z\mid c_2)}.
$$
Every velocity difference the frozen model gives us is therefore the gradient of a log-density ratio we may ascend (attract) or descend (repel). This is the whole toolbox: no auxiliary network, only frozen forward passes.

### 3.2 The spear: CFG-augmentation as the style driver

Decoupled distribution matching writes its distillation gradient as
$$
\nabla \;\propto\; \underbrace{(s_{\text{cond}}-s_{\text{fake}})}_{\text{shield (distribution match)}}\;+\;\underbrace{(\alpha-1)\,(s_{\text{cond}}-s_{\text{uncond}})}_{\text{spear (CFG augmentation)}}.
$$
The spear is a *conditional contrast* evaluated at the sample's own state; it needs only the frozen teacher. We instantiate the style spear with two frozen velocities: $v_{\text{with}}$, conditioned on the reference image (the style evidence), and $v_{\text{without}}$, conditioned on the style *text* alone. The style guidance is
$$
g \;=\; v_{\text{with}} - v_{\text{without}} \;\propto\; \nabla_z \log \frac{p_t(z\mid \text{reference})}{p_t(z\mid \text{style text})},
$$
the extra style evidence the reference *image* supplies beyond what words can.

### 3.3 The regularization substitution: the frozen reverse process *acts as* the shield

**The shield exists to keep the spear from collapsing the sample off-manifold during distillation training.** In DMD, thousands of generator updates with only the spear would drift the student off the data manifold; the fake-score shield pulls it back. Our setting is different in the one way that matters: we do not train a generator. We run a frozen model's ordinary $T$-step reverse process and merely nudge the latent with the spear between steps. **Each reverse step (`scheduler.step`) already projects the latent back toward the model's data manifold — a free, exact, per-step distribution-matching regularizer.** This is the substitution: the reverse process replaces the trained shield.

**The trained shield is unnecessary.** If the spear alone drove samples off-manifold, large guidance would visibly collapse or oversaturate outputs; it does not — we sweep guidance strength well past standard values and observe no collapse, because every reverse step re-anchors the sample. The spear alone reproduces the effect the full DMD update was introduced to obtain.

**The trained shield is harmful.** We instantiate the literal shield with a real trained fake-score critic and add it back. It *hurts*: the critic pulls samples toward the natural-image manifold and undoes precisely the extrapolation that carries structured style — a mosaic reference is dragged back to naturalism and loses its tesserae. The shield's job (stay on the data manifold) is antagonistic to style transfer's job (leave the natural-image manifold for the reference's manifold), and in the training-free regime the reverse process already supplies as much manifold discipline as we can afford without erasing style.

Dropping the shield and keeping the spear gives **Test-time Score Distillation (TSD)**.

### 3.4 The rms-normalized pseudo-target: a fixed-magnitude extrapolation

We apply the spear as a pseudo-target in $x_0$-space rather than as a raw gradient. With $\hat x_0 = z - \sigma\,v_{\text{without}}$ the current denoised estimate, we form a detached target that is the estimate nudged along the *unit* style direction,
$$
x_0^{\text{tgt}} = \operatorname{sg}\!\big(\hat x_0 + \eta\,\widehat{g}\big),\qquad \widehat{g}=\text{rmsnorm}(g),\qquad
\mathcal L = \tfrac12\lVert \hat x_0 - x_0^{\text{tgt}}\rVert^2,
$$
so the latent update is exactly $\eta\,\widehat{g}$: a **fixed-magnitude step** of size $\eta$ along the style direction, applied every inner iteration. Two consequences follow. First, because rms-normalization removes the overall scale of $g$, the guidance weight $\alpha$ and any linear coefficient are **washed out** — only the *direction* of $g$ and the step size $\eta$ affect the output. Second, a fixed-magnitude step is an *extrapolation past* the weak fixed point $v_{\text{with}}=v_{\text{without}}$, whereas minimizing $\lVert g\rVert^2$ would *interpolate to* that fixed point and saturate early; the former transfers structured texture the latter misses while preserving content.

### 3.5 Style locality: one operator, two modes, one free knob

**Semantic Style Editing is the full spear on the reference.** With $g=v_{\text{with}}-v_{\text{without}}$ attracting to the whole reference — its medium *and* its composition — the prompt's subject is rendered in the reference's medium *within the reference's composition*. Subject preservation is emergent: no mask, no segmentation, no region blending.

**Global Style Transfer is the same spear with fewer reverse steps.** Reducing the reverse-step count under-determines the sample's structure, so the reference's specific composition never solidifies, while the per-step style spear still imprints the reference's medium and palette. Composition dissolves; medium remains. The step count is a *free, continuous* knob along the style-locality axis — from many steps (local, composition-preserving) to few steps (global, composition-dissolving).

### 3.6 Why velocity-space stops at the transferable part (analysis)

**The frozen-velocity spear transfers the position-free part of style, not the exact motif layout — which is exactly what the axis should control.** Reproducing a reference's *specific* tessellation or leading requires attention-level key/value manipulation, which we exclude. To characterize the boundary, we probe a taboo-clean, *position-only* construction: we compute the reference velocity with the reference tokens' positional identifiers scrambled, isolating the reference's appearance from its layout (this touches no attention statistics — only input positions consumed by the frozen model). Guided by this layout-free velocity, global transfer succeeds cleanly on *medium-dominant* styles (relief, halftone, stained-glass leading — the medium survives position scrambling) but collapses the subject on *figure-dominant painterly* styles (portrait mosaics, oil portraits), where medium and composition are entangled. The reverse-step knob is the robust instrument; the layout-free probe explains *why* certain references dissolve cleanly and others do not, and it supplies the generator-native metric of §4.

---

## 4. Evaluation Protocol and a Generator-Native Style Metric

**We measure three separable axes, because style transfer is not one number.** Style fidelity (is the medium transferred?), subject fidelity (is the prompt's subject present?), and composition transfer (is the reference's layout kept or dissolved?) can move independently, and a single metric hides failures. We report **CSD** (Contrastive Style Descriptor cosine similarity to the reference — style fidelity, designed to be content-invariant, though as we show below it is gameable at *extreme* over-stylization), **CLIP-T** (image–text alignment to the content prompt — subject fidelity), and a **layout-leak** measure (correlation of off-diagonal patch self-similarity — composition transfer). Layout-leak cleanly separates the two modes in our study: high for semantic editing (composition kept), low for global transfer (composition dissolved).

**CSD alone is misleading, which is why we pair it with CLIP-T.** In the semantic grid, pushing the spear harder monotonically raises CSD but past a point *destroys the subject*: the highest-CSD configuration (CSD ≈ 0.82) has CLIP-T collapsed to ≈ 0.21 and layout-leak ≈ 0.82 — the panda is being over-painted into pure texture. The right operating point balances CSD and CLIP-T rather than maximizing CSD; reporting both makes over-stylization visible instead of rewarding it. This is a boundary artifact, not our operating regime: we verify in §5 that our *benchmark* CSD advantage is not driven by it — the gap survives at matched layout-leak.

**A generator-native style metric, explored and rejected (honest negative result).** External style metrics (CSD, DINO) judge with a *separate* pretrained network; it is tempting to read style straight from the generator by re-evaluating the spear on the output, $\text{SRSD}(x,R)=\lVert v_{\text{with}}(x;R)-v_{\text{without}}(x)\rVert_{\text{rms}}$, expecting a small residual when $x$ already carries $R$'s style. We tested this ($n{=}20$, four configurations $\times$ five styles) and reject it. Its correlation with CSD is **positive, not negative** (Pearson $+0.79$, Spearman $+0.88$), and the correlation is carried almost entirely by over-stylized samples whose *content has collapsed* into a near-verbatim copy of the reference — exactly the samples CSD wrongly rewards. Controlling for spear strength, the residual correlation vanishes ($r{=}0.20$, n.s.), and scale-correction does not rescue it. The spear magnitude marks *content collapse / composition copying*, not style faithfulness; flipping its sign would reward the pathology. This is the same style–composition entanglement that bounds velocity-space transfer (§3.6): a frozen-velocity scalar cannot isolate pure style. We therefore keep the three external, separable axes and report this as a cautionary null.

---

## 5. Experiments

**Setup.** A single frozen text-to-image rectified-flow transformer serves as both conditional and unconditional predictor (one model, shared runtime); all results are training-free of weights. Unless noted, generations use seed 65 and the prompt "a panda eating bamboo" on a 40-reference style set spanning mosaic, relief, oil, halftone pop-art, and stained-glass media. We report CSD, CLIP-T, and layout-leak.

### 5.1 The style-locality axis and the parameter study

**The live knobs are step count / inner iterations, step size $\eta$, and per-token normalization; guidance weight is inert.** Consistent with §3.4, sweeping the guidance weight leaves the output unchanged (rms-normalization washes it out); we therefore grid the knobs that matter. For **Global Style Transfer** (few-step regime), CSD rises with step size, peaking at *num-steps 8, $\eta=0.12$* (CSD ≈ 0.63–0.64), above the default 8-step anchor (CSD 0.53), while layout-leak stays low (≈ 0.45, composition dissolved). For **Semantic Style Editing** (25-step regime), CSD rises with inner iterations and step size but over-stylizes at the top; the balanced optimum is *iters 5, $\eta=0.08$* (CSD 0.74, CLIP-T 0.28), versus the canonical anchor (CSD 0.70, CLIP-T 0.285). The full grids (both normalizations, all cells) are shown qualitatively and as CSD heatmaps.

### 5.2 Full benchmark against the frozen model's native edit

**Baseline design (a fair, strong baseline, not a straw man).** The most honest baseline is the *same* frozen model *directly instructed* to edit the reference with text — its native image+text edit. We first verify that this is a genuinely different mechanism: the frozen reverse process in our pipeline does not itself consume the edit image (the reference enters only through the spear), so "our operator with zero spear" is prompt-only text-to-image, not a native edit. We therefore run the model's own native edit pipeline at its documented best settings. Because that checkpoint exposes no system-prompt channel, we fold a strong expert style-transfer instruction into the prompt (and a negative prompt that deliberately does *not* penalize flat media, to avoid disadvantaging pop-art and stained-glass). This strong prompting substantially improves the baseline (mean CSD from 0.24→0.36 for the composition-preserving instruction, 0.03→0.29 for the new-composition instruction), so the comparison is against the baseline's best effort.

**Result (40 styles).** Even at its best, the native edit is dominated on style fidelity by a wide, reliable margin (Table 1). TSD reaches CSD **0.72 ± 0.08** (semantic) and **0.61 ± 0.09** (global) versus **0.27 ± 0.16** and **0.25 ± 0.16** for the native edit — a **2.5–2.7× style-fidelity gain** (2.69× semantic, 2.48× global). The most robust form of the claim is not the mean: **ours wins CSD on every one of the 40 styles, in both pairings (40/40 semantic, 40/40 global)** — despite the native edit's large per-style variance (whose lower tail overlaps ours), it never wins a single style. A spot-check on the native edit's worst style (a neon-anime reference, its CSD ≈ 0) shows why: the native edit defaults to a generic realistic panda that ignores the reference medium entirely, so its near-zero CSD is a genuine style-transfer failure, not a scoring artifact. Two things sharpen the result. First, the gap is at *matched or slightly lower* subject fidelity: the native edit keeps CLIP-T marginally higher (≈ 0.315 vs 0.30) because it stylizes less — an honest style↔subject trade-off, visible on the CSD–CLIP-T frontier. Second, the native edit's CSD variance is *twice* ours (std 0.16 vs 0.08): it is not only weaker but far less reliable, succeeding on genuinely relief-like media (e.g., bronze) and reverting to a *generic* engraving/line-art look elsewhere, missing the specific tesserae/impasto/halftone/leading that TSD transfers consistently. Finally, when instructed to change composition, the native edit *loses the medium* (kleinsys-gst CSD 0.25, below its own composition-preserving 0.27), whereas TSD's step-knob dissolves composition while *keeping* the medium (global CSD 0.61) — the native edit cannot separate style from layout, which is exactly the global-transfer end of the axis.

**Table 1. Full benchmark (40 styles, mean ± std).** Higher CSD = more faithful medium; higher CLIP-T = subject present; layout-leak = composition transfer (high = kept, low = dissolved).

| Method | CSD ↑ | CLIP-T ↑ | layout-leak |
|---|---|---|---|
| **Ours — Semantic Style Editing** | **0.719 ± 0.078** | 0.295 ± 0.030 | 0.572 ± 0.146 |
| **Ours — Global Style Transfer** | **0.615 ± 0.095** | 0.304 ± 0.026 | 0.414 ± 0.108 |
| Native edit — style prompt (keep comp.) | 0.267 ± 0.163 | 0.319 ± 0.012 | 0.317 ± 0.072 |
| Native edit — style prompt (new comp.) | 0.247 ± 0.155 | 0.315 ± 0.011 | 0.299 ± 0.064 |

CLIP-T is computed against the literal content prompt ("a panda eating bamboo"), uniformly across all four methods, so the comparison is fair; scoring against the fuller style-template prompt shifts every method up by a near-constant ≈ 0.03 and preserves the ordering. CSD and layout-leak are prompt-independent.

**Is the style-fidelity gap real style, or just more reference-copying?** Because CSD can be gamed by near-verbatim reference copies at extreme over-stylization (§4), and our outputs carry higher layout-leak than the native edit, a skeptic could ascribe the CSD gap to copying rather than style. It is not. We lead the pure-style claim with **Global Style Transfer**, whose whole point is to *dissolve* the reference's composition: its layout-leak (0.41) is close to the native edit's (0.30), yet it still wins CSD 0.61 vs 0.25 — a 2.5× gap at comparable composition-copying. More directly, we restrict to the 14 styles where our global output copies *no more* layout than the native edit (layout-leak ≤ its mean+1σ = 0.36) and find ours still wins CSD **14/14** (0.61 vs 0.24); within the global grid the CSD–leak correlation is weak (+0.29). The gap survives at matched layout-leak, so it is style fidelity, not composition-copying.

**The CSD advantage is robust across content prompts.** To confirm the win is not an artifact of the single benchmark prompt, we re-ran a media-diverse 5-style subset (mosaic, bronze, oil, pop-art, stained-glass) across four additional content prompts (a seaside castle, a cat on a windowsill, a sunflower field, an aged fisherman). Ours wins CSD on every style×prompt cell — **20/20 in both pairings, zero flips** — with per-prompt ratios of 1.55–2.31× (semantic) and 1.65–2.20× (global). The multiplier on this subset is ≈ 1.9–2×, not the 2.5–2.7× headline: the headline is the full 40-style figure, whereas this deliberately media-diverse subset is harder and its own single-prompt anchor already sits at ≈ 2×. The style↔subject trade-off is prompt-dependent — near-matched CLIP-T on subject-light prompts, but on the *portrait* prompt our CLIP-T falls further (gap up to ≈ 0.08) as the spear over-stylizes the face — the same honest trade-off, more visible where subject identity is strong.

*(5-style calibration, consistent with the current operating configuration: ours semantic 0.70 / native-sem 0.36; ours global 0.63 / native-gst 0.29; the full 40-style benchmark strengthens the gap to 2.5–2.7× and to 40/40 per-style wins.)*

### 5.3 Ablations

**Shield vs. no shield.** Adding a trained fake-score shield (the literal DMD term) degrades style: structured media are pulled back toward naturalism (mosaic tesserae lost). The frozen reverse process alone is the better regularizer in the training-free regime. **Spear direction vs. magnitude.** The fixed-magnitude rms-normalized step (extrapolation) transfers texture that magnitude-minimizing objectives (interpolation to the fixed point) miss, at equal content preservation. **Guidance weight.** Sweeping $\alpha$ leaves outputs unchanged, confirming rms-normalization removes scale — the direction is the signal.

---

## 6. Conclusion

Training-free style transfer on a frozen transformer is one controllable *style-locality* axis, and a single derived operator — the CFG-augmentation spear of decoupled distribution matching, applied as a fixed-magnitude rms-normalized latent step — serves both its ends: Semantic Style Editing with the full spear, Global Style Transfer with fewer reverse steps. The key design choice is a *regularization substitution*: the frozen reverse process replaces the trained distribution-matching shield, which is both unnecessary and harmful in the single-sample test-time regime. Evaluated on three separable axes, the operator roughly doubles style fidelity over the same model's strongest native edit while preserving the subject and, uniquely, dissolving composition without losing medium. A generator-native single-forward style metric read from the spear is tempting but, we show, invalid — it measures content collapse, not style — a cautionary result that reinforces why style and composition must be read on separate axes.

**Limitations (stated plainly).** TSD is training-free of weights but optimization-based at test time (a few latent iterations per step); we do not claim a feed-forward free lunch. Our empirical comparison is to the *same frozen model's* strongest native image+text edit — this isolates what TSD adds to the base model, and is not yet a comparison to published training-free style-transfer methods; that comparison is the natural next step (the K/V-based faithful-motif line is out of scope by design). Exact structured-motif reproduction is likewise out of scope. On already-graphic references the medium is near-saturated, so gains are neutral — a limitation, not a failure. The "regularization substitution" is an *empirical* finding (no collapse under large guidance; a trained shield hurts), not a formal theorem.

---

## Self-review (five-dimension, per skill)

- **Contribution.** Reframing (one axis, two tasks) + regularization substitution (reverse process replaces trained shield) + 3-axis protocol & SRSD + fair strong baseline. *Risk:* reviewers may see "spear-only DMD" as incremental — mitigate by foregrounding the *test-time* substitution argument and the harmful-shield ablation, which are non-obvious. | Evidence: §3.3, §5.3 | Status: supported (5-style); ⟨bench⟩ full-scale pending.
- **Writing clarity.** One message per paragraph, topic sentences first, decoupled-DMD spear/shield vocabulary kept stable. Status: pass on this draft; needs paragraph-level flow pass.
- **Experimental strength.** Param grids + full 40-style benchmark + fair strengthened baseline + ablations. *Risk:* single prompt (kp0) at benchmark scale — extend to kp1–kp4. | Status: ⟨bench⟩ running; multi-prompt = TODO.
- **Evaluation completeness.** Three separable axes; CSD-alone-misleading shown quantitatively. A generator-native metric (SRSD) was tested and rejected (honest null, §4). *Risk:* single prompt at benchmark scale. | Status: protocol supported; SRSD resolved (rejected).
- **Method soundness.** Derivation from velocity=score; α-washout verified in code; shield-harmful verified empirically. *Risk:* the "reverse process = shield" claim is argued + supported by no-collapse and harmful-shield evidence, not proven formally. | Status: supported empirically; consider a short formal note.

### Claim–evidence map (major claims)
- Claim: two modes are one controllable axis served by one operator | Evidence: §3.5, param study §5.1, layout-leak separation §4 | Status: supported.
- Claim: frozen reverse process substitutes the trained shield (unnecessary + harmful) | Evidence: no-collapse under large guidance; ca_dmd erases mosaic | Status: supported (empirical).
- Claim: α/λ washed out by rms-norm | Evidence: code (pseudo_grad rms-normalized) + guidance sweep unchanged | Status: supported.
- Claim: 2.5–2.7× CSD over the native edit, winning every style | Evidence: 40-style benchmark (0.72 vs 0.27 semantic, 0.61 vs 0.25 global; ours std 0.08 vs klein 0.16; **40/40 per-style wins in both pairings**, zero give-ups) | Status: **supported (full 40-style, per-style)**.
- Claim: a spear-magnitude style metric (SRSD) is invalid | Evidence: SRSD↔CSD +0.79/+0.88, driven by content-collapse; within-config residual r=0.20 n.s. | Status: supported (rejected as metric).

### TODO to finalize
1. ~~Fill ⟨bench⟩ numbers~~ **DONE** — Table 1 filled from the 40-style benchmark (`benchmark_full_summary.csv`); gap is 2.5–2.7×.
2. ~~SRSD validation~~ **DONE — rejected** (§4, honest null; positive correlation via content-collapse).
3. ~~Multi-prompt (kp1–kp4) robustness~~ **DONE** — 5 media-diverse styles × 4 prompts, ours wins CSD 20/20 in both pairings (zero flips); the *win* is prompt-robust, the *multiplier* is ≈1.9–2× on this hard subset (2.5–2.7× is the 40-style headline); portrait prompt sharpens the CLIP-T trade-off (§5.2). Also fixed the stale 5-style calibration number (ours global 0.53→0.63, the operating-config value).
4. ~~Port to `AAAI/main.tex`~~ **DONE** — `AAAI/main.tex` compiles clean (9pp, 0 undefined, no backbone name in text layer); all 4 post-review edits synced. Teaser deferred (user directive). **Remaining: trim body 8pp→7pp for AAAI limit.**
5. Figures ready on R2: paper_q1_method_csd_bar, paper_q2_paramgrid_csd_heatmap, paper_q3_csd_clipt_scatter, paper_q4_benchmark_40style; klein_fair_compare, mainline_semedit_multiprompt, gst_e5/se_e6 param grids.
