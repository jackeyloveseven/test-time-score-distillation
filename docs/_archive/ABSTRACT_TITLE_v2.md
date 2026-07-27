# Title + Abstract v2 (honest-scope, panel-validated) — 2026-07-20

> Reflects the VLM-panel findings (docs/research/VLM_JUDGE_FINDINGS.md). Honest-scope
> framing (no method change). If the fix-probe direction is chosen and succeeds, add
> one sentence on the regularizer-schedule rescue. NOT yet applied to main.tex (held).
> Every number traces to results/vlm_judge/verdicts.jsonl + results/*metrics*.csv.

## Recommended title
**Style for Free, but Not for All: Native Score Residuals for Training-Free
Style Transfer in Frozen Rectified-Flow Transformers**

### Alternatives
1. Content for Free, Style When Nameable: A Native Score-Residual Method and the
   Limits of Style Metrics
2. What Transfers for Free? Characterizing Native Score Residuals for Training-Free
   Style Transfer on a Frozen Rectified-Flow Transformer
3. (conservative, method-forward) Native Same-State Score Residuals for Training-Free
   Style Transfer in Frozen Rectified-Flow Transformers

## Abstract
Training-free style transfer on modern text-to-image diffusion usually bolts on
extra machinery — attention-feature replacement, learned adapters, or hand-designed
transport. We show that on a frozen rectified-flow transformer (FLUX.2 [klein] 9B) a
single native signal drives high-quality transfer: the *same-state image-CFG score
residual* — the difference between the model's score with and without a style
reference as an image condition, evaluated at the identical noise state — applied by
optimizing only the current latent (AdamW, a few inner iterations per denoising step).
We train nothing and add no modules. Rather than claim uniform success, we
characterize the method with an independent, discrimination-controlled VLM judge
panel. Content is preserved essentially perfectly across every style, and the panel
rates 66% of transfers as successful — 54% as indistinguishable in artistic medium;
however, transfer robustly fails on a family of synthetic-render, mosaic, and
crystalline styles, where the content-preservation term dominates and only the color
palette transfers. We further show, and independently validate, that the CSD style
metric cannot rank quality among successful transfers (CSD spans 0.36–0.86 across
panel-excellent cells), cautioning against CSD-only evaluation.

## Number provenance
- 66% = 63/95 cells with panel style-match >=4; 54% = 51/95 at style-match 5.
- content preserved = content_clarity 5.0 on all 95 cells (verdicts.jsonl).
- 7 robust-failure style types (04,08,14,21,31,34,37) = synthetic-render/mosaic/crystalline.
- mechanism = 31/31 failures are content 5.0 + style 1.65 (content-reg dominates).
- CSD 0.36–0.86 among panel-excellent (pro>=4, n=63); discrimination Δ=2.66; inter-judge r=0.74.
