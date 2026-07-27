# Canonical Title and Abstract

Status date: 2026-07-23.

This file mirrors `AAAI/main.tex`. The LaTeX source remains authoritative.

## Title

**Test-Time Score Distillation: Mask-Free Semantic Style Editing with
Reverse-Process Regularization**

## Abstract

Training-free style transfer with a frozen text-to-image transformer conflates
two useful outcomes: preserving a reference composition while changing its
subject, and transferring only the reference medium to a new composition. We
formulate them as endpoints of a single *style-locality* axis and control both
with **Test-Time Score Distillation (TSD)**. TSD evaluates a same-state contrast
between reference-conditioned and reference-free frozen velocities, converts
it into a fixed-magnitude rms-normalized latent update, and interleaves that
update with the model's ordinary reverse steps. This gives an empirical
*regularization substitution*: in our single-sample test-time regime, the
frozen reverse trajectory repeatedly re-anchors the edited latent, while an
added learned fake-score term suppresses structured media rather than improving
stability. The number of reverse steps then acts as a locality control: more
steps preserve reference composition for mask-free semantic style editing;
fewer steps dissolve composition for global style transfer. We evaluate style
fidelity, subject fidelity, and composition transfer separately using CSD,
CLIP-T, and layout leak. Across 40 references, TSD improves CSD from 0.267 to
0.719 in the composition-preserving setting and from 0.247 to 0.615 in the
composition-dissolving setting, winning all 40 paired styles in both cases,
with a modest CLIP-T decrease of about 0.02. A five-style, four-prompt extension
yields 20/20 additional paired wins. TSD trains no weights, but remains an
optimization-based test-time method.

## Claim Boundary

- The 40-style result is fixed seed 65; reported SD is cross-style dispersion.
- Multi-seed uncertainty remains pending for seeds 64 and 66.
- Reverse-process regularization is an empirical interpretation, not a theorem.
- The five-style FireRed/Qwen audit is not a compute-matched leaderboard.
- Agent evaluation is a perceptual diagnostic, not human validation.

