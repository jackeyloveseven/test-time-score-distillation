# AAAI Paper Draft

This directory contains the AAAI-style paper draft for the second paper:

**Goal, Carrier, and Flow: Online Image-Conditioned Score Distillation for
Style Transfer**

## Files

- `main.tex`: main paper draft, written in a self-contained two-column format.
- `supplement.tex`: supplementary material, including the formal protocol audit
  and the online image-reward diagnostic.
- `references.bib`: working bibliography for replacing the inline references
  with the official AAAI bibliography workflow.
- `BUILD.md`: local compilation and evidence provenance notes.

## Scope

The main claim is an online, training-free, image-conditioned score refinement
method. The reference image supplies a native style goal; the current content
latent supplies a reference-independent spatial carrier; the flow/scheduler
executes the resulting update. Offline student distillation is optional and is
not required by the main method.

The image-reward experiments are supplementary diagnostics only. They are not
presented as a method improvement. In particular, the local CSD oracle is not a
true final-image best-of-N oracle: it scores noisy intermediate VAE decodes and
selects a myopic action.

## Evidence Boundary

The strict formal score-carrier audit reproduces CSD `0.689619` on style 27,
prompt kp0, seed 64. The historical formal record is approximately `0.688` and
the stored score-match record is `0.710720`. The supplementary image-reward
comparison is intentionally reported as a negative alignment result.

## Compilation

The files are written to compile with a normal LaTeX installation without the
proprietary AAAI style file. For an actual submission, replace the local
article preamble with the official AAAI template while preserving the body.

See `BUILD.md` for the recommended command and the exact experiment sources.
