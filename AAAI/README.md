# AAAI Paper Draft

This directory contains the submission draft for the second paper:

**From Visual Goal to Denoising Flow: Carrier-Anchored Image CFG for
Training-Free Style Transfer**

## Files

- `main.tex`: seven-page technical paper, two-page bibliography, and checklist.
- `supplement.tex`: protocol, per-pair values, field conditioning, and the
  negative image-reward diagnostic.
- `references.bib`: checked bibliography used by BibTeX.
- `reproducibility.tex`: AAAI reproducibility answers included after references.
- `BUILD.md`: exact compile and validation workflow.
- `SUBMISSION_READINESS.md`: claim boundaries and remaining top-tier gates.

## Current Position

The headline method is online carrier-anchored image CFG. It requires no
adapter training: a same-state native with/without-reference score contrast is
the visual goal, a detached content-only affinity is the spatial carrier, and a
scheduler-consistent clean-state target executes the update.

The main paper deliberately does not claim a completed CA+DMD improvement. A
valid CA+DMD extension requires a persistent dynamic fake-score backend. The
optional image-code student is amortization for repeated deployment, not a
prerequisite for the online method.

Headline evidence:

- strict representative audit: CSD `0.689619`;
- three-pair matched mean: K/V `0.402729` vs. GoalCarrier `0.566176` CSD;
- carrier intervention on the representative pair: leakage `0.3018 -> 0.2780`;
- local noisy-state CSD reward is a negative supplementary diagnostic, not
  final-image best-of-N.

## Page Contract

The current main PDF contains seven pages of technical content, two pages of
references, and one page of the required AAAI reproducibility checklist. The
checklist is after the references and does not count toward the technical page
limit. The supplement is compiled separately.

The working source uses a portable article preamble. It must be switched to the
official AAAI-26 author-kit style before submission. See `BUILD.md` and
`SUBMISSION_READINESS.md` before changing claims or the page budget.
