# Current TODO

Status date: 2026-07-27.

Paper is final and pushed; remaining = human upload (7/28) + code
appendix/license (7/31).

`PROJECT_STATE.md` defines the current claims. This queue contains only work
that remains relevant to the present TSD paper.

## P0: Human Submission Actions

1. Upload the 3 PDFs (main.pdf, supplement.pdf, checklist.pdf) to the
   AAAI/OpenReview site. Main-PDF deadline: 2026-07-28.
2. Finalize the code/supplement appendix and license wording. Due: 2026-07-31.

## P1: Submission Strengthening

Non-human items are done: the third-judge protocol has been replicated with
a publicly downloadable open-weight VLM (Gemma-4 evaluation-robustness
study), and anonymous-submission/format checks are complete (see Completed,
below). What remains is human action:

- Finalize code/data release manifests, evaluator hashes, model/license
  notes, and the Data/Code Availability wording. Due 2026-07-31.

## P2: Optional Extensions

- Multi-backbone replication.
- Human preference study and Agent-human correlation.
- Feed-forward amortization of the test-time optimization.
- Attention-level exact-motif transfer, as a separate method family rather
  than a hidden extension of TSD.

## Deferred / Future Work (not submission blockers)

- Complete the paired 40-style, three-seed matrix for TSD Semantic, TSD
  Global, native keep-composition, and native new-composition (seeds 64 and
  66; 320 cells total). The paper is honestly single-seed and labels its SD
  as cross-style dispersion; this is out-of-budget future work, not a gate.
- Free root-filesystem space and relaunch
  `scripts/run_remaining_validation.sh` when disk and GPU headroom are
  adequate, to support the above.
- Aggregate style-bootstrap confidence intervals, seed-level variability,
  cell/style win rates, runtime, peak memory, forward calls, and backward
  calls once the multi-seed matrix exists.
- Update paper numbers only if the completed paired evidence changes the
  current fixed-seed conclusions.

## Completed

- Official AAAI 2027 style applied; main.pdf = 9 US-Letter pages (technical
  p1-7, references p8-9); supplement.pdf = 11 pages; checklist.pdf = 2 pages
  (standalone AAAI-2027 28-item reproducibility checklist, not appended to
  the supplement).
- All AAAI format gates pass: 0 overfull/undefined, all fonts embedded,
  Type-3 fonts = 0 (found and fixed 2026-07-27 via matplotlib
  `pdf.fonttype=42`), US-Letter, aaai2027.sty/.bst byte-identical to the
  official AuthorKit27, no forbidden packages, no `\nocopyright`, 39 cited
  references, anonymous author block.
- Related Work compressed to two paragraphs and bibliography expanded.
- Fixed-seed 40-style benchmark and full per-style table completed.
- Five-style multi-prompt extension completed.
- Six-axis task-aware Agent benchmark, secondary reliability audit, local
  wrong-reference control, and five-style FireRed/Qwen audit completed.
- External-model radar and qualitative plate completed.
- Agent benchmark workflow figure generated, corrected, packaged, validated,
  and inserted into the supplement.
- Figure-package validation passes 15 active packages.
- SRSD validation completed and rejected as an honest null.
- Reverse-shield ρ-knockout ablation completed (Fig 4): ρ=0 collapses all
  five preselected styles.

## Guardrails

- Never kill foreign GPU jobs.
- Never describe fixed-seed cross-style SD as random-seed uncertainty.
- Never impute persistent malformed Agent responses.
- Never use radar area as a scalar score.
- Never imply that Agent evaluation replaces human evaluation.
- Never promote a figure without its package, source/prompt, intent, caption,
  QA, and provenance.
