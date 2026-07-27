# Submission Readiness Audit

Status date: 2026-07-27.

## Current Verdict

SUBMISSION-READY. The paper is structurally complete and empirically strong
at fixed seed; no non-human validation runs remain. The reverse-shield
ablation is done (Fig 4, ρ=0 collapses all five preselected styles). The
multi-seed matrix (seeds 64/66) is intentionally deferred future work, not a
submission blocker — the paper is honestly single-seed and labels its SD as
cross-style dispersion. Human evaluation is explicitly outside the current
scope. The only remaining items are human actions: upload the 3 PDFs
(main.pdf, supplement.pdf, checklist.pdf) to the AAAI/OpenReview site
(main-PDF deadline 2026-07-28), and finalize the code/supplement appendix and
license wording (due 2026-07-31).

## Completed Scientific Work

- Implementation-exact same-state TSD objective and latent-only optimization.
- Semantic and Global task definitions with mode-specific operating points.
- Fixed-seed 40-style benchmark with complete per-style reporting.
- Five-style multi-prompt extension.
- CSD/CLIP-T/layout-leak joint interpretation and matched-layout analysis.
- Six-axis primary Agent benchmark.
- Secondary-Agent agreement audit with explicit 68/80 coverage and no
  imputation.
- Local wrong-reference style control.
- Fixed five-style FireRed/Qwen matched-input perceptual audit.
- Negative fake-score study and rejected SRSD metric analysis.
- Explicit validity boundary: Agent evaluation is not human validation.

## Completed Presentation Work

- Official AAAI 2027 author kit applied.
- Related Work constrained to two paragraphs with expanded citations.
- Three separate PDFs: main.pdf = 9 US-Letter pages (technical p1-7,
  references only p8-9); supplement.pdf = 11 pages; checklist.pdf = 2 pages
  (standalone official AAAI-2027 28-item reproducibility checklist, not
  appended to the supplement).
- Large blank regions removed through float and caption optimization.
- 15 active self-contained figure packages validate.
- Canonical TSD framework figure, quantitative results figure, dense style
  grid, reverse-shield ablation figure, external-model audit, and Agent
  benchmark workflow are wired into the paper/supplement. (The six-axis
  Agent radar figure was removed from the paper; see Canonical Evidence.)
- Current builds have no undefined references, missing figures, or overfull
  boxes.
- All AAAI format gates pass: Type-3 fonts = 0 (found and fixed 2026-07-27
  via matplotlib `pdf.fonttype=42`, 6 figures regenerated), all fonts
  embedded, 39 cited references (1 uncited entry removed from the 40-entry
  .bib), aaai2027.sty/.bst byte-identical to the official AuthorKit27, no
  forbidden packages, no `\nocopyright`, anonymous author block.

## Remaining P0 Evidence Gates

None. All P0 evidence gates are closed. The reverse-step-strength ablation
`{1, 0.5, 0}` is complete and bounds the mechanism claim (Fig 4).

Historical note: seeds 64 and 66 for the four-method, 40-style paired matrix
were queued to report seed-level uncertainty separately from cross-style
dispersion, and a reconcile-Abstract/Introduction/Results/Conclusion step was
queued in case that run materially changed conclusions. This multi-seed run
is now deferred future work (see `docs/todo.md`), not a submission gate, so
the reconcile step is moot for this submission.

## Remaining P1 Submission Gates

All non-human P1 gates are closed (open-weight third-judge replication is
done via the Gemma-4 evaluation-robustness study; format/anonymity/font
checks are done, see Completed Presentation Work above). What remains is
human action only:

- Upload the 3 PDFs (main.pdf, supplement.pdf, checklist.pdf) to the
  AAAI/OpenReview site (main-PDF deadline 2026-07-28).
- Finalize the code/data release manifest, evaluator hashes,
  redistribution/license notes, and Data/Code Availability wording (due
  2026-07-31).

## Claims That Must Not Drift

- TSD is training-free of weights but optimization-based at test time.
- The fixed seed-65 SD is cross-style dispersion, not random-seed uncertainty.
- The controlled baseline is the same native model path, not a broad external
  leaderboard.
- The FireRed/Qwen comparison is a five-style matched-input audit, not a
  compute-matched SOTA claim.
- The reverse-process regularizer is an empirical interpretation, not an exact
  projection theorem.
- Persistent malformed Agent responses are not imputed.
- Radar area is not a scalar metric.
- Agent evaluation does not replace human evaluation.
- Exact structured-motif transfer and attention/K/V surgery are out of scope.

## Canonical Evidence

- `main.tex`, `supplement.tex`, `checklist.tex`
- `supplement/data/`
- `figures/reverse_shield/`
- `figures/main_results/`
- `figures/external_model_audit/`
- `figures/open_judge_validation/`
- `figures/agent_benchmark_pipeline/`
- manifests and summaries produced by the validation scripts

Note: the six-axis Agent radar figure (`figures/agent_multidim_radar/`) was
removed from the paper; it is no longer canonical evidence.
