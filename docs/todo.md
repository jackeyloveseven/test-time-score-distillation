# Current TODO

Status date: 2026-07-23.

`PROJECT_STATE.md` defines the current claims. This queue contains only work
that remains relevant to the present TSD paper.

## P0: Finish Queued Validation

1. Free root-filesystem space; the server currently has roughly 18 MiB free.
2. Relaunch `scripts/run_remaining_validation.sh` when disk and GPU headroom are adequate.
3. Complete the paired 40-style, three-seed matrix for TSD Semantic, TSD Global,
   native keep-composition, and native new-composition. Seeds 64 and 66 remain:
   320 cells total.
4. Aggregate style-bootstrap confidence intervals, seed-level variability,
   cell/style win rates, runtime, peak memory, forward calls, and backward calls.
5. Run the reverse-shield ablation with reverse-step strengths `{1, 0.5, 0}`
   under standard and stress settings on the preselected five styles.
6. Update paper numbers only if the completed paired evidence changes the
   current fixed-seed conclusions.

## P1: Submission Strengthening

- Replicate the local third-judge protocol with a publicly downloadable
  open-weight VLM once GPU memory permits.
- Finalize code/data release manifests, evaluator hashes, model/license notes,
  and the Data/Code Availability wording.
- Run final anonymous-submission checks: author metadata, PDF metadata, font
  embedding, figure legibility, references, and supplementary file naming.
- Perform a final adversarial paper review focused on the empirical status of
  reverse-process regularization and the bounded external-model audit.

## P2: Optional Extensions

- Multi-backbone replication.
- Human preference study and Agent-human correlation.
- Feed-forward amortization of the test-time optimization.
- Attention-level exact-motif transfer, as a separate method family rather
  than a hidden extension of TSD.

## Completed

- Official AAAI 2027 style applied; main and supplement both compile to 8 pages.
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

## Guardrails

- Never kill foreign GPU jobs.
- Never describe fixed-seed cross-style SD as random-seed uncertainty.
- Never impute persistent malformed Agent responses.
- Never use radar area as a scalar score.
- Never imply that Agent evaluation replaces human evaluation.
- Never promote a figure without its package, source/prompt, intent, caption,
  QA, and provenance.
