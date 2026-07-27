# Current Loop Plan

Status date: 2026-07-23.

The historical autonomous loop is archived under
`docs/_archive/2026-07-23-pre-doc-sync/LOOP_PLAN.md`. The current loop is much
smaller because the paper, supplement, benchmark, and figure surface are
already assembled.

## Loop Objective

Close the remaining non-human validation without changing the benchmark,
backbone, prompts, or current paper story.

## Each Iteration

1. Check GPU free memory and foreign processes.
2. If sufficient headroom exists, run or resume
   `scripts/run_remaining_validation.sh`.
3. If GPU work is blocked, advance one non-GPU submission task: documentation,
   release manifest, citation audit, PDF inspection, or reviewer-risk review.
4. Inspect every completed batch qualitatively before trusting aggregate
   metrics.
5. Update `PROJECT_STATE.md`, `HANDOFF.md`, `todo.md`, and
   `SUBMISSION_READINESS.md` only when evidence status changes.

## GPU Queue

The serial queue performs:

1. paired multi-seed benchmark completion;
2. reverse-shield ablation.

The queue is resumable and skip-existing. It is not running at this update.
The most recent log was blocked by the 24-GiB headroom gate. Relaunch only when
`nvidia-smi` shows adequate free memory; never terminate foreign jobs.

## Non-GPU Queue

- Maintain the 8-page main and 8-page supplement contracts.
- Keep all 15 active figure packages valid.
- Prepare public-open-weight judge replication without changing the existing
  primary/secondary reports.
- Complete release, license, anonymity, and final-review checklists.

## Stop Conditions

The loop is complete when:

- seeds 64--66 are complete for all four matched methods;
- reverse-shield outputs and diagnostics are aggregated;
- paper claims are reconciled with those results;
- no required submission metadata, release statement, or rendering check is
  unresolved.
