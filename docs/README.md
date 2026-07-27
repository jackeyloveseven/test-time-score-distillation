# Documentation Map

Status date: 2026-07-27.

The active documentation surface is intentionally small. Historical experiment
logs and pre-TSD designs remain available for provenance, but they do not define
the current method, claims, or queue.

## Read in This Order

1. `PROJECT_STATE.md` - shortest authoritative project state.
2. `HANDOFF.md` - operational takeover, current artifacts, and resume commands.
3. `todo.md` - prioritized remaining work.
4. `EXPERIMENT_PLAN.md` - exact unfinished validation matrix.
5. `environment.md` - runtime, model, evaluator, and coexistence rules.
6. `../AAAI/main.tex` - canonical paper prose.
7. `../AAAI/supplement.tex` - canonical supplementary prose.
8. `../AAAI/figures/FIGURE_WORKFLOW.md` - mandatory figure-package contract.

## Active Documents

| Document | Purpose |
|---|---|
| `PROJECT_STATE.md` | Current method, evidence, limitations, PDF and figure status |
| `HANDOFF.md` | From-zero operational continuation guide |
| `todo.md` | Current P0/P1/P2 queue |
| `EXPERIMENT_PLAN.md` | Multi-seed and reverse-shield protocols still to run |
| `LOOP_PLAN.md` | Current autonomous loop and GPU-resume policy |
| `environment.md` | Runtime and reproducibility contract |
| `research/PAPER_PITCH.md` | Reviewer-facing paper story and claim/evidence map |
| `research/VLM_JUDGE_FINDINGS.md` | Consolidated Agent benchmark validity findings |

## Paper Documents

```text
../AAAI/main.tex                 canonical main paper
../AAAI/supplement.tex           canonical supplementary material
../AAAI/references.bib           bibliography
../AAAI/README.md                paper-level status
../AAAI/BUILD.md                 compile and validation commands
../AAAI/SUBMISSION_READINESS.md  evidence and submission gates
../AAAI/figures/README.md        active figure inventory
```

Current compiled outputs are `AAAI/main.pdf` (9 US-Letter pages: technical
content p.1-7, references p.8-9), `AAAI/supplement.pdf` (11 US-Letter pages),
and the standalone `AAAI/checklist.pdf` (2 US-Letter pages), as of 2026-07-27.

## Historical Material

- `docs/_archive/` contains superseded handoffs, plans, and design records.
- `docs/research/PAPER_MAIN_20260721.md` is a historical Markdown draft; use
  `AAAI/main.tex` for current wording and numbers.
- `AAAI/main_backup_pre_20260721.tex` and `AAAI/main_v1_kv_frozen.tex` are
  archival LaTeX snapshots and must not be edited into the submission.
- `AAAI/figures/pipeline/` and `AAAI/figs/_archive/` are visual archives.

## Source-of-Truth Rule

When documents disagree, use this order:

1. current generated data and manifests;
2. `AAAI/main.tex` and `AAAI/supplement.tex`;
3. `PROJECT_STATE.md` and `HANDOFF.md`;
4. historical research notes and logs.
