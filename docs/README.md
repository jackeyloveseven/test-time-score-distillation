# Documentation Map

Status date: 2026-07-14. Documents are classified so historical research notes
cannot silently override the current method.

## Read in This Order

1. `PROJECT_STATE.md`: shortest current truth.
2. `AGENT_HANDOFF.md`: complete operational takeover.
3. `RESULTS_CATALOG.md`: paper-facing metrics and provenance.
4. `environment.md`: server and Mac environments.
5. `todo.md`: active queue.
6. `experiment_plan.md`: remaining matched protocol.
7. `../AAAI/main.tex`: only source of truth for paper prose.

## Current Documents

| document | purpose |
|---|---|
| `PROJECT_STATE.md` | current method, decisions, validated/unvalidated claims |
| `AGENT_HANDOFF.md` | from-zero operational guide |
| `RESULTS_CATALOG.md` | compact canonical numbers and artifact paths |
| `environment.md` | H200 runtime, models, evaluators, Mac limitations |
| `LOCAL_MAC_HANDOFF.md` | local workspace layout and usage |
| `todo.md` | prioritized active work |
| `experiment_plan.md` | matched experiment matrix and reporting contract |
| `formal25_teacher_comparison.md` | formal GoalCarrier evidence ledger |
| `carrier_anchored_score_residual.md` | carrier implementation note |
| `paper_story_goal_carrier_flow.md` | motivation and future causal-probe blueprint |
| `full_paper_draft.md` | short Markdown synopsis; points to LaTeX truth |

## Experiment Ledgers

These contain useful chronological detail but do not define current status:

| document | scope |
|---|---|
| `with_style_teacher_score_plan.md` | native teacher, CA/DMD, and long experiment ledger |
| `score_residual_distillation_plan.md` | optional offline student history |
| `klein9b_probe_notes.md` | probe and adapter experiment chronology |
| `profile_fields_notes.md` | field profiling notes |
| `reference_data_and_dmd_inventory.md` | model/data/backend inventory |

## Historical Design Documents

The following are retained for provenance. Each now carries a status banner.
Do not use their old priorities as current instructions:

- `adaptation_notes.md`
- `migration_plan.md`
- `technical_plan.md`
- `residual_optimization_plan.md`
- `klein9b_probe_plan.md`
- `reference_kv_score_distillation_spec.md`

## Paper Documents

```text
../AAAI/main.tex                 paper prose truth
../AAAI/supplement.tex           optional student and reward diagnostics
../AAAI/references.bib           bibliography
../AAAI/reproducibility.tex      AAAI checklist
../AAAI/SUBMISSION_READINESS.md  evidence and format gates
../AAAI/BUILD.md                 compile/validation commands
```

## Trial Documents

`../trial/online_rl_critic/README.md` is the index for the isolated reward
trial. `PROTOCOL_AUDIT.md` is the final corrected result. Earlier `RESULTS`,
`GRPO_RESULTS`, and `SAME_STATE_RESULTS` files are chronological diagnostics.

## Generated Evidence

Raw results are present on `ubuntu-107` under `results/` and ignored by Git.
The Mac workspace is documentation-only and does not mirror the 1.7 GB result
tree. Use `RESULTS_CATALOG.md` for compact offline access; inspect raw artifacts
on the server when needed.
