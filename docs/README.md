# Documentation Map

This directory contains both the current takeover documentation and historical
research notes. A new agent should read the files in this order.

## Priority Order

1. [`AGENT_HANDOFF.md`](AGENT_HANDOFF.md): authoritative operational guide.
2. [`full_paper_draft.md`](full_paper_draft.md): current paper story,
   equations, evidence, and claim boundary.
3. [`formal25_teacher_comparison.md`](formal25_teacher_comparison.md): native
   teacher comparison and the first carrier results.
4. [`environment.md`](environment.md): verified environments and startup
   checks.
5. [`todo.md`](todo.md): current status and prioritized work queue.
6. [`experiment_plan.md`](experiment_plan.md): experiment matrix and reporting
   protocol.

## Current Method Notes

| document | role | status |
|---|---|---|
| `carrier_anchored_score_residual.md` | carrier-anchored field implementation | current technical note |
| `with_style_teacher_score_plan.md` | native with-style teacher derivation | current derivation note |
| `reference_data_and_dmd_inventory.md` | real-reference and DMD inventory | current inventory, update when assets change |
| `reference_kv_score_distillation_spec.md` | old K/V teacher baseline | baseline/specification, not main method |
| `score_residual_distillation_plan.md` | offline student path | optional acceleration path |
| `formal25_teacher_comparison.md` | teacher experiments | current evidence |
| `full_paper_draft.md` | paper draft | current narrative |
| `paper_story_goal_carrier_flow.md` | motivation/story blueprint | current narrative support |

## Historical or Exploratory Notes

These files are useful for context, but they do not override the current main
method or the handoff document:

- `adaptation_notes.md`
- `migration_plan.md`
- `technical_plan.md`
- `residual_optimization_plan.md`
- `profile_fields_notes.md`
- `klein9b_probe_plan.md`
- `klein9b_probe_notes.md`

They document earlier adapter, K/V replacement, residual, and layer-probe
ideas. Keep them for provenance; when a command or claim conflicts with the
handoff, use the handoff and current source code.

## Evidence Locations

Results are generated under `../results/` and ignored by Git. Important current
artifacts include:

```text
results/online25_ca_only_multiseed_manifest.csv
results/online25_ca_only_multiseed_metrics.csv
results/online25_ca_only_multiseed_metrics_summary.json
results/formal25_score_carrier_compare_metrics_summary.json
results/formal25_multi_matched_metrics_summary.json
results/native_carrier_goal_imagecode_s65_summary.json
results/native_imagecode_student_s65_holdout_style27_eval_scale_sweep_summary.json
```

Logs for every completed run are under `../agent_runs/`. Large field dumps and
student checkpoints are intentionally not tracked.
