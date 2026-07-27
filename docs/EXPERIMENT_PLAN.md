# Remaining Experiment Plan

Status date: 2026-07-23.

The benchmark itself is complete and should not be redesigned. This plan covers
only unfinished robustness and mechanism validation.

## E1: Paired Multi-Seed Completion - P0

**Question:** Does the fixed-seed 40-style advantage persist across seeds
64--66 under a strictly paired protocol?

**Matrix:**

- methods: TSD Semantic, TSD Global, native keep-composition, native
  new-composition;
- styles: all 40 fixed references;
- seeds: 64, 65, 66;
- prompt: the existing matched benchmark prompt;
- current coverage: seed 65 complete for all four methods;
- remaining coverage: seeds 64 and 66, 320 cells.

**Outputs:** style-bootstrap 95% confidence intervals, seed-level SD,
cell/style win rates, runtime, peak memory, forward calls, backward calls, and
machine-readable manifests.

**Scripts:**

- `scripts/run_benchmark_multiseed.sh`
- `scripts/benchmark_multiseed.py`

**Integrity constraints:** do not change model, prompt, style set, scheduler,
resolution, or evaluator while filling missing cells. Do not mix the resulting
seed uncertainty with the existing cross-style SD.

## E2: Reverse-Shield Ablation - P0

**Question:** Is the interleaved frozen reverse step empirically responsible
for regularizing the bounded test-time update?

**Protocol:** reverse-step strength `{1, 0.5, 0}` under standard and stress
settings on five preselected styles. Strength 1 is the unchanged method.

**Outputs:** matched image plate, metric table, artifact diagnostics, local VLM
ratings, and explicit failure cases. Interpret this as empirical mechanism
evidence, not proof of exact projection or distribution matching.

**Scripts:**

- `scripts/run_reverse_shield_ablation.sh`
- `scripts/reverse_shield_ablation.py`

## E3: Public Open-Weight Judge Replication - P1

**Question:** Does an independently downloadable local VLM reproduce the
reference-specific discrimination and broad agreement trends?

The existing private local judge already covers 95/95 calibration cells and
passes the wrong-reference control. The remaining task is a public-open-weight
replication when GPU headroom allows. It must remain an audit, not a source of
the reported primary means.

## Explicitly Out of Scope

- Rebuilding or redefining the benchmark.
- A compute-matched external leaderboard.
- Human evaluation in the current validation budget.
- K/V or attention-surgery exact-motif transfer.
- A feed-forward student.

## Execution Order

1. E1 paired multi-seed completion.
2. E2 reverse-shield ablation.
3. Re-aggregate and reconcile paper claims.
4. E3 public-open-weight judge replication when resources permit.

