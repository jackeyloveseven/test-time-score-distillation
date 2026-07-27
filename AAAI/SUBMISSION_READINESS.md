# Submission Readiness Audit

Status date: 2026-07-23.

## Current Verdict

The paper is structurally submission-ready and empirically strong at fixed
seed, but two non-human validation runs remain: paired multi-seed completion and
the reverse-shield ablation. Human evaluation is explicitly outside the current
scope.

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
- Main and supplement both fit 8 US-Letter pages.
- Large blank regions removed through float and caption optimization.
- 15 active self-contained figure packages validate.
- Canonical TSD framework, six-axis radar, external-model audit, and Agent
  benchmark workflow are wired into the paper/supplement.
- Current builds have no undefined references, missing figures, or overfull
  boxes.

## Remaining P0 Evidence Gates

1. Complete seeds 64 and 66 for the four-method, 40-style paired matrix and
   report seed-level uncertainty separately from cross-style dispersion.
2. Complete the reverse-step-strength ablation `{1, 0.5, 0}` under standard and
   stress settings and bound the resulting mechanism claim.
3. Reconcile Abstract, Introduction, Results, and Conclusion if either run
   materially changes current conclusions.

## Remaining P1 Submission Gates

- Public-open-weight replication of the local third-judge protocol.
- Code/data release manifest, evaluator hashes, and redistribution/license
  notes.
- Final Data Availability and Code Availability wording.
- Final anonymity, PDF metadata, font embedding, filename, and upload checks.
- Final reviewer-style adversarial read.

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

- `main.tex`, `supplement.tex`
- `supplement/data/`
- `figures/agent_multidim_radar/`
- `figures/external_model_audit/`
- `figures/open_judge_validation/`
- `figures/agent_benchmark_pipeline/`
- manifests and summaries produced by the remaining validation scripts
