# Handoff

Updated: 2026-07-27.

Repository root:

```text
/home/ubuntu/chengyanli/items/style_residual_distill_20260709
```

Read `PROJECT_STATE.md` first. This file is the operational continuation guide.
The previous long-form handoff is preserved at
`docs/_archive/2026-07-23-pre-doc-sync/HANDOFF.md`.

## Current Paper State

- Method: Test-Time Score Distillation (TSD), latent-only optimization on one
  frozen rectified-flow image+text model.
- Title: *Test-Time Score Distillation: Mask-Free Semantic Style Editing with
  Reverse-Process Regularization*.
- Target: anonymous AAAI 2027 submission.
- `AAAI/main.pdf`: 9 US-Letter pages (technical content p1-7, references only
  on p8-9).
- `AAAI/supplement.pdf`: 11 US-Letter pages.
- `AAAI/checklist.pdf`: 2 US-Letter pages, the official AAAI-2027 28-item
  reproducibility checklist compiled as a standalone document
  (`AAAI/checklist.tex`), not appended to the supplement.
- Build status: no undefined references, missing figures, or overfull boxes;
  all fonts embedded, 0 Type-3 fonts (fixed 2026-07-27 via matplotlib
  `pdf.fonttype=42`); US-Letter throughout; `aaai2027.sty`/`.bst`
  byte-identical to the official AuthorKit27; no forbidden packages; 39
  references cited (one uncited entry removed from the 40-entry `.bib`).
- Figure validator: 15 active packages pass.

## Current Scientific Story

TSD evaluates reference-conditioned and reference-free frozen branches at the
same state, rms-normalizes their velocity contrast, uses it to construct a
detached clean-state pseudo-target, updates only the current latent, and then
applies the normal frozen reverse step. The paper's central empirical claim is
that this reverse process supplies inference-time regularization that makes a
trained fake-score network unnecessary in the tested single-sample regime.

Two operating modes use the same operator:

- Semantic Style Editing: 25 steps, 5 inner iterations, step size 0.08.
- Global Style Transfer: 8 steps, dynamic inner schedule, step size 0.12.

Do not describe the regularization substitution as an exact theorem. Do not
describe TSD as feed-forward or training-free of computation; it is
training-free of weights and optimization-based at inference.

## Established Evaluation

- Fixed-seed 40-style benchmark at seed 65, with 40/40 CSD wins for TSD in both
  Semantic and Global pairings.
- Five-style, four-additional-prompt robustness extension.
- Six-axis task-aware Agent benchmark with a complete primary judge.
- Secondary judge coverage is 68/80 shared paired cells; malformed responses
  are recorded with no imputation.
- A third, independent judge (Gemma-4) evaluation-robustness analysis is
  complete, including the wrong-reference control.
- Local wrong-reference control validates only reference-specific style
  discrimination.
- Reverse-shield rho-knockout ablation is complete and appears as Fig. 4 in the
  paper; rho=0 collapses all five styles to noise. This ablation is qualitative
  only; no CSD/VLM aggregation was run for it.
- Fixed five-style FireRed/Qwen matched-input audit; this is not a compute-
  matched leaderboard.
- SRSD tested and rejected as an honest null.

## Agent Benchmark Figures

The six-axis radar figure has been removed from the paper; it is no longer
canonical.

### Workflow

`AAAI/figures/agent_benchmark_pipeline/` contains:

- canonical `agent_benchmark_pipeline.png`;
- initial candidate and accepted revision;
- generation and correction prompts;
- scientific intent, design research, target caption, and QA;
- reproducible Kaon GPT Image 2 script;
- canonical generation metadata.

The accepted Stage 3 has exactly six dimensions: Medium, Texture, Palette,
Subject, Task locality, and Quality. The figure is inserted in
`AAAI/supplement.tex` and the footer explicitly states that Agent evaluation is
not a replacement for human evaluation.

## Remaining Validation

Status: **submission-ready**. The only remaining work is human upload
(main-PDF deadline 2026-07-28; code/supplement appendix due 2026-07-31).

### Multi-Seed Benchmark (deferred, future work)

Seed 65 is complete for four matched methods and backs all reported results.
Seeds 64 and 66 are intentionally out of budget for this submission, 320 cells
total; they are deferred to future work, not a submission blocker. The
aggregation contract includes seed variability, style-bootstrap intervals, win
rates, runtime, memory, and forward/backward counts.

### Reverse-Shield Ablation (done)

Reverse-step strengths `{1, 0.5, 0}` under standard and stress settings on five
styles are complete and reported as Fig. 4 in the paper; rho=0 collapses all five
styles to noise. This is a qualitative ablation only; CSD/VLM aggregation was
not run for it.

### Public Judge Replication (done)

The private local third judge is complete on 95 calibration cells. A third,
independent, open-weight judge (Gemma-4) completed the evaluation-robustness
replication, including the wrong-reference control.

## Wrap-up State

The paper is final and pushed. Disk usage is back to normal (~3.8 GB used,
plenty free); no validation is queued for the submission. The multi-seed queue
(`./scripts/run_remaining_validation.sh`, seeds 64/66) is deferred future work,
not a submission blocker, and is not currently running.

## Paper and Figure Commands

```bash
cd AAAI
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
cd ..
python AAAI/figures/validate_packages.py
```

## Non-Negotiable Boundaries

- Do not redo the benchmark.
- Do not alter fixed style IDs, prompts, model, or seed-65 records while filling
  seeds 64 and 66.
- Do not claim seed uncertainty from cross-style SD.
- Do not impute failed Agent cells.
- Do not use radar area as a score.
- Do not imply Agent evaluation is human validation.
- Do not use K/V manipulation as part of TSD.
- Do not overwrite canonical figures until a revision passes package QA.
- Do not store API credentials in figure packages or metadata.

## Canonical Entry Points

- Paper prose: `AAAI/main.tex`, `AAAI/supplement.tex`
- Project state: `docs/PROJECT_STATE.md`
- Remaining work: `docs/todo.md`, `docs/EXPERIMENT_PLAN.md`
- Figure contract: `AAAI/figures/FIGURE_WORKFLOW.md`
- Figure inventory: `AAAI/figures/README.md`
- Paper story: `docs/research/PAPER_PITCH.md`
- Agent validity: `docs/research/VLM_JUDGE_FINDINGS.md`
