# AAAI 2027 Paper

Status date: 2026-07-27.

This directory contains the anonymous AAAI 2027 submission draft:

**Test-Time Score Distillation: Mask-Free Semantic Style Editing with
Reverse-Process Regularization**

## Canonical Files

- `main.tex` / `main.pdf` - canonical main paper, 9 US-Letter pages (technical
  content p.1-7, references p.8-9).
- `supplement.tex` / `supplement.pdf` - canonical supplement, 11 US-Letter
  pages.
- `checklist.tex` / `checklist.pdf` - standalone reproducibility checklist,
  2 US-Letter pages.
- `references.bib` - bibliography used by `aaai2027.bst`.
- `aaai2027.sty`, `aaai2027.bst` - local author-kit files.
- `figures/` - self-contained active figure packages.
- `supplement/` - package-local quantitative tables and source data.
- `BUILD.md` - compile and validation workflow.
- `SUBMISSION_READINESS.md` - evidence, format, and release gates.

`main_backup_pre_20260721.tex` and `main_v1_kv_frozen.tex` are historical
snapshots, not submission sources.

## Method Position

TSD uses one frozen rectified-flow image+text model. Reference-conditioned and
reference-free branches are evaluated at the same state; their velocity
contrast defines a fixed-energy detached pseudo-target for latent-only
optimization. The standard frozen reverse process is interleaved with these
updates and is interpreted as an empirical inference-time regularizer. No model
weights, adapters, fake-score networks, or generators are trained.

The same operator supports Semantic Style Editing and Global Style Transfer by
changing the reverse-step regime. TSD is training-free of weights but
optimization-based at inference.

## Evidence Surface

- Fixed-seed 40-style matched benchmark and complete per-style table.
- Five-style, four-additional-prompt robustness extension.
- Six-axis primary Agent benchmark with secondary reliability audit and local
  wrong-reference control.
- Fixed five-style matched-input audit against FireRed and Qwen image editors.
- Parameter grids, full qualitative comparison, fake-score negative study, and
  rejected SRSD diagnostic in the supplement.

The seed-65 standard deviations are cross-style dispersion. Full seed-level
uncertainty over seeds 64--66 remains pending.

## Figure State

`python figures/validate_packages.py` validates 15 active figure packages and
all `main.tex`/`supplement.tex` references. The canonical Agent workflow is in
`figures/agent_benchmark_pipeline/`; its package contains prompts, intent,
research, caption, QA, generation script, accepted image, and provenance.

## Page Contract

- Main paper: 9 pages (technical content p.1-7, references p.8-9).
- Supplement: 11 pages.
- Checklist: 2 pages (standalone document, `checklist.pdf`).
- Page size: US Letter.
- No undefined references, missing figures, or overfull boxes in the current
  build.

See `BUILD.md` before changing figures, floats, bibliography spacing, or page
breaks.
