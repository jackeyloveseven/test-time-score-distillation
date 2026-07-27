# Agent Benchmark and Judge Validity Findings

Status date: 2026-07-23.

This document consolidates the current task-aware Agent benchmark. Earlier
score-match calibration findings are historical and remain in archived logs;
the paper-facing evaluation is the six-axis TSD benchmark below.

## Rubric

Every anonymous candidate is scored from 1 to 5 on six independent axes:

1. Medium
2. Texture
3. Palette
4. Subject
5. Task locality
6. Visual quality

Task locality is mode-specific: preserve reference composition for Semantic
Style Editing and form an independent composition for Global Style Transfer.
Radar area is never used as an aggregate score.

## Primary Agent

The complete primary Agent evaluates all 40 Semantic and 40 Global paired
cells and supplies the reported means. It prefers TSD in 39/40 Semantic pairs
and 37/40 Global pairs. TSD's largest gains are on medium, texture, and palette;
subject fidelity decreases modestly, and locality moves in opposite directions
by task as intended.

## Secondary Reliability Audit

The secondary Agent independently returns valid structured responses for 68/80
paired cells. Twelve cells remain empty or malformed after retry. These
failures are reported and never imputed. The secondary model does not alter the
primary means.

On the 68 shared cells:

- preferred-method agreement: 68/68;
- Pearson correlation: 0.957 medium, 0.960 texture, 0.977 palette, 0.925 task
  locality;
- visual quality is less rank-stable at `r=0.431`, so no key claim depends on
  that axis.

## Local Wrong-Reference Control

A separate local Gemma-4-family judge evaluates reference-specific style
discrimination with correct versus wrong references. It covers the style axis
only and must not be described as a third complete six-axis judge. The control
supports the claim that the benchmark reads reference-specific style rather
than generic image quality.

## Public-Editor Audit

A fixed five-style audit compares TSD, native MMDiT, FireRed Image Edit, and
Qwen Image Edit under matched references, requests, seeds, and task
instructions. It is a matched-input perceptual stress test, not a compute- or
architecture-matched leaderboard.

TSD is strongest on the three style axes in the reported audit. The native edit
remains strongest at Global composition independence, and TSD has lower subject
scores in some cells. Visual-quality correlation across Agents is unstable, so
the external-model conclusion is based on explicit style and locality axes,
not an aggregate quality score.

## Workflow Controls

- Deterministic candidate ordering and hidden method names.
- Schema validation and retry for malformed responses.
- Persistent failure logging with no imputation.
- Separate primary means, secondary agreement audit, and local style control.
- Bootstrap intervals and raw JSON/CSV logs.
- Explicit statement that the benchmark is a perceptual diagnostic, not a
  replacement for human evaluation.

The canonical workflow figure and complete provenance are in
`AAAI/figures/agent_benchmark_pipeline/`.

## Remaining Validity Work

- Replicate the local control with a publicly downloadable open-weight VLM.
- Human correlation is outside the current validation scope.
- Do not promote the current Agent results to consensus truth or automatic
  ground truth.

