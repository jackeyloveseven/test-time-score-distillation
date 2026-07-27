# Agent Handoff: GoalCarrier Style Transfer

Status date: 2026-07-14. Read `PROJECT_STATE.md` first. This document explains
how to take over the project without hidden context.

## 1. Mission

Build and validate a training-free reference-style transfer method on Klein9B
that treats an image as an instruction. The current paper method, GoalCarrier,
separates:

```text
native score contrast -> visual goal
content-only affinity -> spatial carrier
rectified-flow target -> execution
```

The reference decides what appearance change is requested. The current sample
decides where the change is transported. The sampler executes the result.

## 2. Locations and Ownership

```text
server:      ubuntu-107
server repo: /home/ubuntu/chengyanli/items/style_residual_distill_20260709
Mac clone:   /Users/jackeylove/Documents/New project/style_residual_distill_20260709
```

The server repository is authoritative for code, experiments, compilation, and
commits. The Mac clone is for reading and editing documentation only. Do not
prepare a local CUDA or LaTeX runtime.

Original Hyperion source, read-only:

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

Never edit it, copy adapted files back into it, or run destructive cleanup
there. Reading `sty/` references is allowed.

## 3. Source-of-Truth Order

1. current source code and tests;
2. `docs/PROJECT_STATE.md`;
3. `AAAI/main.tex` for paper prose;
4. `docs/RESULTS_CATALOG.md` and canonical result summaries;
5. this handoff;
6. historical plans and experiment ledgers.

Old notes are provenance, not authority.

## 4. Environments

Formal runtime:

```text
/home/ubuntu/chengyanli/flux/bin/python3
Python 3.12.12
PyTorch 2.8.0+cu128
GPU: NVIDIA H200, 143771 MiB
```

Models and references:

```text
main model:     /mnt/local/klein9b
native teacher: /mnt/local/klein9b-base
style root:     /home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
```

The Mac has no CUDA and currently has no `pdflatex`/`bibtex`; use it for source,
paper editing, and lightweight tests only. Compile and run formal experiments
on `ubuntu-107`.

## 5. Current Method Modes

| mode | implementation meaning | status |
|---|---|---|
| `score_carrier` | native score match + explicit carrier pseudo-target | **paper main method** |
| `score_match` | native score match without carrier | ablation |
| `kv_match` | reference K/V replacement | baseline |
| `ca_only` | detached native image-CFG pseudo-guidance | online ablation |
| `ca_dmd` | CA plus dynamic fake-score correction | extension; not validated as a win |
| `carrier_score` | older detached transport variant | exploratory |

The optional student predicts a teacher field for repeated deployment. It is
not required by GoalCarrier and must not be described as an adapter dependency.

## 6. Implementation Map

```text
src/native_edit_teacher.py
  prepares frozen reference condition and enforces same-state branch metadata

src/flux2_score.py
  exact scheduler sigma lookup; velocity, score, epsilon, and x0 conversion;
  detached pseudo-target construction

src/carrier_score.py
  normalized content-token affinity and reference-independent transport

src/style_objectives.py
  score matching and CA/CA+DMD pseudo-guidance contracts

src/ca_dmd_backend.py
  fake-score backend, conditioning, optimizer, replay, and persistent state

scripts/run_tpami_with_style_teacher.py
  formal sampler, latent optimizer, matched modes, logging, and safety checks

scripts/evaluate_matched_teacher_outputs.py
  CSD, CLIP-T, DINO style, VGG Gram, and DINO layout leakage

scripts/probe_native_edit_teacher.py
scripts/train_score_residual_student.py
scripts/sample_student_images.py
  optional field dump/student path

trial/online_rl_critic/
  isolated reward, GRPO, same-state, and protocol-audit diagnostics
```

## 7. Mathematical Contract

For a shared current state `(z_t,t,p)`:

```text
Delta v_native = v_with_reference - v_without_reference
Delta s_native = -((1 - sigma) / sigma) Delta v_native
```

The implementation must remove the shared `-z_t/sigma` term before applying
detach boundaries. In `score_carrier`, gradients enter through the base branch
`v_0(z_t)` and its clean prediction. The styled branch, affinity, transported
goal, and pseudo-target are detached.

Content carrier:

```text
A_t = softmax(normalize(c_t) normalize(c_t)^T / tau)
T_A(g) = g + lambda_A (A_t g - g)
```

The reference must not enter `A_t`. For `0 <= lambda_A <= 1`, this transport is
a convex token mixer and does not increase the maximum token L2 norm.

For CA+DMD:

```text
g_total = lambda_A T_A(g_native)
        + lambda_DM (score_style - score_fake_phi)
```

`phi` must persist and be refreshed on current fake samples. A reset or frozen
fake model is not a formal DMD experiment.

## 8. Verify Before Work

```bash
ssh ubuntu-107
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
git status --short
/home/ubuntu/chengyanli/flux/bin/python3 -m py_compile src/*.py scripts/*.py
/home/ubuntu/chengyanli/flux/bin/python3 -m unittest -q \
  scripts.test_score_residual_student \
  scripts.test_tpami_with_style_teacher
```

Expected regression result as of 2026-07-14: `53 tests, OK`.

## 9. Formal GoalCarrier Command

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 25 --iters 5 --start-time 1000 \
  --style-mode score_carrier \
  --style-grad-rms-target 2.159326726314248e-06 \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --dist-weight 0.3 --content-weight 0.4 \
  --carrier-temperature 0.1 --carrier-strength 0.75 \
  --carrier-highpass-gain 0 --carrier-loss-weight 0.25 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix formal25_reproduce_score_carrier_s27_kp0_seed64
```

Use a unique prefix. Do not overwrite an evidence run.

## 10. Evaluation Contract

The unified manifest columns are:

```text
method, output_path, style_path, prompt, style_id, prompt_id, seed
```

Example:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/evaluate_matched_teacher_outputs.py \
  --manifest-csv results/formal25_multi_matched_manifest.csv \
  --out-csv results/formal25_multi_matched_metrics.csv \
  --summary-json results/formal25_multi_matched_metrics_summary.json \
  --device cuda
```

Never mix evaluator checkpoints or preprocessing in one comparison. Summary
JSON files store resolved paths and SHA-256 values.

## 11. Current Evidence

Read `RESULTS_CATALOG.md` for all numbers. The bounded conclusions are:

- native same-state score gives a stronger style signal than the K/V baseline
  on the controlled three-pair comparison;
- carrier anchoring improves CSD, Gram, and leakage over score-only on the
  representative pair;
- the three-pair mean has a Gram/layout trade-off and only one seed per pair;
- CA-only has a separate 3x3 stability diagnostic;
- optional image-code conditioning works at field level but does not yet prove
  unseen-style superiority;
- intermediate CSD reward is negative and stays supplementary.

## 12. Paper Workflow

Paper source:

```text
AAAI/main.tex
AAAI/references.bib
AAAI/reproducibility.tex
AAAI/supplement.tex
```

Compile from the server repository root using `AAAI/BUILD.md`. The current
target is 7 technical + 2 reference + 1 checklist pages, plus a separate
3-page supplement. `AAAI/main.tex` is the only prose source; do not maintain a
second full paper in Markdown.

## 13. Next-Agent Priority

P0:

1. matched 3-reference x 3-seed runs for K/V, score match, and GoalCarrier;
2. matched persistent CA+DMD under the same protocol;
3. carrier identity/shuffle/uniform/reference-affinity interventions;
4. human preference and runtime/peak-memory measurement;
5. larger license-documented references.

P1:

- carrier temperature/strength Pareto sweep;
- CSD calibration and confidence intervals;
- official AAAI style conversion;
- layer probes only if the layer-emergence claim returns to the paper.

## 14. Common Failure Modes

- Different latent/timestep branches invalidate the score residual.
- Adding score tensors directly to a velocity scheduler breaks scale.
- Detaching two full scores before shared-state cancellation creates an
  unintended identity gradient.
- A reference-aware carrier reintroduces layout copying.
- `start-time=850` is not the current formal setting.
- Nonzero AdamW decay is an ablation, not the formal default.
- Good field MSE does not establish image quality.
- Noisy-state reward is not a final-trajectory value function.
- The original Hyperion tree is never the working directory.

## 15. Handoff Checklist

- Server and Mac clone are on the same commit.
- `git status --short` is understood on both machines.
- The 53-test regression passes on the server.
- Every new run has a unique prefix, manifest row, and environment record.
- Paper numbers trace to a summary JSON or protocol audit.
- Main, ablation, extension, and negative diagnostic are labeled correctly.
- Paper and supplement compile on the server with the documented assets.
- No file under the original Hyperion path was modified.
