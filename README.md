# Style Residual Distillation

This repository is the complete, isolated workspace for the second style
transfer paper. It is designed so that a new agent can start from this file
and then follow [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md).

## Current Paper Position

The main method is **online carrier-anchored native score refinement** for
image-conditioned style transfer. At each denoising state, the native Klein9B
teacher evaluates the same latent with and without a real reference image:

```text
native with-style score - native without-style score = style goal
content-derived affinity transport                         = spatial carrier
rectified-flow scheduler                                   = flow executor
```

The main `ca_only` protocol runs directly at inference time. It does not
require offline student training or an adapter. The offline image-conditioned
student is an optional latency/amortization ablation. `ca_dmd` and
`score_carrier` are experimental extensions and must not be described as the
main result until their fake-score protocol is fully validated.

## Start Here

1. Read [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md) for the full takeover
   guide, environment paths, data contracts, commands, results, and next tasks.
2. Read [`docs/README.md`](docs/README.md) for the document map and which notes
   are current versus historical.
3. Read [`docs/full_paper_draft.md`](docs/full_paper_draft.md) for the current
   paper narrative and claim boundary.
4. Read [`docs/formal25_teacher_comparison.md`](docs/formal25_teacher_comparison.md)
   for native teacher and carrier comparison results.

## Repository Boundary

All implementation, documentation, logs, manifests, and generated results for
this paper belong under:

```text
/home/ubuntu/chengyanli/items/style_residual_distill_20260709
```

The original Hyperion implementation is read-only reference material:

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

Never edit, overwrite, or run destructive cleanup against the original source.
The code in this repository is intentionally adapted and isolated.

## Quick Verification

Use the fixed runtime used for the benchmark:

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 -m py_compile src/*.py scripts/*.py
/home/ubuntu/chengyanli/flux/bin/python3 -m unittest -q \
  scripts.test_score_residual_student \
  scripts.test_tpami_with_style_teacher
```

The last verified regression was `53 tests, OK`.

## Main Online Smoke

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 25 --iters 5 --start-time 1000 \
  --style-mode ca_only \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix online_ca_only_formal25_s27_kp0_seed64
```

The nine-run validation uses styles/prompts `00/kp1`, `13/kp3`, `27/kp0` and
seeds `64,65,66`. Its metric summary is:

```text
CSD style similarity: 0.5605 +/- 0.0662
CLIP-T:               0.3287 +/- 0.0171
DINO style similarity:0.0602 +/- 0.0225
VGG Gram distance:    3.9673 +/- 1.3332
DINO layout leakage: 0.3444 +/- 0.0363
```

The manifest and summary are under `results/online25_ca_only_multiseed_*`.

## Code Map

```text
src/native_edit_teacher.py       native with-style / without-style field
src/flux2_score.py               score <-> velocity conversion
src/carrier_score.py             content-affinity carrier transport
src/ca_dmd_backend.py             online CA+DMD fake-score backend
src/score_residual_student.py    optional offline student architecture
src/score_student_runtime.py      student inference and target conversion
scripts/run_tpami_with_style_teacher.py  main online sampler
scripts/probe_native_edit_teacher.py      native field dump
scripts/train_score_residual_student.py   optional student training
scripts/sample_student_images.py          optional student sampling
scripts/evaluate_matched_teacher_outputs.py image metrics
scripts/test_*.py                          regression tests
```

Generated files are ignored by Git by design. Commit source and documentation,
not large checkpoints, images, logs, or field dumps.

## Current Next Step

The next experiment is a matched multi-seed comparison of `score_carrier` and
`ca_dmd`, including a `carrier_loss_weight` sweep and explicit fake-score state
logging. Until that is complete, the paper should claim a validated online
CA-only method and a promising carrier-anchored score decomposition, not a
formal DMD result or universal image-quality dominance.
