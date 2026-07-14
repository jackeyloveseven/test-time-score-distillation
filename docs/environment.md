# Environment and Reproducibility

Status date: 2026-07-14. Formal work runs on `ubuntu-107`; the Mac is for
documentation only.

## Server

```text
host: ubuntu-107
OS: Ubuntu, Linux 6.8.0-71-generic, x86_64
repo: /home/ubuntu/chengyanli/items/style_residual_distill_20260709
runtime: /home/ubuntu/chengyanli/flux/bin/python3
Python: 3.12.12
PyTorch: 2.8.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA H200, 143771 MiB
driver: 590.48.01
```

Models and references:

```text
main model:     /mnt/local/klein9b
native teacher: /mnt/local/klein9b-base
style images:   /home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
```

The original source root is read-only:

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

## Verified Startup

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 -m py_compile src/*.py scripts/*.py
/home/ubuntu/chengyanli/flux/bin/python3 -m unittest -q \
  scripts.test_score_residual_student \
  scripts.test_tpami_with_style_teacher
```

Last verified: `53 tests, OK` on 2026-07-14.

## Alternative Server Environments

These passed earlier import-level checks only:

```text
/home/ubuntu/miniconda3/envs/yzh-dmd/bin/python
/home/ubuntu/miniconda3/envs/diffsynth/bin/python
```

Do not mix runtimes inside a matched comparison. The formal default is
`flux/bin/python3`.

## Mac Documentation Environment

```text
macOS: 26.5.1
architecture: arm64
Git: 2.50.1
local docs path:
  /Users/jackeylove/Documents/New project/style_residual_distill_20260709
```

The Mac currently has no `pdflatex` or `bibtex` and has no CUDA target. Do not
install project runtimes or copy the result tree merely for handoff. Read and
edit Markdown/LaTeX there; run tests, experiments, evaluation, and compilation
on `ubuntu-107`.

## Evaluators

The evaluator resolves local server caches for CSD, OpenCLIP, DINOv2, and
VGG19. Every summary JSON records exact resolved paths, SHA-256 values, metric
direction, and preprocessing. A comparison is invalid if a cache or preprocess
changes between rows.

## Reproducibility Record

Every run must record:

```text
code commit and Python environment
model roots and model-loading/offload settings
reference path, prompt text, IDs, seed, and resolution
outer steps, exact timestep, scheduler sigma, and start time
inner iteration schedule and latent optimizer
all goal, carrier, content, distance, and DMD weights
fake-score optimizer/replay/state path when using CA+DMD
metric caches, hashes, and preprocessing
unique output prefix and manifest path
```

The main formal setting is documented in `formal25_teacher_comparison.md` and
`../AAAI/main.tex`.
