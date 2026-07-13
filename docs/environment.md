# Environment and Reproducibility

This file records the environments actually used by the current repository.
For the complete takeover procedure, read `docs/AGENT_HANDOFF.md` first.

## Host and Repository

```text
host: ubuntu-107
repo: /home/ubuntu/chengyanli/items/style_residual_distill_20260709
original read-only reference:
  /home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

The main benchmark uses the bundled project runtime:

```text
/home/ubuntu/chengyanli/flux/bin/python3
```

## Model and Asset Paths

```text
main model:     /mnt/local/klein9b
native teacher: /mnt/local/klein9b-base
style images:   /home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
```

The evaluator also resolves local model caches for CSD, OpenCLIP, DINOv2, and
VGG19. The exact resolved paths and hashes are recorded in each evaluator
summary JSON. Do not replace a cache halfway through a comparison.

## Alternative Environments

The following environments were checked for import-level compatibility:

```text
/home/ubuntu/miniconda3/envs/yzh-dmd/bin/python
/home/ubuntu/miniconda3/envs/diffsynth/bin/python
```

The older `diffsynth` sanity check reported `torch==2.11.0+cu130`; the older
`yzh-dmd` check reported `torch==2.10.0+cu128`. These are compatibility notes,
not permission to mix runtimes in one experiment. Use `flux/bin/python3` for
the current formal protocol unless a new comparison explicitly records the
environment.

## Startup Checks

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 -m py_compile src/*.py scripts/*.py
/home/ubuntu/chengyanli/flux/bin/python3 -m unittest -q \
  scripts.test_score_residual_student \
  scripts.test_tpami_with_style_teacher
```

If a runtime import fails, check the interpreter first, then the model paths,
then CUDA visibility. Do not copy dependencies into the original source tree.

## Reproducibility Contract

Record all of the following in a manifest or JSON summary:

```text
model roots
style root and exact reference path
prompt text and prompt ID
seed, resolution, number of steps, start time
inner iterations and optimizer
all style/carrier/DMD weights
fake-score backend and replay-state path, if applicable
metric model cache paths and hashes
```

The main CA-only validation uses:

```text
styles/prompts: 00/kp1, 13/kp3, 27/kp0
seeds:          64, 65, 66
steps/iters:    25 / 5
start-time:     1000
optimizer:      AdamW, weight decay 0
SLT/AdaIN:      disabled
```
