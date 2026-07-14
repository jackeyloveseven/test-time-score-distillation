# Style Residual Distillation

This is the isolated code, evidence, and paper workspace for the second style
transfer paper. The current paper method is **GoalCarrier**, an online,
training-free, carrier-anchored image-CFG objective.

Status date: 2026-07-14. Authoritative state: [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md).

## One-Sentence Method

At the same latent and timestep, the native model's with-reference minus
without-reference score defines the visual goal; a detached affinity built only
from the current content state transports that goal; a rectified-flow
clean-state target applies it to the current sample.

```text
native image-CFG residual = what appearance to apply
content-only affinity     = where it should act
flow clean-state target   = how the sampler executes it
```

The main method optimizes the current latent only. It does not train or require
an adapter. The image-code student is optional amortization. CA+DMD is an
implemented extension, but remains outside the headline claim until a
persistent dynamic fake-score experiment wins under a matched protocol.

## Start Here

1. [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md): current decisions, evidence,
   and open gates.
2. [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md): from-zero operational guide.
3. [`docs/README.md`](docs/README.md): status of every document.
4. [`docs/RESULTS_CATALOG.md`](docs/RESULTS_CATALOG.md): canonical numbers and
   artifact locations.
5. [`AAAI/main.tex`](AAAI/main.tex): only source of truth for paper prose.
6. [`AAAI/SUBMISSION_READINESS.md`](AAAI/SUBMISSION_READINESS.md): remaining
   top-tier evidence and format gates.

Mac-specific documentation use is described in
[`docs/LOCAL_MAC_HANDOFF.md`](docs/LOCAL_MAC_HANDOFF.md). The Mac is not an
experiment or compilation environment.

## Repository Boundaries

```text
server host: ubuntu-107
server repo: /home/ubuntu/chengyanli/items/style_residual_distill_20260709
Mac clone:   /Users/jackeylove/Documents/New project/style_residual_distill_20260709
```

The original Hyperion implementation is read-only:

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

Never edit, overwrite, clean, or commit from the original directory. Style
images may be read from its `sty/` directory.

## Method Modes

| mode | role | paper status |
|---|---|---|
| `score_carrier` | differentiable native score match plus content-carrier pseudo-target | **main GoalCarrier method** |
| `score_match` | native score matching without explicit carrier | main ablation |
| `kv_match` | reference K/V replacement | legacy baseline |
| `ca_only` | detached native image-CFG pseudo-guidance | online ablation; has 3x3 multi-seed diagnostic |
| `ca_dmd` | CA direction plus persistent fake-score correction | extension; not a headline result |
| `carrier_score` | older detached transported-score variant | compatibility/exploration |

## Verified Evidence

- Strict representative GoalCarrier audit: CSD `0.689619`.
- Representative comparison: K/V `0.5614`, score match `0.6770`, GoalCarrier
  `0.6882` CSD.
- Three matched reference--prompt pairs: K/V `0.4027` vs. GoalCarrier `0.5662`
  mean CSD; CLIP-T is comparable, while Gram and layout expose a trade-off.
- CA-only nine-run diagnostic: CSD `0.5605 +/- 0.0625` over three references and
  three seeds. It is not a matched GoalCarrier dominance result.
- Intermediate-image CSD reward is negative on the formal protocol and remains
  supplementary; it is not final-image best-of-N.

All exact values and provenance are in `docs/RESULTS_CATALOG.md`.

## Quick Verification on ubuntu-107

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 -m py_compile src/*.py scripts/*.py
/home/ubuntu/chengyanli/flux/bin/python3 -m unittest -q \
  scripts.test_score_residual_student \
  scripts.test_tpami_with_style_teacher
```

Last verified on 2026-07-14: `53 tests, OK` with Python 3.12.12,
PyTorch 2.8.0+cu128, and one NVIDIA H200 143771 MiB GPU.

## Reproduce the Formal GoalCarrier Pair

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

Always use a new output prefix. Generated results are intentionally ignored by
Git; source and documentation are tracked.

## Code Map

```text
src/native_edit_teacher.py             same-state native branches
src/flux2_score.py                     score/velocity/clean-state conversion
src/carrier_score.py                   content-only affinity transport
src/style_objectives.py                score and pseudo-target objectives
src/ca_dmd_backend.py                  persistent fake-score extension
scripts/run_tpami_with_style_teacher.py main online runner
scripts/evaluate_matched_teacher_outputs.py unified image metrics
scripts/probe_native_edit_teacher.py    field dump for optional student
scripts/train_score_residual_student.py optional amortized student
trial/online_rl_critic/                 isolated reward/GRPO diagnostics
AAAI/                                   paper, figures, bibliography, checklist
```

## Current Priority

The next decisive work is the full matched multi-seed matrix for K/V,
score-only, GoalCarrier, and CA+DMD, followed by human preference and runtime
measurement. Do not broaden the paper claim before those gates are complete.
