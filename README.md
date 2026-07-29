# Style Residual Distillation

This is the isolated code, evidence, and paper workspace for the second style
transfer paper. The current paper method is **Test-Time Score Distillation
(TSD)**: a training-free, latent-only style-transfer operator that runs on a
single frozen rectified-flow image+text model.

Status date: 2026-07-27. Authoritative state: [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md).
Paper prose is the only source of truth: [`AAAI/main.tex`](AAAI/main.tex).

> Historical note: this repo began as an online *carrier-anchored score-match*
> line ("GoalCarrier" / `score_carrier`) and a trained fake-score / student
> family. The paper **pivoted** to the frozen test-time operator below; the old
> training-line code and its evidence are kept for provenance but are no longer
> the method. See `docs/research/LOSS_DECOMPOSITION.md` for the pivot record.

## One-Sentence Method

At each frozen reverse step, TSD reads the model's velocity **with** the
reference and **without** it at the same latent and timestep, takes the
rms-normalized contrast as a fixed-magnitude "spear", nudges **only the current
latent** a few inner steps along it, and then takes one ordinary frozen reverse
step. That frozen reverse step is the only regularizer — no weights are trained,
no fake-score network is added, and no mask or segmentation is used.

```text
with - without reference velocity contrast = the style signal (spear)
rms-normalized fixed-magnitude latent step  = how much to apply
one ordinary frozen reverse step            = the only (implicit) regularizer
```

The reverse-step budget is the locality knob: **Semantic Style Editing** (25
reverse steps) preserves the reference composition; **Global Style Transfer** (8
reverse steps) dissolves it into a more independent layout. Both use the same
operator (`--style-mode ca_only`).

## Start Here

1. [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md): shortest authoritative state.
2. [`docs/HANDOFF.md`](docs/HANDOFF.md): operational continuation guide.
3. [`docs/todo.md`](docs/todo.md): remaining work (now: human upload + code appendix).
4. [`docs/research/PAPER_PITCH.md`](docs/research/PAPER_PITCH.md): reviewer-facing story and claim/evidence map.
5. [`AAAI/main.tex`](AAAI/main.tex): only source of truth for paper prose.
6. [`AAAI/SUBMISSION_READINESS.md`](AAAI/SUBMISSION_READINESS.md): submission/format gates.

Older long-form handoffs (Mac guide, results catalog, agent handoff) are archived
under [`docs/_archive/`](docs/_archive/).

## Submission State

Three separate PDFs, all submission-ready and pushed:

- `AAAI/main.pdf` — 9 US-Letter pages (technical content p1–7, references only p8–9).
- `AAAI/supplement.pdf` — 11 pages.
- `AAAI/checklist.pdf` — 2 pages, the official AAAI-2027 28-item reproducibility
  checklist as a standalone document (`AAAI/checklist.tex`).

Format gates all pass: 0 overfull / 0 undefined, all fonts embedded, **Type-3
fonts = 0** (AAAI hard-rejects Type 3 even in graphics), US-Letter,
`aaai2027.sty/.bst` byte-identical to the official AuthorKit27, no forbidden
packages, 39 references cited. Anonymous submission. Main-PDF deadline
2026-07-28; code/supplement appendix due 2026-07-31.

## Repository Boundaries

```text
server repo: style_score/style_residual_distill_20260709
```

The original Hyperion implementation is read-only:

```text
../../test_time_distillation/save/attention_20260426_090948
```

Never edit, overwrite, clean, or commit from the original directory. Style
images may be read from its `sty/` directory.

## Method Modes

`--style-mode` on `scripts/run_tpami_with_style_teacher.py`. Only `ca_only` is
the paper method; the rest are the abandoned training-line and are kept for
provenance/ablation only.

| mode | role | paper status |
|---|---|---|
| `ca_only` | detached same-state image-CFG velocity contrast, latent-only | **the TSD method (Semantic + Global)** |
| `ca_dmd` | CA direction plus a persistent trained fake-score correction | negative study: the fake score harms structured media |
| `score_carrier` | differentiable native score match + content-carrier target | legacy "GoalCarrier"; abandoned pre-pivot line |
| `score_match` | native score matching without explicit carrier | legacy exploration |
| `kv_match` | reference K/V replacement | out of scope by design (never in TSD) |
| `carrier_score` | older detached transported-score variant | legacy compatibility |

## Verified Evidence (paper, seed 65)

- Fixed 40-style benchmark: TSD improves CSD from **0.267 to 0.719** in the
  composition-preserving (Semantic) setting and from **0.247 to 0.615** in the
  composition-dissolving (Global) setting, winning **40/40** paired references in
  both. The reported SD is cross-style dispersion, not random-seed uncertainty.
- Five fixed styles × four additional prompts: **20/20** paired wins.
- Primary six-axis Agent benchmark: 39/40 (Semantic) and 37/40 (Global) preferred.
- Reverse-shield ρ-knockout ablation (Fig. 4): ρ=0 collapses all five styles to
  noise, evidencing the frozen reverse step as the implicit regularizer.
- A third independent judge (Gemma-4) confirms evaluation robustness; a
  wrong-reference control drops the style score 4.21→1.33.
- The fake-score correction (`ca_dmd`) is a documented negative study; SRSD was
  tested and rejected as an honest null.

All exact numbers live in `AAAI/main.tex` and `AAAI/supplement/data/`.

## Quick Verification

Create an isolated environment and install the dependencies first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch wheel matching the local CUDA driver first if needed.
python -m pip install -r requirements.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio
python -m pip install -e third_party/DiffSynth-Studio
hf auth login  # required for gated Black Forest Labs checkpoints
bash scripts/download_models.sh core
```

`core` downloads the frozen generator and teacher from
`black-forest-labs/FLUX.2-klein-9B` and
`black-forest-labs/FLUX.2-klein-base-9B`. `bash scripts/download_models.sh all`
also downloads the optional 4B critic and CLIP scorer. Expected paths:

```text
models/klein9b/       # main generator
models/klein9b-base/  # paper teacher / native editor
models/klein4b-base/  # optional legacy fake critic
models/clip/          # optional CLIP evaluation
```

```bash
cd style_score/style_residual_distill_20260709
python -m py_compile src/*.py scripts/*.py
```

Environment: Python 3.12, PyTorch 2.8.0+cu128, one NVIDIA H200. Each TSD
generation is single-GPU, bf16, ~24 GB VRAM, ~1 min.

Place model weights under `models/` and DiffSynth-Studio under
`third_party/DiffSynth-Studio/` as shown above.

## Reproduce One TSD Output (ca_only)

```bash
PY=python
MODEL_ROOT=models/klein9b-base
# Semantic: --num-steps 25 --pseudo-step-size 0.05 ; Global: --num-steps 8 --pseudo-step-size 0.12
$PY scripts/run_tpami_with_style_teacher.py \
  --style-id 27 --prompt-id kp0 --seed 65 \
  --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 \
  --pseudo-step-size 0.05 --num-steps 25 --iters 5 \
  --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --without-prompt-mode full_style_prompt --pseudo-grad-normalization rms \
  --reverse-step-strength 1 \
  --main-model-root "$MODEL_ROOT" --teacher-model-root "$MODEL_ROOT" \
  --out-prefix tsd_reproduce_ca_only_s27_kp0_seed65
```

Always use a new output prefix. Generated results are intentionally ignored by
Git; source and documentation are tracked.

## Code Map

```text
src/native_edit_teacher.py               same-state native branches (with/without reference)
src/flux2_score.py                       score/velocity/clean-state conversion
src/carrier_score.py                     content-only affinity transport (legacy carrier line)
src/style_objectives.py                  score and pseudo-target objectives
src/ca_dmd_backend.py                    persistent fake-score extension (negative study)
scripts/run_tpami_with_style_teacher.py  main online runner (ca_only = TSD)
scripts/diagnostic_kleinedit_sysprompt_baseline.py  matched native keep/new-composition baselines
scripts/evaluate_matched_teacher_outputs.py         unified image metrics
scripts/probe_native_edit_teacher.py     field dump for the optional (abandoned) student
scripts/train_score_residual_student.py  optional amortized student (abandoned training line)
trial/online_rl_critic/                  isolated reward/GRPO diagnostics
AAAI/                                     paper, figures, bibliography, checklist
```

## Current Priority

The paper is final and pushed. Remaining work is **human**: upload the three
PDFs to the AAAI-2027 site (main-PDF deadline 2026-07-28) and finalize the
code/supplement appendix and license wording (2026-07-31). Multi-seed
generality, human-preference correlation, and multi-backbone replication are
deferred future work, not submission blockers.
