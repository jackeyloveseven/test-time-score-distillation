# Formal With-Style Teacher Comparison

Status: current evidence ledger, reviewed 2026-07-14. GoalCarrier
(`score_carrier`) is the paper-facing method. The three-pair table has one seed
per pair and is not matched multi-seed dominance. See `RESULTS_CATALOG.md`.

This note records the native with-style score teacher and the first
carrier-anchored score comparison for the second paper. All code, manifests,
logs, and generated outputs belong to this repository.

## Protocol

- model: `main=/mnt/local/klein9b`, native teacher=`/mnt/local/klein9b-base`
- style: `style_id=27`, reference image `sty/27.png`
- content: `prompt_id=kp0`, `seed=64`, `512x512`
- sampler: 25 outer steps, 5 inner iterations
- loss isolation: SLT disabled, warm-start AdaIN disabled, final AdaIN disabled
- gradient matching: `style-grad-rms-target=2.159326726314248e-06`
- latent optimizer: AdamW (`--latent-optimizer adamw`); `--latent-weight-decay 0`
  isolates the optimizer choice from an explicit latent norm penalty
- evaluation: CSD style similarity, CLIP-T, DINO style similarity, VGG Gram distance, DINO layout leakage

The native teacher uses the same post-Step1 `z_sigma`, timestep, model, and
image IDs for the positive and negative branches. `score_match` and `ca_only`
use the reference-conditioned native edit velocity; `kv_match` is the existing
attention-carrier baseline.

The key schedule discovery is that the style goal is already visible at the
early noisy state. In this runner, `--start-time 1000` produces the first
optimized style timestep `993.686...` and 24 optimized outer steps. The
original `850` setting starts too late for the native score difference to be a
strong teacher. The formal score experiments therefore use `start-time=1000`.

## Results

| method | CSD style sim | CLIP-T | DINO style sim | VGG Gram dist | DINO layout leak |
|---|---:|---:|---:|---:|---:|
| `kv_match` (start 850) | 0.56138 | 0.34651 | 0.06769 | 2.94373 | 0.28934 |
| `score_match` (start 850) | 0.42778 | 0.34322 | 0.06098 | 3.12333 | 0.29593 |
| `ca_only` (start 850) | 0.38698 | 0.34400 | 0.06220 | 3.19797 | 0.30009 |
| `kv_match` (start 1000) | 0.56444 | 0.34586 | 0.06566 | 2.36454 | 0.27491 |
| `score_match` (start 1000, dist=.3, content=.4) | 0.67696 | 0.34758 | 0.07488 | 2.44513 | 0.30182 |
| `score_carrier` (start 1000, dist=.3, content=.4) | 0.68816 | 0.32815 | 0.07267 | 2.29147 | 0.27799 |

The native score teacher is a genuine model-native with-style signal. At the
late `850` start it is weak, while at `1000` it becomes substantially stronger.
Adding the carrier term improves the style metrics and layout relative to
plain score matching for this pair, although CLIP-T is sensitive to the
regularization choice.

The carrier-anchored objective is:

```text
goal_score = score(with-reference) - score(without-reference)
carrier_score = Transport(goal_score, Affinity(content_anchor))
L = L_score(goal_score) + lambda_carrier * L_pseudo(carrier_score)
```

The affinity transport is computed from the post-Step1 content latent and is
reference-independent. The reference only enters the native with-style score,
so the carrier cannot directly copy reference spatial arrangement.

For the three-pair matched validation (`style/prompt = 00/kp1, 13/kp3,
27/kp0`, seed 64, start 1000, same `dist=.3`, `content=.4` protocol), the
means are:

| method | CSD style sim | CLIP-T | DINO style sim | VGG Gram dist | DINO layout leak |
|---|---:|---:|---:|---:|---:|
| `kv_match` | 0.40273 | 0.32863 | 0.04757 | 4.02987 | 0.33518 |
| `score_carrier` | **0.56618** | **0.32962** | **0.06185** | 4.08912 | 0.34791 |

This supports a precise claim: `score_carrier` gives a stronger semantic/style
signal than the K/V replacement baseline on these paired samples, while Gram
distance and layout leakage remain open trade-offs rather than solved claims.
The result is not yet a multi-seed benchmark.

## Latent Optimizer Check

The inner latent update now exposes both `adam` and `adamw`. AdamW is the
default implementation, while `--latent-weight-decay` is reported separately.
The formal protocol uses `--latent-weight-decay 0` because the optimized object
is a latent, not a trainable network, and a nonzero decay can change the
content/style trade-off rather than simply improve optimization.

On the `style27/kp0` pair, the matched `score_carrier` comparison was:

| optimizer | decay | CSD style sim | CLIP-T | DINO style sim | VGG Gram dist | DINO layout leak |
|---|---:|---:|---:|---:|---:|---:|
| Adam | 0 | 0.67643 | 0.31872 | 0.07335 | 2.37964 | 0.30491 |
| AdamW | 0 | 0.67581 | 0.32689 | 0.07002 | 2.47064 | 0.30430 |
| AdamW | 1e-4 | **0.70453** | 0.31981 | 0.06264 | 2.40385 | 0.31243 |
| AdamW | 1e-3 | 0.68131 | 0.31580 | 0.06885 | 2.49848 | 0.30836 |

The three-pair check (`00/kp1`, `13/kp3`, `27/kp0`) gives a more conservative
picture for AdamW with decay `1e-4`: CSD `0.56618 -> 0.56070`, CLIP-T
`0.31768 -> 0.31851`, DINO style `0.06185 -> 0.06490`, Gram `4.08912 ->
4.11729`, and layout leakage `0.34791 -> 0.35168`. We therefore keep AdamW as
the implementation default with zero decay, and treat nonzero decay as an
explicit regularization ablation rather than part of the core teacher claim.

## Reproduction

Run each method from the repository root. The representative command is:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \\
  scripts/run_tpami_with_style_teacher.py \\
  --style-id 27 --prompt-id kp0 --seed 64 \\
  --num-steps 25 --iters 5 --start-time 1000 \\
  --style-mode score_carrier \\
  --style-grad-rms-target 2.159326726314248e-06 \\
  --latent-optimizer adamw --latent-weight-decay 0 \\
  --dist-weight 0.3 --content-weight 0.4 \\
  --carrier-temperature 0.1 --carrier-strength 0.75 \\
  --carrier-highpass-gain 0 --carrier-loss-weight 0.25 \\
  --disable-slt --adain-alpha 0 --disable-final-adain \\
  --out-prefix formal25_reproduce_score_carrier_s27_kp0_seed64
```

For a matched baseline, change only `--style-mode score_carrier` to
`--style-mode kv_match` and remove the carrier-specific arguments. Then build
`results/formal25_multi_matched_manifest.csv` and evaluate it with:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \\
  scripts/evaluate_matched_teacher_outputs.py \\
  --manifest-csv results/formal25_multi_matched_manifest.csv \\
  --out-csv results/formal25_multi_matched_metrics.csv \\
  --summary-json results/formal25_multi_matched_metrics_summary.json \\
  --device cuda
```

## Next experiment

The next required validation is multi-seed evaluation and an explicit
`carrier_loss_weight` sweep. The paper hypothesis remains: the score field
identifies what style change is desired, while the carrier determines where it
can be rendered without copying reference layout.
