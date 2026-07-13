# Protocol Audit: The `0.68` CSD Baseline

## Finding

The previous image-reward table in `IMAGE_REWARD_RESULTS.md` was not a valid
comparison with the paper baseline. It compared a short `ca_only` smoke against
the formal `score_match` / `score_carrier` protocol.

The CSD evaluator is working. A strict rerun of the formal protocol inside this
trial produced CSD `0.689619`, which matches the existing formal record around
`0.688` and the stored `score_match` record at `0.710720`.

## Apples-to-Apples Difference

| item | formal baseline | invalid image-reward smoke |
|---|---|---|
| style objective | `score_carrier` or `score_match` | `ca_only` |
| outer steps | 25 | 6 |
| inner iterations | 5 with the native dynamic schedule | fixed 2 |
| start time | 1000 | 1000 |
| gradient calibration | `style-grad-rms-target=2.159326726314248e-06` | omitted |
| distance weight | 0.3 | default |
| content weight | 0.4 | default |
| carrier terms | enabled for `score_carrier` | absent |
| CSD style similarity | about 0.688 to 0.711 | 0.122 to 0.265 |

The short smoke is useful only for checking that candidate VAE decoding and
online reward plumbing work. It cannot measure whether image reward improves
the paper method.

## Strict Audit Run

The audit used the trial snapshot and wrote all generated files under
`trial/online_rl_critic/`:

```bash
/home/ubuntu/chengyanli/flux/bin/python3 \
  trial/online_rl_critic/run_image_reward_grpo.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 25 --iters 5 --start-time 1000 \
  --style-mode score_carrier \
  --style-grad-rms-target 2.159326726314248e-06 \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --dist-weight 0.3 --content-weight 0.4 \
  --carrier-temperature 0.1 --carrier-strength 0.75 \
  --carrier-highpass-gain 0 --carrier-loss-weight 0.25 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --disable-grpo \
  --out-prefix audit_formal_score_carrier_s27_kp0_seed64
```

The unified evaluator reports:

| method | CSD style | CLIP-T | DINO style | VGG Gram distance | layout leakage |
|---|---:|---:|---:|---:|---:|
| strict formal score-carrier audit | **0.689619** | 0.337495 | 0.074046 | 2.390118 | 0.280550 |

The reference image, prompt, seed, resolution, CSD checkpoint, and evaluator
are the same as the earlier formal results. The small numerical difference from
the historical `0.68816` is normal for the rerun and confirms the protocol,
rather than indicating a new method result.

## Consequence for the Paper

The previous claim that image-level reward improved CSD from `0.191` to `0.265`
is withdrawn. Those numbers remain diagnostic artifacts, not evidence against
the original method.

Before testing image reward, the next implementation must preserve the formal
`score_carrier` objective and schedule. Any reward comparison must use the same
25/5 protocol and report the original formal baseline in the same manifest.
The initial wrapper only allowed reward candidates for `ca_only`. It has now
been extended to evaluate the formal `score_carrier` candidate objective. No
claim about image-reward gains is made because the formal comparison is still
negative on CSD.

## Formal Image-Reward Results

The candidate reward was then integrated with the exact formal
`score_carrier` objective. All rows use the same style/prompt/seed, 25 outer
steps, dynamic 5-to-2 inner iterations, start time 1000, gradient calibration,
and carrier regularization. The baseline is the strict audit above.

| method | CSD style | CLIP-T | DINO style | VGG Gram distance | layout leakage |
|---|---:|---:|---:|---:|---:|
| formal score-carrier baseline | **0.689619** | 0.337495 | 0.074046 | 2.390120 | 0.280550 |
| image reward GRPO at every stage | 0.644513 | 0.332246 | 0.066161 | 2.330238 | 0.308398 |
| image reward same-state oracle | 0.681706 | 0.341500 | 0.067942 | **2.270237** | 0.298368 |
| image reward GRPO after outer fraction 0.5 | 0.683222 | **0.354927** | **0.074363** | 2.280977 | 0.304994 |

The late-only schedule avoids most of the early-noise damage and improves
content/style auxiliary metrics, but it still does not beat the original CSD
baseline. Even the same-state image-reward oracle remains below baseline CSD,
so the current low-resolution CSD reward is not aligned enough with the final
image trajectory to justify a paper claim.

The implementation is retained as a negative diagnostic: it proves that the
formal score-carrier path can be evaluated with online image reward without
changing the teacher. The main method remains the original score-carrier
objective; no image-reward term is promoted into the paper method.
