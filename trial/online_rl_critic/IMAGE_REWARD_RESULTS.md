# Image-Level Reward Trial Results

## Question

The earlier same-state GRPO trial optimized only the native CA objective. Its
oracle could rank the candidate latent updates correctly for that objective,
but the final image metrics did not improve. This trial asks whether a delayed
image-level style signal can provide a better reward.

All code and generated artifacts remain inside `trial/online_rl_critic/`.
The source implementation under `save/attention_20260426_090948` is untouched.

## Reward Definition

For one fixed `latents_before`, evaluate candidate updates with scales
`{0.5, 1.0, 1.5}`. Decode each candidate with the frozen VAE at low resolution,
then compute frozen CSD style similarity against the reference image:

```text
image_reward_i = 10 * (CSD(candidate_i) - CSD(latents_before))
reward_i = image_reward_i + 0.1 * native_CA_reward_i
```

The CSD model is used only for reward measurement. It receives no gradient and
is not trained. The actual latent update remains the raw AdamW update multiplied
by the selected action scale.

## Runs

All runs use style `27`, prompt `kp0`, seed `64`, 512x512, 6 outer steps, 2
inner iterations, `start-time=1000`, CA-only native teacher, AdamW with zero
weight decay, disabled SLT and AdaIN paths.

| method | CSD style | CLIP-T | DINO style | VGG Gram distance | DINO layout leakage |
|---|---:|---:|---:|---:|---:|
| no image reward | 0.191231 | 0.361378 | 0.040371 | 3.893074 | 0.282008 |
| image reward, default GRPO (`lr=3e-4`) | 0.121565 | 0.358140 | 0.042605 | 3.932587 | 0.279900 |
| image reward oracle | **0.265210** | **0.363667** | **0.046478** | **3.793562** | 0.280858 |
| image reward GRPO (`lr=0.05`) | **0.265210** | **0.363667** | **0.046478** | **3.793562** | 0.280858 |

The high-learning-rate GRPO run selected action `1.5` in all 8 inner stages
and exactly matched the same-state image-reward oracle. The default run selected
`0.5` in all 8 stages despite the measured reward increasing monotonically from
`0.5` to `1.5`.

## Implementation Finding

The original same-state update used a clipped PPO ratio while evaluating every
candidate action once from the same state. In this diagnostic setting, that
ratio has no useful rollout-policy distinction. It was replaced in the trial
with the one-step relative-advantage log-prob surrogate:

```text
L = - mean(log pi(a_i | s) * A_i)
```

where `A_i` is normalized across the three candidate rewards. A short horizon
still needs a larger policy learning rate; `3e-4` was too small to overcome the
random initial action preference, while `0.05` followed the reward ordering.

## Interpretation

This is the first positive evidence in the trial that a frozen image-level
reward can improve the final style metrics, and that the improvement is not
coming from changing the teacher or adding a trained adapter. However, it is
only one style, one prompt, one seed, and six outer steps. It is not evidence
that the method is ready for the main paper.

The useful next experiment is a multi-seed, multi-style sweep with the fixed
relative-advantage update and a small learning-rate grid. The main method stays
unchanged until that sweep confirms that CSD gains do not trade away content or
layout quality.

## Reproduction

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 \
  trial/online_rl_critic/run_image_reward_grpo.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 6 --iters 2 --disable-dynamic-iters --start-time 1000 \
  --style-mode ca_only --image-reward \
  --grpo-lr 0.05 --grpo-entropy-weight 0 \
  --image-reward-weight 1.0 --native-shaping-weight 0.1 \
  --image-reward-scale 10 \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix trial_image_reward_grpo_lr005_s27_kp0_seed64
```

The complete evaluator manifest and metrics are generated under this same
trial directory and are intentionally ignored by git.
