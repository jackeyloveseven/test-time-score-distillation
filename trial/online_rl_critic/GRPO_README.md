# Trial: GRPO Relative Advantage

This is the second isolated trial inside `trial/online_rl_critic/`. It leaves
the main runner and all files outside this trial unchanged.

## Difference From the First Critic

The first trial trained a Q critic from absolute local-loss changes. This trial
uses a categorical policy over latent update multipliers:

```text
actions = {0.5, 1.0, 1.5}
group reward = relative loss improvement for adjacent gradient transitions
advantage = (reward - group_mean) / (group_std + eps)
policy loss = clipped GRPO/PPO-style objective - entropy bonus
```

The group is a short online window, not parallel rollouts from an identical
state. Therefore the correct label is **GRPO-style relative-advantage trial**,
not a full formal GRPO claim.

## Smoke Command

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 \
  trial/online_rl_critic/run_online_grpo.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 12 --iters 4 --disable-dynamic-iters --start-time 1000 \
  --style-mode ca_only \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix trial_grpo_ca_only_s27_kp0_seed64
```

The copied runner defaults to the trial results directory and rejects output
paths outside this directory. Use a new prefix for every run.

## What to Inspect

The per-stage CSV should contain:

```text
grpo_action_scale
grpo_log_prob
grpo_reward_prev_transition
grpo_policy_loss
grpo_group_mean
grpo_group_std
grpo_advantage
grpo_update_count
```

The corresponding summary records `trial_name=online_grpo_advantage` and the
number of policy updates. Compare against the no-policy run with identical
seed and settings before interpreting the result.
