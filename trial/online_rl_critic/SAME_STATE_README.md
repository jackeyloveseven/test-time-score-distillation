# Trial: Same-State GRPO Group

This trial addresses the main flaw in the previous GRPO experiment. Instead
of comparing rewards from different timesteps, it saves one `latents_before`
and one raw optimizer update, constructs three hypothetical candidates from
that exact state, and evaluates all candidates at the same timestep:

```text
candidate_i = latents_before + action_scale_i * raw_update
action_scale_i in {0.5, 1.0, 1.5}
reward_i = relative decrease of the same CA objective
advantage_i = (reward_i - group_mean) / (group_std + eps)
```

The policy is updated from this same-state candidate group and then selects the
action that is applied to the actual latent. This is still a one-step
GRPO-style approximation, not a full multi-step rollout or formal GRPO result.

## Smoke

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 \
  trial/online_rl_critic/run_same_state_grpo.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 12 --iters 3 --disable-dynamic-iters --start-time 1000 \
  --style-mode ca_only \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix trial_same_state_grpo_s27_kp0_seed64
```

The extra teacher forwards are intentional. They are the price of obtaining a
reward group whose candidate actions share one latent state and one timestep.
All output-path safety checks from the earlier trial remain active.

Inspect these CSV fields:

```text
same_state_selected_reward
same_state_best_reward
same_state_reward_mean
same_state_reward_std
same_state_candidate_loss_05
same_state_candidate_loss_10
same_state_candidate_loss_15
grpo_advantage
grpo_policy_loss
```
