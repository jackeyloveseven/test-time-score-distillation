# Trial: Image-Level Delayed Reward

This trial replaces the local objective reward with a frozen CSD style reward
computed from low-resolution VAE decodes. It remains isolated under
`trial/online_rl_critic/`.

## Reward

For each same-state candidate action:

```text
image_reward_i = scale * (CSD(decoded_candidate_i, reference)
                           - CSD(decoded_before, reference))
reward_i = image_reward_weight * image_reward_i
           + native_shaping_weight * native_CA_reward_i
```

The CSD model is frozen. Candidate images remain in memory; no reward images
are written during the loop.

For same-state groups, the trial uses a relative-advantage log-prob update:
`-mean(log pi(a_i|s) * A_i)`. This preserves the ranking signal when all three
candidate actions are evaluated from one state. The short diagnostic horizon is
usually too short for the default `3e-4` policy learning rate; the documented
positive smoke uses `--grpo-lr 0.05 --grpo-entropy-weight 0`.

## Smoke

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 \
  trial/online_rl_critic/run_image_reward_grpo.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 6 --iters 2 --disable-dynamic-iters --start-time 1000 \
  --style-mode ca_only \
  --image-reward --image-reward-weight 1.0 \
  --native-shaping-weight 0.1 --image-reward-scale 10 \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix trial_image_reward_grpo_s27_kp0_seed64
```

Inspect the CSV fields:

```text
same_state_image_reward_05/10/15
same_state_native_reward_05/10/15
same_state_selected_reward
grpo_advantage
```

This is still an expensive diagnostic. It is not part of the main method until
multi-seed image-level results justify the extra VAE/CSD forwards.

See `IMAGE_REWARD_RESULTS.md` for the baseline, oracle, fixed-GRPO comparison,
and the exact evaluator metrics. See `PROTOCOL_AUDIT.md` for the corrected
formal `score_carrier` comparison; the short CA-only smoke must not be compared
with the paper's approximately `0.68` CSD baseline.

For the formal path, add the original score-carrier arguments from
`PROTOCOL_AUDIT.md` and use `--style-mode score_carrier`. The optional
`--image-reward-start-fraction 0.5` keeps the original update unchanged during
the first half of the noisy trajectory and enables image-reward action
selection only in the later stages.
