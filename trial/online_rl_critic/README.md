# Trial: Online Critic for Real-Time Gradient Stages

This is an isolated, deliberately experimental trial. It must not modify any
file outside this directory. The main method and main runner remain untouched.

## Question

Can each real-time latent-gradient stage choose its own update strength using a
small online critic? The trial treats the existing native teacher gradient as
the action direction and lets a Q critic choose one scalar multiplier:

```text
action scales = {0.5, 1.0, 1.5}
reward = relative decrease of the next inner-stage objective
critic = tiny online Q network trained with one-step TD
```

This is not yet a learned image reward model. It is a cheap signal probe for
whether stage-wise adaptive control has any useful correlation with progress.
The reward is deliberately local and proxy-based; final image metrics remain
the authority.

## What Is Copied

`run_online_rl_critic.py` is a trial-local snapshot of the current online
teacher runner. Only this copy is modified. It imports the repository's
existing `src/` implementation for the teacher, score conversion, carrier,
and losses.

## Run a Bounded Smoke

Run from the repository root so the existing `src/` package resolves:

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 \
  trial/online_rl_critic/run_online_rl_critic.py \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --num-steps 12 --iters 3 --start-time 1000 \
  --style-mode ca_only \
  --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain \
  --out-prefix trial_rl_critic_ca_only_s27_kp0_seed64
```

All trial output is forced into:

```text
trial/online_rl_critic/results/
```

The CSV logs include `rl_action`, `rl_action_scale`, `rl_q_value`,
`rl_reward_prev_transition`, `rl_critic_loss`, and `rl_grad_rms` for every
inner stage. The critic state is diagnostic and does not alter the native
teacher or score equations.

## Interpretation

- Positive reward means the objective decreased from the previous stage.
- A critic loss that falls is only evidence that the proxy transition is being
  fitted, not that the final image improves.
- If actions collapse to one scale, the current proxy may not distinguish
  stages or the epsilon schedule may be too short.
- If action scales oscillate, inspect loss curves and gradient norms before
  concluding that RL is useful.
- Compare against the same seed with the unmodified main runner.

## Safety Boundary

Do not import this trial from the main runner. Do not copy its changes back to
`scripts/`, `src/`, `README.md`, or `docs/`. If the idea survives, port it in a
separate reviewed change after comparing image metrics and compute cost.
