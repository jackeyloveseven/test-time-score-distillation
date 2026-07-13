# Same-State Group and Oracle Result

## Protocol

All rows use style `27`, prompt `kp0`, seed `64`, 512x512, 12 outer steps,
3 fixed inner iterations, `start-time=1000`, CA-only native teacher, AdamW,
zero weight decay, disabled SLT and AdaIN paths.

For every inner stage, the trial evaluates three hypothetical updates from one
identical `latents_before` and one identical timestep. The reward is the
relative decrease of the same CA objective. The oracle selects the candidate
with the largest measured reward; it is a diagnostic upper bound, not a method.

## Image Metrics

| method | CSD style | CLIP-T | DINO style | VGG Gram distance | DINO layout leakage |
|---|---:|---:|---:|---:|---:|
| no same-state control | **0.4620** | 0.3448 | 0.0615 | **3.2275** | **0.2859** |
| same-state GRPO greedy | 0.4536 | 0.3448 | **0.0631** | 3.2319 | 0.2865 |
| same-state GRPO sampled | 0.4419 | **0.3555** | 0.0583 | 3.4833 | 0.2878 |
| same-state reward oracle | 0.4363 | **0.3525** | 0.0595 | 3.3461 | 0.2908 |

The sampled row is the first same-state run before greedy execution was added.

## Reward-Group Diagnostics

The oracle selected the measured best candidate in `30/30` stages. Its mean
selected reward was `0.0139`, identical to the mean best-candidate reward.
Therefore the same-state candidate evaluation and relative ranking are working
as implemented.

## Conclusion

The reward proxy is the bottleneck. Even a same-state oracle that optimizes the
current CA objective does not improve final image-level style metrics. Further
GRPO tuning on this objective is unlikely to solve the problem.

The next reward should include a delayed image-level signal, for example a
low-resolution VAE decode followed by a frozen CSD/style encoder at selected
stages, with the native CA objective used only as a shaping term. A useful
experiment would also normalize rewards by timestep and separate style reward
from content/layout reward.

This entire diagnostic remains isolated under `trial/online_rl_critic/`.
