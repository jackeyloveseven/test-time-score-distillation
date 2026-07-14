# GRPO-Style Trial Result

> **Status (2026-07-14): supplementary negative result.** Do not compare these
> values to the headline number without the protocol distinctions documented
> in `PROTOCOL_AUDIT.md`.

## Protocol

Both runs use style `27`, prompt `kp0`, seed `64`, 512x512, 12 outer steps,
4 fixed inner iterations, `start-time=1000`, CA-only native teacher, AdamW,
zero weight decay, SLT disabled, and AdaIN paths disabled. The only intended
difference is the GRPO-style action policy.

The policy samples a scalar latent update multiplier from `{0.5, 1.0, 1.5}`.
Rewards are relative local objective improvements and are normalized within a
rolling group of four transitions:

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
L = -mean(min(ratio*A, clip(ratio)*A)) - entropy_weight*H(policy)
```

This is a stage-window approximation. The group members are not parallel
rollouts from exactly the same latent, so this must not be called formal GRPO.

## Runtime Check

The policy generated 40 stage rows, accumulated relative advantages, and made
7 clipped policy updates. The action distribution was nontrivial and shifted
over the trajectory. Group standard deviations were nonzero for the update
groups, so the relative normalization was active.

## Image Metrics

| method | CSD style | CLIP-T | DINO style | VGG Gram distance | DINO layout leakage |
|---|---:|---:|---:|---:|---:|
| no GRPO | **0.5488** | 0.3376 | 0.0533 | **2.9517** | **0.2869** |
| GRPO-style advantage | 0.5352 | **0.3422** | 0.0526 | 3.1679 | 0.2898 |

## Conclusion

Relative advantage provides a better learning signal than the previous raw
local-loss critic in the narrow sense that it produces nonzero group-normalized
advantages and updates the action policy. It does **not** improve this image
sample. The likely remaining issue is not only reward scale: the group mixes
different noise levels and different latent states, so its ranking is not a
valid same-state action comparison.

This trial should remain isolated. The next technically meaningful version
would evaluate all candidate action scales from the same saved latent state,
obtain a same-state reward group, and only then apply the selected action.
That costs extra teacher forwards but would make the relative advantage
well-defined. A second option is to use decoded low-resolution style rewards at
selected stages, with timestep-normalized native loss only as shaping.
