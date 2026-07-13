# Trial Result: Online Critic

## Protocol

The critic and no-critic runs use the same style `27`, prompt `kp0`, seed `64`,
512x512 resolution, 12 outer steps, 3 inner iterations, `start-time=1000`,
CA-only native teacher, AdamW, zero weight decay, disabled SLT, and disabled
AdaIN paths. The only intended difference is the stage-wise scalar action
multiplier.

The critic selects from `{0.5, 1.0, 1.5}` and receives a delayed local reward:

```text
reward_t = clip((total_loss_{t-1} - total_loss_t) / (abs(total_loss_{t-1}) + eps), -1, 1)
```

## Runtime Check

The critic was active for 20 inner-stage rows and performed 10 TD updates. The
observed action sequence was nontrivial:

```text
1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.0,
0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5
```

This confirms that the online critic is connected to each real-time gradient
stage and changes the applied latent update magnitude.

## Image Metrics

| method | CSD style | CLIP-T | DINO style | VGG Gram distance | DINO layout leakage |
|---|---:|---:|---:|---:|---:|
| no critic | **0.3725** | **0.3525** | **0.0621** | **3.5612** | **0.2870** |
| online critic | 0.2891 | 0.3552 | 0.0637 | 3.7680 | 0.2872 |

## Conclusion

This smoke is a **negative result for the current reward design**. The critic
is technically online and stage-wise, but a decrease in the optimization loss
is not a reliable proxy for final reference style quality. The critic should
not be added to the main method or described as an image reward model.

The likely issue is credit assignment: the reward is measured before the next
teacher evaluation and only sees local loss progress. It has no direct access
to decoded image style, prompt alignment, or layout leakage.

## Possible Follow-Up Inside This Trial

1. Decode a low-resolution preview only every `K` stages and use CSD/CLIP-T as
   a delayed reward, with the native loss as a shaping term.
2. Replace the scalar action with a bounded residual blend between CA and DMD
   score components, rather than scaling the whole latent update.
3. Compare reward normalization across timesteps; current loss magnitudes vary
   substantially with sigma.
4. Keep a frozen reference image encoder and train the critic on image/style
   features, not only scalar optimization losses.

All artifacts for this trial are under this directory. No file outside the
trial directory was modified by the trial implementation or its outputs.
