# Build and Evidence Notes

## Compile

From this directory:

```bash
python3 /Users/jackeylove/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  /home/ubuntu/chengyanli/items/style_residual_distill_20260709/AAAI/main.tex \
  --output-directory /home/ubuntu/chengyanli/items/style_residual_distill_20260709/AAAI/build
```

The draft uses an ordinary two-column article preamble so it remains portable
without the official AAAI class. Before submission, use the current AAAI style
package and its required author, copyright, and anonymization fields.

## Evidence Sources

- Main method and story: `docs/full_paper_draft.md` and
  `docs/paper_story_goal_carrier_flow.md`.
- Formal score-carrier protocol and metrics:
  `docs/formal25_teacher_comparison.md`.
- Formal correction and image-reward diagnostics:
  `trial/online_rl_critic/PROTOCOL_AUDIT.md`.
- Formal baseline image and logs:
  `trial/online_rl_critic/results/audit_formal_score_carrier_s27_kp0_seed64_score_carrier_style27_kp0_seed64.*`.

## Reproducibility Contract

The paper must report the model roots, reference path, prompt, seed, resolution,
start time, scheduler, inner-iteration schedule, gradient calibration, loss
weights, optimizer, and evaluator checkpoint hashes. The main code and all
paper-side trial artifacts are kept under this repository; the original source
tree is not modified by the trial.
