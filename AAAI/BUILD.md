# Build and Evidence Notes

## Compile

From the repository root on `ubuntu-107`:

```bash
mkdir -p AAAI/build
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory AAAI/build AAAI/main.tex
bibtex AAAI/build/main
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory AAAI/build AAAI/main.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory AAAI/build AAAI/main.tex

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory AAAI/build AAAI/supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory AAAI/build AAAI/supplement.tex
```

The draft uses an ordinary two-column article preamble so it remains portable
without the official AAAI class. Before submission, replace only the preamble
with the official `aaai2026.sty` author-kit setup and recheck the page budget.
Do not ship the article preamble as a conference submission.

The expected main PDF is 10 pages: pages 1--7 technical content, pages 8--9
references, and page 10 the AAAI reproducibility checklist.

## Validation

```bash
pdfinfo AAAI/build/main.pdf | grep -E 'Pages|Page size'
pdftotext -layout AAAI/build/main.pdf AAAI/build/main.txt
grep -nE 'Undefined|Overfull|Underfull|Fatal|Error' AAAI/build/main.log
pdffonts AAAI/build/main.pdf
```

The page size must be US Letter. The warning grep should be empty, all fonts
must be embedded, references must begin on page 8, and the checklist must begin
after the two reference pages. Render pages 1--10 to images before submission
and visually inspect figures, tables, equations, headers, and page breaks.

## Evidence Sources

- Story and method boundary: `docs/paper_story_goal_carrier_flow.md`.
- Formal score-carrier protocol: `docs/formal25_teacher_comparison.md`.
- Reward protocol correction: `trial/online_rl_critic/PROTOCOL_AUDIT.md`.
- Three-pair metrics: `results/formal25_multi_matched_metrics_summary.json`.
- CA-only multi-seed diagnostic: `results/online25_ca_only_multiseed_metrics_summary.json`.
- Strict baseline artifacts:
  `trial/online_rl_critic/results/audit_formal_score_carrier_s27_kp0_seed64_score_carrier_style27_kp0_seed64.*`.

## Reproducibility Contract

Every paper result must report model roots, reference path, prompt, seed,
resolution, start time, scheduler, inner-iteration schedule, gradient
calibration, loss weights, optimizer, and evaluator checkpoint hashes. The
main code and paper-side trial artifacts remain in this repository; the
original Hyperion source tree is not modified.
