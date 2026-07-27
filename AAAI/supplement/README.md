# Supplement Data and Tables

Status date: 2026-07-23.

This directory contains the package-local quantitative inputs and generated
LaTeX tables used by `../supplement.tex`.

## Source Data

- `data/benchmark_full_metrics.csv` - 40 styles x 4 matched methods.
- `data/benchmark_multiprompt_summary_raw.csv` - five-style prompt extension.
- `data/paramgrid_metrics.csv` - complete parameter-grid observations.
- `data/srsd_results.csv` - rejected generator-native residual diagnostic.
- `data/stylebench_ca_vs_sm.json` - structured VLM diagnostic on hard styles.

Agent and external-model evaluation sources live with their figure packages,
where prompts, raw verdicts, coverage, failures, aggregation scripts, QA, and
plots remain together.

## Generated Tables

- `tables/agent_multidim.tex`
- `tables/benchmark_per_style.tex`
- `tables/external_model_audit.tex`
- `tables/multiprompt.tex`
- `tables/diagnostic_macros.tex`

## Regenerate

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709/AAAI
python supplement/build_tables.py
```

After regeneration, compile `supplement.tex` twice and verify the eight-page
contract. Do not manually change generated values without changing the
package-local source and documenting the provenance.

