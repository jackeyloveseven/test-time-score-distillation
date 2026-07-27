# Build and Validation

Status date: 2026-07-23.

## Compile

Run from the `AAAI/` directory so the author kit, bibliography, tables, and
figure paths resolve exactly as in the current submission.

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709/AAAI

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

Expected outputs:

```text
main.pdf        8 pages, US Letter
supplement.pdf  8 pages, US Letter
```

## Figure-Package Validation

From the repository root:

```bash
python AAAI/figures/validate_packages.py
```

The expected result is 15 active packages with all main/supplement figure
references resolved.

## PDF Validation

```bash
pdfinfo main.pdf | rg '^Pages|^Page size'
pdfinfo supplement.pdf | rg '^Pages|^Page size'
pdffonts main.pdf
pdffonts supplement.pdf
rg -n 'Overfull|Undefined control|undefined references|multiply defined' \
  main.log supplement.log
```

Underfull box warnings are acceptable only after visual inspection confirms
that they do not create incoherent whitespace. Overfull boxes, undefined
references, missing figures, and unembedded fonts must be fixed.

## Visual QA

Render and inspect every page after any float, caption, table, or bibliography
change:

```bash
mkdir -p /tmp/tsd-main-pages /tmp/tsd-supp-pages
pdftoppm -jpeg -r 130 main.pdf /tmp/tsd-main-pages/page
pdftoppm -jpeg -r 130 supplement.pdf /tmp/tsd-supp-pages/page
```

Check figure labels at final size, column balance, page-bottom whitespace,
caption continuity, anonymity, and bibliography legibility.

## Rebuild Tables

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709/AAAI
python supplement/build_tables.py
```

Rebuild tables only from package-local source data. Do not manually edit
generated table values without updating the source file and regeneration path.

## Reproducibility Contract

Every reported experiment must retain matched inputs, seed, method parameters,
runtime/memory/call accounting, evaluator identity, response coverage, and
failure handling. Agent failures are logged without imputation. Fixed-seed
cross-style SD must not be described as seed uncertainty.
