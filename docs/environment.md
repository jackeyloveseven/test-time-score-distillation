# Environment and Reproducibility

Status date: 2026-07-27.

## Server

```text
host: ubuntu-107
OS: Ubuntu, Linux 6.8, x86_64
repo: /home/ubuntu/chengyanli/items/style_residual_distill_20260709
formal Python: /home/ubuntu/chengyanli/flux/bin/python3
GPU: NVIDIA H200, shared with foreign jobs
```

Models and references:

```text
main frozen checkpoint: /mnt/local/klein9b
native checkpoint path: /mnt/local/klein9b-base
style references: /home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
```

The original source root is read-only:

```text
/home/ubuntu/chengyanli/items/save/attention_20260426_090948
```

## Runtime Rules

- Use `/home/ubuntu/chengyanli/flux/bin/python3` for matched generation and
  evaluation unless a script documents another fixed environment.
- Do not mix Python environments within one comparison.
- Check `nvidia-smi` before every GPU launch.
- Never terminate or renice foreign GPU processes.
- Long jobs must be resumable, skip-existing, and manifest-backed.

No validation queue is running: the paper is final and pushed, so no generation
or evaluation is scheduled. The multi-seed matrix (seeds 64/66) is deferred
future work, not a submission blocker.

The workspace uses roughly 3.8 GB with ample free space. If any future run needs
room, remove only confirmed disposable caches or owned temporary files; do not
remove results, checkpoints, or foreign data by assumption.

## Basic Verification

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
/home/ubuntu/chengyanli/flux/bin/python3 -m py_compile src/*.py scripts/*.py
python AAAI/figures/validate_packages.py
```

Run targeted tests for changed modules rather than relying on the historical
2026-07-14 `53 tests, OK` record as a current full-suite result.

## Paper Build Environment

The local TeX installation builds the official AAAI 2027 style:

```bash
cd AAAI
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

Expected outputs as of 2026-07-27: main 9 pages (technical content p.1-7,
references p.8-9), supplement 11 pages, and a standalone checklist.pdf
(2 pages), all US Letter.

## Evaluators

The evaluation surface includes:

- CSD for reference-style fidelity;
- CLIP-T for literal prompt alignment;
- DINO patch self-similarity correlation for layout transfer;
- VGG/ArtFID diagnostics where explicitly reported;
- primary and secondary remote Agents;
- a local wrong-reference style-control judge.

Every summary must record evaluator identity, preprocessing, metric direction,
input ordering, failures, and coverage. Persistent malformed Agent responses
are recorded and never imputed.

## Figure Generation

AI schematics use package-local Kaon GPT Image 2 scripts. Credentials are read
from environment variables or protected local configuration and must not be
written into source, prompts, metadata, or user-facing reports. Generation
responses may contain `b64_json`, HTTPS URLs, or `data:` URLs; active scripts
must support all three.

## Reproducibility Record

Every experiment must retain:

```text
code state and Python environment
checkpoint paths and loading/offload settings
reference, prompt, task, style ID, and seed
resolution, reverse schedule, and timestep configuration
inner iteration schedule, optimizer, and step size
normalization and all active objective weights
runtime, peak memory, forward calls, and backward calls
evaluator versions, prompts, ordering, failures, and hashes
unique output prefix and manifest path
```
