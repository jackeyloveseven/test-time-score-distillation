# Local Mac Documentation Handoff

The Mac is a documentation and paper-editing workspace only.

## Location

```text
/Users/jackeylove/Documents/New project/style_residual_distill_20260709
```

The server remains authoritative for code execution, experiments, LaTeX
compilation, generated results, and Git commits:

```text
ubuntu-107:/home/ubuntu/chengyanli/items/style_residual_distill_20260709
```

## What to Read Locally

1. `README.md`
2. `docs/PROJECT_STATE.md`
3. `docs/AGENT_HANDOFF.md`
4. `docs/RESULTS_CATALOG.md`
5. `AAAI/main.tex`
6. `AAAI/SUBMISSION_READINESS.md`

## Local Rules

- Edit Markdown and LaTeX only.
- Do not install CUDA/model environments for this project.
- Do not copy the 1.7 GB server result tree.
- Do not treat absent local result images as missing server evidence.
- Run tests, image generation, metrics, and PDF compilation on `ubuntu-107`.
- Commit the final synchronized version on the server.

## Local Organization

Older Mac-only patches, temporary GRPO files, duplicate paper sources, and
rendered QA images were preserved under:

```text
/Users/jackeylove/Documents/New project/local_archive/
  style_residual_distill_legacy_20260714/
```

They are archival provenance, not current source. Other projects in
`/Users/jackeylove/Documents/New project/` were not changed.

## Refresh Documentation

When the local working tree is clean, refresh the tracked documentation from
the authoritative server commit with:

```bash
cd "/Users/jackeylove/Documents/New project/style_residual_distill_20260709"
git pull --ff-only
```

If local documentation has uncommitted edits, preserve them as a patch before
refreshing and apply/commit the final version on the server.
