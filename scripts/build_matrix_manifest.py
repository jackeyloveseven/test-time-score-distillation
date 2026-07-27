#!/usr/bin/env python3
"""Build the matched-matrix evaluation manifest from per-run summary JSONs.

Scans results/<prefix>_*_summary.json produced by run_tpami_with_style_teacher.py
and writes a manifest CSV with the exact columns the evaluator expects:
    method, output_path, style_path, prompt, style_id, prompt_id, seed

The prompt text is taken verbatim from each run's summary (`prompt_text`) so the
CLIP-T score is computed against the identical instruction the sampler used.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

REPO = "/home/ubuntu/chengyanli/items/style_residual_distill_20260709"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="matrix")
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--out-csv", default=os.path.join(REPO, "results", "matrix_manifest.csv"))
    args = ap.parse_args()

    pattern = os.path.join(args.results_dir, f"{args.prefix}_*_summary.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[build_matrix_manifest] no summaries match {pattern}", file=sys.stderr)
        return 2

    rows = []
    missing_image = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        img = d.get("image_path")
        if not img or not os.path.exists(img):
            missing_image.append(f)
            continue
        rows.append(
            {
                "method": d["style_mode"],
                "output_path": img,
                "style_path": d["style_path"],
                "prompt": d["prompt_text"],
                "style_id": d["style_id"],
                "prompt_id": d["prompt_id"],
                "seed": d["seed"],
            }
        )

    rows.sort(key=lambda r: (r["method"], str(r["style_id"]), str(r["prompt_id"]), int(r["seed"])))
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["method", "output_path", "style_path", "prompt", "style_id", "prompt_id", "seed"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[build_matrix_manifest] wrote {len(rows)} rows -> {args.out_csv}")
    if missing_image:
        print(f"[build_matrix_manifest] WARNING: {len(missing_image)} runs had no image:", file=sys.stderr)
        for m in missing_image:
            print(f"    {m}", file=sys.stderr)
    # quick coverage report
    from collections import Counter

    per_method = Counter(r["method"] for r in rows)
    print("[build_matrix_manifest] per-method run counts:", dict(per_method))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
