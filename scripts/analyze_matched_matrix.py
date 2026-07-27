#!/usr/bin/env python3
"""Aggregate the matched-matrix metrics into paper-facing statistics.

Reads the per-image metrics CSV produced by evaluate_matched_teacher_outputs.py
and reports, per method over all (style,prompt,seed) cells:
  - mean / std of every metric
  - 95% bootstrap CI on mean CSD
  - paired win-rate vs kv_match on CSD (per matched cell)

No external deps beyond the stdlib. Deterministic bootstrap (fixed seed) so the
CI is reproducible across reruns.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict

REPO = "/home/ubuntu/chengyanli/items/style_residual_distill_20260709"
METRICS = ["csd_style_sim", "clip_t", "dino_style_sim", "vgg_gram_dist", "dino_layout_leak"]
HIGHER_BETTER = {"csd_style_sim", "clip_t", "dino_style_sim"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pick_metric_key(fieldnames, wanted):
    # tolerate minor column-name variants
    lut = {k.lower(): k for k in fieldnames}
    if wanted in lut:
        return lut[wanted]
    for k in fieldnames:
        if wanted.split("_")[0] in k.lower():
            return k
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def bootstrap_ci(xs, n=10000, alpha=0.05):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(1234)
    means = []
    k = len(xs)
    for _ in range(n):
        s = [xs[rng.randrange(k)] for _ in range(k)]
        means.append(sum(s) / k)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default=os.path.join(REPO, "results", "matrix_metrics.csv"))
    ap.add_argument("--out-json", default=os.path.join(REPO, "results", "matrix_aggregate.json"))
    ap.add_argument("--baseline", default="kv_match", help="primary baseline for the printed column")
    ap.add_argument("--expect-cells", type=int, default=9, help="expected (style,prompt,seed) cells per method")
    args = ap.parse_args()

    rows = load_rows(args.metrics_csv)
    if not rows:
        print("[analyze] empty metrics csv")
        return 2
    fn = rows[0].keys()
    mkey = {m: pick_metric_key(fn, m) for m in METRICS}
    method_key = pick_metric_key(fn, "method") or "method"

    # index: cell = (style_id, prompt_id, seed) -> method -> row
    def cell_of(r):
        return (r.get("style_id"), r.get("prompt_id"), r.get("seed"))

    by_method = defaultdict(list)
    by_cell = defaultdict(dict)
    for r in rows:
        m = r[method_key]
        by_method[m].append(r)
        by_cell[cell_of(r)][m] = r

    agg = {"n_cells": len(by_cell), "methods": {}}
    for m, mrows in sorted(by_method.items()):
        entry = {"n": len(mrows)}
        for metric, key in mkey.items():
            if key is None:
                continue
            vals = [_f(r.get(key)) for r in mrows]
            entry[metric] = {"mean": mean(vals), "std": std(vals)}
        csd_vals = [_f(r.get(mkey["csd_style_sim"])) for r in mrows] if mkey["csd_style_sim"] else []
        entry["csd_style_sim"]["ci95"] = bootstrap_ci(csd_vals)
        agg["methods"][m] = entry

    # paired win-rate on CSD vs every baseline present (kv_match = generic prior
    # art; score_match = the differentiation ablation -- the headline stat under
    # gen-2 framing). Compute both so the money comparison is never missing.
    ckey = mkey["csd_style_sim"]
    baselines = [b for b in ("kv_match", "score_match") if b in by_method]
    if ckey:
        for base in baselines:
            for m in by_method:
                if m == base:
                    continue
                wins = ties = losses = paired = 0
                deltas = []
                for cell, mm in by_cell.items():
                    if base in mm and m in mm:
                        a = _f(mm[m].get(ckey))
                        b = _f(mm[base].get(ckey))
                        if a is None or b is None:
                            continue
                        paired += 1
                        deltas.append(a - b)
                        if a > b:
                            wins += 1
                        elif a < b:
                            losses += 1
                        else:
                            ties += 1
                agg["methods"].setdefault(m, {})["vs_%s_csd" % base] = {
                    "paired_cells": paired,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": (wins / paired) if paired else float("nan"),
                    "mean_delta": mean(deltas),
                }

    # completeness gate: every method must cover args.expect_cells cells
    coverage = {m: e["n"] for m, e in agg["methods"].items()}
    incomplete = {m: n for m, n in coverage.items() if n < args.expect_cells}
    agg["coverage"] = coverage
    agg["expect_cells"] = args.expect_cells
    agg["coverage_ok"] = not incomplete

    with open(args.out_json, "w") as fh:
        json.dump(agg, fh, indent=2)

    # markdown table
    def wr_str(e, base, m):
        if m == base:
            return "--"
        wr = e.get("vs_%s_csd" % base, {}).get("win_rate")
        return f"{wr:.2f}" if isinstance(wr, float) and not math.isnan(wr) else "n/a"

    print("\n### Matched matrix aggregate (%d cells/method)\n" % agg["n_cells"])
    print("| method | CSD mean | CSD std | CSD 95%CI | CLIP-T | DINO | Gram | Leak | win/kv | win/score_match |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for m, e in sorted(agg["methods"].items()):
        ci = e.get("csd_style_sim", {}).get("ci95", (float("nan"), float("nan")))

        def g(metric):
            return e.get(metric, {}).get("mean", float("nan"))

        print(
            f"| {m} | {g('csd_style_sim'):.4f} | {e.get('csd_style_sim',{}).get('std',float('nan')):.4f} | "
            f"[{ci[0]:.4f},{ci[1]:.4f}] | {g('clip_t'):.4f} | {g('dino_style_sim'):.4f} | "
            f"{g('vgg_gram_dist'):.4f} | {g('dino_layout_leak'):.4f} | "
            f"{wr_str(e,'kv_match',m)} | {wr_str(e,'score_match',m)} |"
        )

    # completeness gate (loud)
    print("\n[analyze] coverage per method (expect %d): %s" % (args.expect_cells, agg["coverage"]))
    if agg["coverage_ok"]:
        print("[analyze] COVERAGE OK -- every method has all cells")
    else:
        bad = {m: n for m, n in agg["coverage"].items() if n < args.expect_cells}
        print("[analyze] !!! COVERAGE INCOMPLETE -- rerun missing cells before trusting: %s" % bad)
    print(f"\n[analyze] wrote {args.out_json}")
    return 0 if agg["coverage_ok"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
