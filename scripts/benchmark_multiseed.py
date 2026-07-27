#!/usr/bin/env python3
"""Inventory, manifest, and statistics for the paired 40-style benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "benchmark_multiseed"
STYLE_DIR = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty")
SEEDS = (64, 65, 66)
STYLE_IDS = tuple(f"{index:02d}" for index in range(40))
PROMPT_ID = "kp0"
PROMPT = "A panda eating bamboo"
METHODS = ("tsd_semantic", "tsd_global", "native_keep", "native_new")
PAIRINGS = {
    "semantic": ("tsd_semantic", "native_keep"),
    "global": ("tsd_global", "native_new"),
}
METRICS = (
    "csd_style_sim",
    "clip_t",
    "dino_layout_leak",
    "csd_plus_csls",
    "dino_style_sim",
    "vgg_gram_dist",
)


def output_path(method: str, style_id: str, seed: int) -> Path:
    if method == "tsd_semantic":
        name = f"diagnostic_q25s_ca_only_style{style_id}_{PROMPT_ID}_seed{seed}.png"
    elif method == "tsd_global":
        name = f"diagnostic_pg_n8_s12_r_ca_only_style{style_id}_{PROMPT_ID}_seed{seed}.png"
    elif method in {"native_keep", "native_new"}:
        variant = "sem" if method == "native_keep" else "gst"
        seed_suffix = "" if seed == 65 else f"_seed{seed}"
        name = f"diagnostic_kleinsys_{variant}_style{style_id}_{PROMPT_ID}{seed_suffix}.png"
    else:
        raise ValueError(f"unknown method: {method}")
    return RESULTS / name


def records() -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        for style_id in STYLE_IDS:
            for seed in SEEDS:
                path = output_path(method, style_id, seed)
                summary = path.with_name(path.stem + "_summary.json")
                rows.append(
                    {
                        "method": method,
                        "style_id": style_id,
                        "prompt_id": PROMPT_ID,
                        "seed": seed,
                        "output_path": str(path),
                        "exists": path.exists(),
                        "summary_path": str(summary),
                        "summary_exists": summary.exists(),
                    }
                )
    return rows


def write_inventory(rows: list[dict[str, object]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / "inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_method = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "expected": len(subset),
            "present": sum(bool(row["exists"]) for row in subset),
            "missing": sum(not bool(row["exists"]) for row in subset),
            "summaries_present": sum(bool(row["summary_exists"]) for row in subset),
        }
    payload = {
        "protocol": {
            "styles": list(STYLE_IDS),
            "seeds": list(SEEDS),
            "prompt_id": PROMPT_ID,
            "prompt": PROMPT,
            "methods": list(METHODS),
            "pairings": PAIRINGS,
        },
        "counts": by_method,
        "complete": all(item["missing"] == 0 for item in by_method.values()),
    }
    (OUTDIR / "inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def write_manifest(rows: list[dict[str, object]], require_complete: bool) -> None:
    missing = [row for row in rows if not row["exists"]]
    if require_complete and missing:
        raise SystemExit(f"benchmark incomplete: {len(missing)} / {len(rows)} cells missing")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTDIR / "manifest.csv"
    fields = ("method", "output_path", "style_path", "prompt", "style_id", "prompt_id", "seed")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row["exists"]:
                continue
            writer.writerow(
                {
                    "method": row["method"],
                    "output_path": row["output_path"],
                    "style_path": str(STYLE_DIR / f"{row['style_id']}.png"),
                    "prompt": PROMPT,
                    "style_id": row["style_id"],
                    "prompt_id": PROMPT_ID,
                    "seed": row["seed"],
                }
            )
    print(f"wrote {manifest_path} with {len(rows) - len(missing)} rows")


def bootstrap_ci(values: np.ndarray, seed: int = 20260723, draws: int = 20000) -> list[float]:
    if values.size == 0:
        return [math.nan, math.nan]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def summarize_values(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else math.nan,
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "ci95_bootstrap": bootstrap_ci(array),
    }


def load_metric_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for metric in METRICS:
            if row.get(metric) not in (None, ""):
                row[metric] = float(row[metric])
    return rows


def runtime_summary() -> dict[str, object]:
    out = {}
    for method in METHODS:
        entries = []
        for style_id in STYLE_IDS:
            for seed in SEEDS:
                path = output_path(method, style_id, seed)
                summary_path = path.with_name(path.stem + "_summary.json")
                if not summary_path.exists():
                    continue
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                entries.append(summary)
        method_summary = {"n": len(entries)}
        for field in (
            "wall_time_inference_s",
            "cuda_peak_memory_gb",
            "dit_forward_calls_total",
            "latent_backward_calls",
        ):
            values = [float(item[field]) for item in entries if item.get(field) is not None]
            method_summary[field] = summarize_values(values)
        out[method] = method_summary
    return out


def aggregate(metrics_path: Path) -> None:
    rows = load_metric_rows(metrics_path)
    observed = {(row["method"], row["style_id"], row["seed"]): row for row in rows}
    report: dict[str, object] = {
        "source": str(metrics_path),
        "expected_cells": len(METHODS) * len(STYLE_IDS) * len(SEEDS),
        "observed_cells": len(observed),
        "complete": len(observed) == len(METHODS) * len(STYLE_IDS) * len(SEEDS),
        "methods": {},
        "pairings": {},
        "runtime": runtime_summary(),
    }

    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        metric_report = {}
        for metric in METRICS:
            valid = [row for row in method_rows if isinstance(row.get(metric), float)]
            cells = [float(row[metric]) for row in valid]
            by_style = defaultdict(list)
            by_seed = defaultdict(list)
            for row in valid:
                by_style[row["style_id"]].append(float(row[metric]))
                by_seed[int(row["seed"])].append(float(row[metric]))
            style_means = [float(np.mean(values)) for values in by_style.values()]
            seed_means = [float(np.mean(values)) for _, values in sorted(by_seed.items())]
            metric_report[metric] = {
                "cell_level": summarize_values(cells),
                "style_macro": summarize_values(style_means),
                "seed_macro": summarize_values(seed_means),
            }
        report["methods"][method] = {
            "n_cells": len(method_rows),
            "n_styles": len({row["style_id"] for row in method_rows}),
            "n_seeds": len({row["seed"] for row in method_rows}),
            "metrics": metric_report,
        }

    for pairing, (ours, baseline) in PAIRINGS.items():
        pair_report = {}
        for metric in METRICS:
            deltas = []
            style_deltas = defaultdict(list)
            for style_id in STYLE_IDS:
                for seed in SEEDS:
                    ours_row = observed.get((ours, style_id, seed))
                    baseline_row = observed.get((baseline, style_id, seed))
                    if not ours_row or not baseline_row:
                        continue
                    if not isinstance(ours_row.get(metric), float) or not isinstance(baseline_row.get(metric), float):
                        continue
                    delta = float(ours_row[metric]) - float(baseline_row[metric])
                    deltas.append(delta)
                    style_deltas[style_id].append(delta)
            style_mean_deltas = [float(np.mean(values)) for values in style_deltas.values()]
            pair_report[metric] = {
                "paired_cells": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else math.nan,
                "cell_win_rate": float(np.mean(np.asarray(deltas) > 0)) if deltas else math.nan,
                "style_mean_win_rate": (
                    float(np.mean(np.asarray(style_mean_deltas) > 0)) if style_mean_deltas else math.nan
                ),
                "style_mean_delta_ci95_bootstrap": bootstrap_ci(np.asarray(style_mean_deltas)),
            }
        report["pairings"][pairing] = pair_report

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTDIR / "aggregate.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    table_path = OUTDIR / "main_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "method",
            "metric",
            "mean",
            "style_sd",
            "ci95_low",
            "ci95_high",
            "seed_sd",
            "n_cells",
            "n_styles",
            "n_seeds",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            method_report = report["methods"][method]
            for metric in ("csd_style_sim", "clip_t", "dino_layout_leak"):
                summary = method_report["metrics"][metric]
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "mean": summary["style_macro"]["mean"],
                        "style_sd": summary["style_macro"]["sd"],
                        "ci95_low": summary["style_macro"]["ci95_bootstrap"][0],
                        "ci95_high": summary["style_macro"]["ci95_bootstrap"][1],
                        "seed_sd": summary["seed_macro"]["sd"],
                        "n_cells": method_report["n_cells"],
                        "n_styles": method_report["n_styles"],
                        "n_seeds": method_report["n_seeds"],
                    }
                )
    print(f"wrote {out_json}")
    print(f"wrote {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--require-complete", action="store_true")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=OUTDIR / "metrics.csv",
    )
    args = parser.parse_args()

    rows = records()
    if args.command == "inventory":
        write_inventory(rows)
    elif args.command == "manifest":
        write_manifest(rows, args.require_complete)
    elif args.command == "aggregate":
        aggregate(args.metrics_csv)


if __name__ == "__main__":
    main()
