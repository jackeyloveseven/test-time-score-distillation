"""Rank highpass teacher lambdas from pairwise residual diagnostics.

Input is a ``*_pairwise.csv`` produced by ``probe_klein9b_residuals.py`` or
``postprocess_teacher_sweep.py``. The script selects rows where:

    variant_a == teacher_hp_lam*
    variant_b == otcfg

and ranks lambdas using a conservative score:

    score = cosine_mean
            - diff_weight * normalized_diff_norm_mean
            - ratio_weight * abs(log(norm_ratio_mean))
            - stability_weight * cosine_std

Higher score is better.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank teacher_hp_lam* variants from pairwise CSV.")
    parser.add_argument("--pairwise-csv", required=True, help="Input pairwise CSV.")
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix. Defaults to input path without _pairwise.csv suffix.",
    )
    parser.add_argument("--target", default="otcfg", help="Target variant, default otcfg.")
    parser.add_argument("--teacher-prefix", default="teacher_hp_lam", help="Teacher variant prefix.")
    parser.add_argument(
        "--diff-weight",
        type=float,
        default=0.25,
        help="Penalty weight for diff_norm normalized by max teacher diff_norm.",
    )
    parser.add_argument(
        "--ratio-weight",
        type=float,
        default=0.15,
        help="Penalty weight for abs(log(norm_ratio_b_over_a_mean)).",
    )
    parser.add_argument(
        "--stability-weight",
        type=float,
        default=0.10,
        help="Penalty weight for cosine std across samples.",
    )
    return parser.parse_args()


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return float("nan")
    return float(value)


def _std(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    return pvariance(vals) ** 0.5


def _mean(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return mean(vals)


def _lambda_sort_key(name: str, prefix: str) -> float:
    raw = name.removeprefix(prefix)
    raw = raw.replace("m", "-").replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return float("inf")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "rank",
        "teacher",
        "score",
        "n",
        "cosine_mean",
        "cosine_std",
        "diff_norm_mean",
        "diff_norm_std",
        "diff_norm_normalized",
        "norm_ratio_mean",
        "norm_ratio_std",
        "ratio_log_abs",
        "diff_hi_freq_ratio_mean",
        "diff_low_freq_ratio_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_prefix(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith("_pairwise.csv"):
        name = name[: -len("_pairwise.csv")]
    elif name.endswith(".csv"):
        name = name[:-4]
    return input_path.with_name(name)


def main() -> int:
    args = parse_args()
    pairwise_path = Path(args.pairwise_csv)
    out_prefix = Path(args.out_prefix) if args.out_prefix else default_prefix(pairwise_path)
    ranking_csv = out_prefix.with_name(out_prefix.name + "_teacher_ranking.csv")
    ranking_json = out_prefix.with_name(out_prefix.name + "_teacher_ranking.json")

    rows = read_rows(pairwise_path)
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        teacher = row.get("variant_a", "")
        target = row.get("variant_b", "")
        if not teacher.startswith(args.teacher_prefix):
            continue
        if target != args.target:
            continue
        grouped.setdefault(teacher, []).append(row)

    if not grouped:
        raise ValueError(
            f"No rows found where variant_a starts with {args.teacher_prefix!r} "
            f"and variant_b == {args.target!r}"
        )

    stats = []
    max_diff = 0.0
    for teacher, items in grouped.items():
        diff_vals = [_float(r, "diff_norm") for r in items]
        diff_mean = _mean(diff_vals)
        if math.isfinite(diff_mean):
            max_diff = max(max_diff, diff_mean)

    max_diff = max(max_diff, 1e-12)

    for teacher, items in grouped.items():
        cosine_vals = [_float(r, "cosine") for r in items]
        diff_vals = [_float(r, "diff_norm") for r in items]
        ratio_vals = [_float(r, "norm_ratio_b_over_a") for r in items]
        hi_vals = [_float(r, "diff_hi_freq_ratio") for r in items]
        lo_vals = [_float(r, "diff_low_freq_ratio") for r in items]

        cosine_mean = _mean(cosine_vals)
        cosine_std = _std(cosine_vals)
        diff_mean = _mean(diff_vals)
        diff_std = _std(diff_vals)
        ratio_mean = _mean(ratio_vals)
        ratio_std = _std(ratio_vals)
        ratio_log_abs = abs(math.log(max(ratio_mean, 1e-12)))
        diff_norm_normalized = diff_mean / max_diff

        score = (
            cosine_mean
            - args.diff_weight * diff_norm_normalized
            - args.ratio_weight * ratio_log_abs
            - args.stability_weight * cosine_std
        )
        stats.append(
            {
                "rank": 0,
                "teacher": teacher,
                "score": score,
                "n": len(items),
                "cosine_mean": cosine_mean,
                "cosine_std": cosine_std,
                "diff_norm_mean": diff_mean,
                "diff_norm_std": diff_std,
                "diff_norm_normalized": diff_norm_normalized,
                "norm_ratio_mean": ratio_mean,
                "norm_ratio_std": ratio_std,
                "ratio_log_abs": ratio_log_abs,
                "diff_hi_freq_ratio_mean": _mean(hi_vals),
                "diff_low_freq_ratio_mean": _mean(lo_vals),
            }
        )

    stats.sort(
        key=lambda item: (
            -float(item["score"]),
            _lambda_sort_key(str(item["teacher"]), args.teacher_prefix),
        )
    )
    for idx, item in enumerate(stats, start=1):
        item["rank"] = idx

    write_csv(ranking_csv, stats)
    meta = {
        "mode": "select_teacher_lambda",
        "pairwise_csv": str(pairwise_path),
        "target": args.target,
        "teacher_prefix": args.teacher_prefix,
        "diff_weight": args.diff_weight,
        "ratio_weight": args.ratio_weight,
        "stability_weight": args.stability_weight,
        "best": stats[0],
        "ranking": stats,
        "ranking_csv": str(ranking_csv),
    }
    with ranking_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"OK: ranked {len(stats)} teachers from {pairwise_path}")
    print(f"OK: wrote ranking csv -> {ranking_csv}")
    print(f"OK: wrote ranking json -> {ranking_json}")
    print(f"BEST: {stats[0]['teacher']} score={stats[0]['score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
