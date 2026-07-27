#!/usr/bin/env python3
"""Generate supplementary LaTeX tables from package-local result files."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)


def fmt(value: float) -> str:
    return f"{value:.3f}".lstrip("0")


def benchmark_table() -> None:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    with (DATA / "benchmark_full_metrics.csv").open() as handle:
        for row in csv.DictReader(handle):
            values[row["style_id"]][row["method"]] = float(row["csd_style_sim"])

    styles = sorted(values)
    left, right = styles[:20], styles[20:]
    rows = []
    for a, b in zip(left, right):
        cells = []
        for style in (a, b):
            row = values[style]
            cells.extend(
                [
                    style,
                    fmt(row["q25s"]),
                    fmt(row["kleinsys_sem"]),
                    fmt(row["pg_n8_s12_r"]),
                    fmt(row["kleinsys_gst"]),
                ]
            )
        rows.append(" & ".join(cells) + r" \\")
    content = "\n".join(
        [
            r"\begin{tabular}{ccccc@{\hspace{9pt}}ccccc}",
            r"\toprule",
            r"Style & TSD-S & Nat.-S & TSD-G & Nat.-G & Style & TSD-S & Nat.-S & TSD-G & Nat.-G \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (TABLES / "benchmark_per_style.tex").write_text(content + "\n")


def multiprompt_table() -> None:
    raw = (DATA / "benchmark_multiprompt_summary_raw.csv").read_text().splitlines()
    header = "pairing,prompt_id,ours_csd,klein_csd,ratio,ours_clipt,klein_clipt"
    start = raw.index(header) + 1
    parsed = []
    for line in raw[start:]:
        if not line or line.startswith("===") or line.startswith('"==='):
            break
        parsed.append(next(csv.DictReader([header, line])))
    by_prompt: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in parsed:
        mode = "semantic" if "Semantic" in row["pairing"] else "global"
        by_prompt[row["prompt_id"]][mode] = row

    names = {
        "kp0": "Panda",
        "kp1": "Castle",
        "kp2": "Cat",
        "kp3": "Sunflowers",
        "kp4": "Fisherman",
    }
    rows = []
    for prompt_id in names:
        semantic = by_prompt[prompt_id]["semantic"]
        global_ = by_prompt[prompt_id]["global"]
        for mode, result in (("Semantic", semantic), ("Global", global_)):
            rows.append(
                " & ".join(
                    [
                        names[prompt_id],
                        mode,
                        result["ours_csd"],
                        result["klein_csd"],
                        result["ratio"],
                    ]
                )
                + r" \\"
            )
    content = "\n".join(
        [
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Prompt & Mode & TSD & Native & Ratio \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (TABLES / "multiprompt.tex").write_text(content + "\n")


def correlation(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            result[position] = rank
        index = end
    return result


def diagnostic_macros() -> None:
    with (DATA / "srsd_results.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["finite"] == "True"]
    xs = [float(row["srsd"]) for row in rows]
    ys = [float(row["csd"]) for row in rows]
    pearson = correlation(xs, ys)
    spearman = correlation(ranks(xs), ranks(ys))
    text = (
        f"\\newcommand{{\\SRSDN}}{{{len(rows)}}}\n"
        f"\\newcommand{{\\SRSDPearson}}{{{pearson:.2f}}}\n"
        f"\\newcommand{{\\SRSDSpearman}}{{{spearman:.2f}}}\n"
    )
    (TABLES / "diagnostic_macros.tex").write_text(text)


def main() -> None:
    benchmark_table()
    multiprompt_table()
    diagnostic_macros()
    print(f"Generated supplementary tables in {TABLES}")


if __name__ == "__main__":
    main()
