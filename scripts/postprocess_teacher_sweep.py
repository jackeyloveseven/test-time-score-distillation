"""Offline highpass-teacher sweep from dumped residual fields.

This script consumes ``results/field_dumps/*.pt`` files produced by
``probe_klein9b_residuals.py --dump-fields`` and recomputes:

    hard_scfg + lambda * highpass(otcfg - hard_scfg)

without loading Klein9B or DiffSynth.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from math import isqrt
from pathlib import Path
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_ops import highpass_corrected_teacher
from src.profiling import (
    compare_field_pairs,
    compare_fields,
    write_csv,
    write_json_summary,
    write_pairwise_csv,
    write_pairwise_json_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline teacher sweep from dumped hard_scfg/otcfg residual fields."
    )
    parser.add_argument(
        "--field-dumps",
        required=True,
        help="Glob for .pt field dumps, e.g. results/field_dumps/run_*.pt.",
    )
    parser.add_argument(
        "--out-prefix",
        default="teacher_sweep_postprocess",
        help="Output file prefix under results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for csv/json/meta outputs.",
    )
    parser.add_argument(
        "--teacher-lambdas",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated lambda values for hard + lambda * highpass(ot-hard).",
    )
    parser.add_argument(
        "--teacher-highpass-kernel",
        type=int,
        default=5,
        help="Odd spatial avg-pool kernel used to build highpass correction teachers.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override token-grid height. If omitted, read from dump or infer square grid.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override token-grid width. If omitted, read from dump or infer square grid.",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    if not raw:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [float(v) for v in values]


def format_lambda(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def resolve_hw(dump: dict, field: torch.Tensor, height: Optional[int], width: Optional[int]) -> tuple[int, int]:
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("--height and --width must be provided together")
        return int(height), int(width)

    if "height" in dump and "width" in dump:
        return int(dump["height"]), int(dump["width"])

    if field.ndim != 3:
        raise ValueError("Cannot infer height/width unless field is [B, L, D]")
    side = isqrt(int(field.shape[1]))
    if side * side != int(field.shape[1]):
        raise ValueError("height/width missing and token count is not a square")
    return side, side


def load_dump(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    args = parse_args()
    dump_paths = [Path(p) for p in sorted(glob.glob(args.field_dumps))]
    if not dump_paths:
        raise FileNotFoundError(f"No field dumps matched: {args.field_dumps}")

    teacher_lambdas = parse_float_list(args.teacher_lambdas)
    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = result_dir / f"{args.out_prefix}.csv"
    summary_path = result_dir / f"{args.out_prefix}_summary.json"
    pairwise_csv_path = result_dir / f"{args.out_prefix}_pairwise.csv"
    pairwise_summary_path = result_dir / f"{args.out_prefix}_pairwise_summary.json"
    meta_path = result_dir / f"{args.out_prefix}_meta.json"

    records = []
    pairwise_records = []
    processed = []

    for dump_path in dump_paths:
        dump = load_dump(dump_path)
        fields = dump.get("fields")
        if not isinstance(fields, dict):
            raise ValueError(f"Dump has no fields dict: {dump_path}")
        if "hard_scfg" not in fields or "otcfg" not in fields:
            raise ValueError(f"Dump must contain hard_scfg and otcfg: {dump_path}")

        hard_scfg = fields["hard_scfg"].detach().float().cpu()
        otcfg = fields["otcfg"].detach().float().cpu()
        height, width = resolve_hw(dump, hard_scfg, args.height, args.width)

        sweep_fields = {
            "hard_scfg": hard_scfg,
            "otcfg": otcfg,
        }
        for lamb in teacher_lambdas:
            name = f"teacher_hp_lam{format_lambda(lamb)}"
            sweep_fields[name] = highpass_corrected_teacher(
                hard_scfg,
                otcfg,
                height=height,
                width=width,
                lamb=lamb,
                kernel_size=args.teacher_highpass_kernel,
            )

        style_id = str(dump.get("style_id", "unknown"))
        prompt_id = str(dump.get("prompt_id", "unknown"))
        timestep = float(dump.get("timestep", 0.0))

        records.extend(
            compare_fields(
                fields=sweep_fields,
                hyperion_grad=None,
                hyperion_delta=None,
                height=height,
                width=width,
                style_id=style_id,
                prompt_id=prompt_id,
                timestep=timestep,
            )
        )

        pairwise_pairs = [("hard_scfg", "otcfg")]
        for lamb in teacher_lambdas:
            name = f"teacher_hp_lam{format_lambda(lamb)}"
            pairwise_pairs.append(("hard_scfg", name))
            pairwise_pairs.append((name, "otcfg"))

        pairwise_records.extend(
            compare_field_pairs(
                fields=sweep_fields,
                pairs=pairwise_pairs,
                height=height,
                width=width,
                style_id=style_id,
                prompt_id=prompt_id,
                timestep=timestep,
            )
        )

        processed.append(
            {
                "path": str(dump_path),
                "style_id": style_id,
                "prompt_id": prompt_id,
                "timestep": timestep,
                "height": height,
                "width": width,
            }
        )

    write_csv(records, csv_path)
    summary = write_json_summary(records, summary_path)
    write_pairwise_csv(pairwise_records, pairwise_csv_path)
    pairwise_summary = write_pairwise_json_summary(pairwise_records, pairwise_summary_path)

    meta = {
        "mode": "teacher_sweep_postprocess",
        "field_dumps_glob": args.field_dumps,
        "num_dumps": len(dump_paths),
        "num_records": len(records),
        "num_pairwise_records": len(pairwise_records),
        "teacher_lambdas": teacher_lambdas,
        "teacher_highpass_kernel": args.teacher_highpass_kernel,
        "variants": ["hard_scfg", "otcfg"] + [f"teacher_hp_lam{format_lambda(l)}" for l in teacher_lambdas],
        "summary": summary,
        "pairwise_summary": pairwise_summary,
        "processed": processed,
        "csv": str(csv_path),
        "summary_path": str(summary_path),
        "pairwise_csv": str(pairwise_csv_path),
        "pairwise_summary_path": str(pairwise_summary_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"OK: processed {len(dump_paths)} dumps")
    print(f"OK: wrote {len(records)} rows -> {csv_path}")
    print(f"OK: wrote summary -> {summary_path}")
    print(f"OK: wrote pairwise rows -> {pairwise_csv_path}")
    print(f"OK: wrote pairwise summary -> {pairwise_summary_path}")
    print(f"OK: wrote meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
