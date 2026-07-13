"""Build a distillation dataset manifest from residual field dumps."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_ops import highpass_corrected_teacher


PROMPTS = {
    "kp0": "A panda eat bamboo",
    "kp1": "A castle on a cliff by the sea",
    "kp2": "A cat sitting on a windowsill",
    "kp3": "A field of sunflowers at sunset",
    "kp4": "A portrait of an old fisherman",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL manifest for residual distillation dumps.")
    parser.add_argument(
        "--field-dumps",
        required=True,
        action="append",
        help="Glob for dumped .pt residual fields. Can be passed multiple times.",
    )
    parser.add_argument("--out-prefix", default="distill_dataset", help="Output prefix under results-dir.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--teacher-kind", default="highpass_delta", choices=["highpass_delta", "hard", "ot"])
    parser.add_argument("--teacher-lambda", type=float, default=1.0)
    parser.add_argument("--teacher-highpass-kernel", type=int, default=5)
    parser.add_argument(
        "--verify-teacher",
        action="store_true",
        help="Materialize teacher tensor once per dump to verify construction.",
    )
    parser.add_argument(
        "--ranking-json",
        default=None,
        help="Optional ranking json used to justify selected teacher.",
    )
    return parser.parse_args()


def load_dump(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def teacher_name(args: argparse.Namespace) -> str:
    if args.teacher_kind == "hard":
        return "hard_scfg"
    if args.teacher_kind == "ot":
        return "otcfg"
    lam = f"{args.teacher_lambda:g}".replace("-", "m").replace(".", "p")
    return f"teacher_hp_lam{lam}"


def verify_teacher_tensor(dump: dict, args: argparse.Namespace) -> Dict[str, object]:
    fields = dump["fields"]
    hard = fields["hard_scfg"].detach().float().cpu()
    ot = fields["otcfg"].detach().float().cpu()
    if args.teacher_kind == "hard":
        teacher = hard
    elif args.teacher_kind == "ot":
        teacher = ot
    else:
        teacher = highpass_corrected_teacher(
            hard,
            ot,
            height=int(dump["height"]),
            width=int(dump["width"]),
            lamb=args.teacher_lambda,
            kernel_size=args.teacher_highpass_kernel,
        )
    return {
        "teacher_shape": [int(v) for v in teacher.shape],
        "teacher_norm": float(torch.linalg.norm(teacher.reshape(teacher.shape[0], -1), dim=1)[0].item()),
    }


def main() -> int:
    args = parse_args()
    matched_paths = []
    for pattern in args.field_dumps:
        matched_paths.extend(glob.glob(pattern))
    dump_paths = [Path(p) for p in sorted(set(matched_paths))]
    if not dump_paths:
        raise FileNotFoundError(f"No field dumps matched: {args.field_dumps}")

    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = result_dir / f"{args.out_prefix}.jsonl"
    summary_path = result_dir / f"{args.out_prefix}_summary.json"
    teacher = teacher_name(args)

    rows: List[Dict[str, object]] = []
    style_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    timestep_counts: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    ranking_meta: Optional[dict] = None

    if args.ranking_json:
        with Path(args.ranking_json).open("r", encoding="utf-8") as f:
            ranking_meta = json.load(f)

    for idx, dump_path in enumerate(dump_paths):
        dump = load_dump(dump_path)
        fields = dump.get("fields")
        if not isinstance(fields, dict):
            raise ValueError(f"Dump has no fields dict: {dump_path}")
        if "hard_scfg" not in fields or "otcfg" not in fields:
            raise ValueError(f"Dump missing hard_scfg/otcfg: {dump_path}")

        hard = fields["hard_scfg"]
        ot = fields["otcfg"]
        if tuple(hard.shape) != tuple(ot.shape):
            raise ValueError(f"hard/ot shape mismatch in {dump_path}: {tuple(hard.shape)} vs {tuple(ot.shape)}")

        style_id = str(dump.get("style_id", "unknown"))
        prompt_id = str(dump.get("prompt_id", "unknown"))
        timestep = float(dump.get("timestep", 0.0))
        height = int(dump.get("height"))
        width = int(dump.get("width"))
        field_shape = [int(v) for v in hard.shape]

        row: Dict[str, object] = {
            "sample_id": idx,
            "dump_path": str(dump_path),
            "style_id": style_id,
            "prompt_id": prompt_id,
            "prompt_text": PROMPTS.get(prompt_id, prompt_id),
            "timestep": timestep,
            "height": height,
            "width": width,
            "field_shape": field_shape,
            "source_fields": {
                "normal_residual_base": "hard_scfg",
                "ot_target": "otcfg",
            },
            "teacher": {
                "name": teacher,
                "kind": args.teacher_kind,
                "lambda": args.teacher_lambda,
                "highpass_kernel": args.teacher_highpass_kernel,
                "formula": (
                    "hard_scfg + lambda * highpass(otcfg - hard_scfg)"
                    if args.teacher_kind == "highpass_delta"
                    else teacher
                ),
            },
        }

        if args.verify_teacher:
            row["verification"] = verify_teacher_tensor(dump, args)

        rows.append(row)
        style_counts[style_id] += 1
        prompt_counts[prompt_id] += 1
        timestep_counts[f"{timestep:.6f}"] += 1
        shapes[str(field_shape)] += 1

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "mode": "distill_manifest",
        "field_dumps_glob": args.field_dumps[0] if len(args.field_dumps) == 1 else args.field_dumps,
        "field_dumps_globs": args.field_dumps,
        "jsonl_path": str(jsonl_path),
        "num_samples": len(rows),
        "teacher": {
            "name": teacher,
            "kind": args.teacher_kind,
            "lambda": args.teacher_lambda,
            "highpass_kernel": args.teacher_highpass_kernel,
        },
        "style_counts": dict(sorted(style_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "timestep_counts": dict(sorted(timestep_counts.items())),
        "field_shapes": dict(shapes),
        "ranking_json": args.ranking_json,
        "ranking_best": None if ranking_meta is None else ranking_meta.get("best"),
        "verified_teacher": bool(args.verify_teacher),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"OK: wrote {len(rows)} samples -> {jsonl_path}")
    print(f"OK: wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
