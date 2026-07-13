"""Build a manifest for direct score/velocity residual distillation.

Unlike build_distill_manifest.py, this script requires each dump to contain
latents.z_t. That keeps direct f_theta(z_t, t, prompt, style) training separate
from the older hard_scfg-only adapter baseline.
"""

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
    parser = argparse.ArgumentParser(
        description="Build JSONL manifest for direct score residual distillation dumps."
    )
    parser.add_argument(
        "--field-dumps",
        required=True,
        action="append",
        help="Glob for .pt dumps produced with --dump-fields --dump-latents.",
    )
    parser.add_argument("--out-prefix", default="score_residual_dataset")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--teacher-kind",
        default="highpass_delta",
        choices=["highpass_delta", "hard", "ot", "native_delta_velocity", "native_delta_score"],
    )
    parser.add_argument("--teacher-lambda", type=float, default=1.0)
    parser.add_argument("--teacher-highpass-kernel", type=int, default=5)
    parser.add_argument(
        "--verify-teacher",
        action="store_true",
        help="Materialize teacher tensor once per dump to verify construction.",
    )
    parser.add_argument(
        "--allow-missing-style-path",
        action="store_true",
        help="Allow rows without style_path metadata. Useful for dry-run fixtures only.",
    )
    return parser.parse_args()


def load_dump(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def format_float_id(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def teacher_name(args: argparse.Namespace) -> str:
    if args.teacher_kind in {"native_delta_velocity", "native_delta_score"}:
        return args.teacher_kind
    if args.teacher_kind == "hard":
        return "hard_scfg"
    if args.teacher_kind == "ot":
        return "otcfg"
    return f"teacher_hp_lam{format_float_id(args.teacher_lambda)}"


def verify_teacher_tensor(dump: dict, args: argparse.Namespace) -> Dict[str, object]:
    fields = dump["fields"]
    if args.teacher_kind in {"native_delta_velocity", "native_delta_score"}:
        teacher = fields[args.teacher_kind].detach().float().cpu()
        return {
            "teacher_shape": [int(v) for v in teacher.shape],
            "teacher_norm": float(torch.linalg.norm(teacher.reshape(teacher.shape[0], -1), dim=1)[0].item()),
        }
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


def validate_dump(dump: dict, dump_path: Path, args: argparse.Namespace) -> dict:
    fields = dump.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"Dump has no fields dict: {dump_path}")
    required_fields = (args.teacher_kind,) if args.teacher_kind in {"native_delta_velocity", "native_delta_score"} else ("hard_scfg", "otcfg")
    for key in required_fields:
        if key not in fields:
            raise ValueError(f"Dump missing fields.{key}: {dump_path}")

    latents = dump.get("latents")
    if not isinstance(latents, dict) or "z_t" not in latents:
        raise ValueError(f"Dump missing latents.z_t; regenerate with --dump-latents: {dump_path}")

    z_t = latents["z_t"]
    if args.teacher_kind in {"native_delta_velocity", "native_delta_score"}:
        teacher = fields[args.teacher_kind]
        if tuple(z_t.shape) != tuple(teacher.shape):
            raise ValueError(f"z_t/native field shape mismatch in {dump_path}: {tuple(z_t.shape)} vs {tuple(teacher.shape)}")
    else:
        hard = fields["hard_scfg"]
        ot = fields["otcfg"]
        if tuple(hard.shape) != tuple(ot.shape):
            raise ValueError(f"hard/ot shape mismatch in {dump_path}: {tuple(hard.shape)} vs {tuple(ot.shape)}")
        if tuple(z_t.shape) != tuple(hard.shape):
            raise ValueError(f"z_t/residual shape mismatch in {dump_path}: {tuple(z_t.shape)} vs {tuple(hard.shape)}")

    style_path = dump.get("style_path")
    if not args.allow_missing_style_path and not style_path:
        raise ValueError(f"Dump missing style_path metadata: {dump_path}")

    return {
        "fields": fields,
        "latents": latents,
        "style_path": style_path,
    }


def main() -> int:
    args = parse_args()
    matched_paths: List[str] = []
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
    latent_dtypes: Counter[str] = Counter()

    for idx, dump_path in enumerate(dump_paths):
        dump = load_dump(dump_path)
        validated = validate_dump(dump, dump_path, args)
        hard = (
            validated["fields"]["hard_scfg"]
            if "hard_scfg" in validated["fields"]
            else validated["fields"][args.teacher_kind]
        )
        z_t = validated["latents"]["z_t"]

        style_id = str(dump.get("style_id", "unknown"))
        prompt_id = str(dump.get("prompt_id", "unknown"))
        prompt_text = str(dump.get("prompt_text", PROMPTS.get(prompt_id, prompt_id)))
        timestep = float(dump.get("timestep", 0.0))
        height = int(dump.get("height"))
        width = int(dump.get("width"))
        field_shape = [int(v) for v in hard.shape]
        latent_shape = [int(v) for v in z_t.shape]
        latent_dtype = str(validated["latents"].get("z_t_dtype", str(z_t.dtype)))
        reference_code = dump.get("reference_code")

        row: Dict[str, object] = {
            "sample_id": idx,
            "dump_path": str(dump_path),
            "latent_key": "latents.z_t",
            "style_id": style_id,
            "style_path": validated["style_path"],
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "timestep": timestep,
            "height": height,
            "width": width,
            "latent_shape": latent_shape,
            "latent_dtype": latent_dtype,
            "field_shape": field_shape,
            "reference_code_key": "reference_code" if reference_code is not None else None,
            "reference_code_type": dump.get("reference_code_type"),
            "reference_code_shape": [int(v) for v in reference_code.shape] if reference_code is not None else None,
            "inputs": {
                "latent": "latents.z_t",
                "optional_hard_residual": "fields.hard_scfg" if "hard_scfg" in validated["fields"] else None,
                "optional_ot_residual": "fields.otcfg" if "otcfg" in validated["fields"] else None,
            },
            "target": {
                "name": teacher,
                "kind": args.teacher_kind,
                "lambda": args.teacher_lambda,
                "highpass_kernel": args.teacher_highpass_kernel,
                "formula": (
                    "hard_scfg + lambda * highpass(otcfg - hard_scfg)"
                    if args.teacher_kind == "highpass_delta"
                    else "native with-style field - native without-style field"
                    if args.teacher_kind in {"native_delta_velocity", "native_delta_score"}
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
        latent_dtypes[latent_dtype] += 1

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "mode": "score_residual_manifest",
        "requires_latents_z_t": True,
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
        "latent_dtypes": dict(latent_dtypes),
        "verified_teacher": bool(args.verify_teacher),
        "allow_missing_style_path": bool(args.allow_missing_style_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"OK: wrote {len(rows)} score residual samples -> {jsonl_path}")
    print(f"OK: wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
