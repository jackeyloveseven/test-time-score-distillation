"""Run fixed-setting score-student samples across multiple styles/prompts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score-student image samples across styles/prompts.")
    parser.add_argument("--style-ids", default="00,13,27")
    parser.add_argument("--prompt-ids", default="kp0,kp1,kp3")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--out-prefix", default="score_student_batch_recommended")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--inject-beta", type=float, default=0.5)
    parser.add_argument("--score-student-scale", type=float, default=1.0)
    parser.add_argument(
        "--score-student-style-scales",
        default="",
        help="Optional per-style scale map, e.g. '08:0.75,22:1.25'.",
    )
    parser.add_argument("--score-student-timestep-min", type=float, default=None)
    parser.add_argument("--score-student-timestep-max", type=float, default=None)
    parser.add_argument("--score-student-checkpoint", default="results/score_residual_student_latent45_lam1_hard_delta/student_best.pt")
    parser.add_argument("--score-student-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_csv(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one CSV value")
    return values


def slug_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_style_scales(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid style scale item {item!r}; expected style_id:scale")
        style_id, value = item.split(":", 1)
        style_id = style_id.strip()
        if not style_id:
            raise ValueError(f"Invalid empty style id in {item!r}")
        out[style_id] = float(value.strip())
    return out


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def image_stats(path: Path) -> Dict[str, object]:
    arr = np.array(Image.open(path).convert("RGB"))
    return {
        "shape": [int(v) for v in arr.shape],
        "mean": [float(x) for x in arr.mean(axis=(0, 1))],
        "std": [float(x) for x in arr.std(axis=(0, 1))],
        "min": int(arr.min()),
        "max": int(arr.max()),
        "bytes": int(path.stat().st_size),
    }


def make_contact_sheet(result_dir: Path, out_prefix: str, rows: List[Dict[str, object]]) -> Path:
    sample_dir = result_dir / "student_samples"
    first = Image.open(rows[0]["normal_path"]).convert("RGB")
    w, h = first.size
    label_h = 36
    cols = 3
    sheet = Image.new("RGB", (cols * w, len(rows) * (h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        y = r * (h + label_h)
        entries = [
            ("normal", Path(row["normal_path"])),
            ("hard", Path(row["hard_path"])),
            ("score_student", Path(row["score_path"])),
        ]
        row_label = f"style={row['style_id']} prompt={row['prompt_id']}"
        for c, (label, path) in enumerate(entries):
            image = Image.open(path).convert("RGB")
            sheet.paste(image, (c * w, y + label_h))
            draw.text((c * w + 8, y + 8), f"{row_label} {label}", fill=(0, 0, 0))
    out_path = sample_dir / f"{out_prefix}_contact.png"
    sheet.save(out_path)
    return out_path


def main() -> int:
    args = parse_args()
    style_ids = parse_csv(args.style_ids)
    prompt_ids = parse_csv(args.prompt_ids)
    result_dir = Path(args.results_dir)
    sample_dir = result_dir / "student_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    rows: List[Dict[str, object]] = []
    style_scale_map = parse_style_scales(args.score_student_style_scales)

    beta_slug = slug_float(args.inject_beta)
    scale_slug = slug_float(args.score_student_scale)
    if style_scale_map:
        scale_slug = f"{scale_slug}_stylemap"
    if args.score_student_timestep_min is not None or args.score_student_timestep_max is not None:
        min_slug = "none" if args.score_student_timestep_min is None else slug_float(args.score_student_timestep_min)
        max_slug = "none" if args.score_student_timestep_max is None else slug_float(args.score_student_timestep_max)
        scale_slug = f"{scale_slug}_tw{min_slug}-{max_slug}"
    for style_id in style_ids:
        effective_scale = float(style_scale_map.get(style_id, args.score_student_scale))
        for prompt_id in prompt_ids:
            combo_prefix = (
                f"{args.out_prefix}_style{style_id}_{prompt_id}"
                f"_b{beta_slug}_s{scale_slug}_steps{args.num_steps}"
            )
            score_path = sample_dir / f"{combo_prefix}_score_student.png"
            if args.force or not score_path.exists():
                cmd = [
                    python,
                    "scripts/sample_student_images.py",
                    "--style-id",
                    style_id,
                    "--prompt-id",
                    prompt_id,
                    "--seed",
                    str(args.seed),
                    "--height",
                    str(args.height),
                    "--width",
                    str(args.width),
                    "--num-steps",
                    str(args.num_steps),
                    "--out-prefix",
                    combo_prefix,
                    "--results-dir",
                    args.results_dir,
                    "--score-student-checkpoint",
                    args.score_student_checkpoint,
                    "--score-student-device",
                    args.score_student_device,
                    "--score-student-scale",
                    str(args.score_student_scale),
                    "--score-student-style-scales",
                    args.score_student_style_scales,
                    "--inject-beta",
                    str(args.inject_beta),
                    "--modes",
                    "normal,hard,score_student",
                ]
                if args.score_student_timestep_min is not None:
                    cmd.extend(["--score-student-timestep-min", str(args.score_student_timestep_min)])
                if args.score_student_timestep_max is not None:
                    cmd.extend(["--score-student-timestep-max", str(args.score_student_timestep_max)])
                if args.vram_limit_gb is not None:
                    cmd.extend(["--vram-limit-gb", str(args.vram_limit_gb)])
                run_cmd(cmd, PROJECT_ROOT)

            normal_path = sample_dir / f"{combo_prefix}_normal.png"
            hard_path = sample_dir / f"{combo_prefix}_hard.png"
            meta_path = result_dir / f"{combo_prefix}_meta.json"
            meta = json.load(meta_path.open("r", encoding="utf-8"))
            ratios = []
            scaled_ratios = []
            step_scales = []
            for item in meta.get("step_summaries", []):
                if item.get("branch") == "score_student":
                    scale = float(item.get("score_student_scale", meta.get("score_student_effective_scale", effective_scale)))
                    ratio = item["score_delta_norm"] / max(item["hard_residual_norm"], 1e-8)
                    ratios.append(ratio)
                    scaled_ratios.append(scale * ratio)
                    step_scales.append(scale)

            rows.append(
                {
                    "style_id": style_id,
                    "prompt_id": prompt_id,
                    "score_student_effective_scale": meta.get("score_student_effective_scale", effective_scale),
                    "prefix": combo_prefix,
                    "normal_path": str(normal_path),
                    "hard_path": str(hard_path),
                    "score_path": str(score_path),
                    "meta_path": str(meta_path),
                    "normal_stats": image_stats(normal_path),
                    "hard_stats": image_stats(hard_path),
                    "score_stats": image_stats(score_path),
                    "ratio_mean": float(np.mean(ratios)) if ratios else None,
                    "ratio_min": float(np.min(ratios)) if ratios else None,
                    "ratio_max": float(np.max(ratios)) if ratios else None,
                    "scaled_ratio_mean": float(np.mean(scaled_ratios)) if scaled_ratios else None,
                    "scaled_ratio_min": float(np.min(scaled_ratios)) if scaled_ratios else None,
                    "scaled_ratio_max": float(np.max(scaled_ratios)) if scaled_ratios else None,
                    "step_scale_mean": float(np.mean(step_scales)) if step_scales else None,
                    "active_step_count": int(sum(1 for value in step_scales if value != 0.0)),
                }
            )

    contact_path = make_contact_sheet(result_dir, args.out_prefix, rows)
    summary = {
        "mode": "score_image_batch",
        "style_ids": style_ids,
        "prompt_ids": prompt_ids,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "inject_beta": args.inject_beta,
        "score_student_scale": args.score_student_scale,
        "score_student_style_scales": style_scale_map,
        "score_student_timestep_min": args.score_student_timestep_min,
        "score_student_timestep_max": args.score_student_timestep_max,
        "score_student_checkpoint": args.score_student_checkpoint,
        "contact_path": str(contact_path),
        "rows": rows,
    }
    summary_path = result_dir / f"{args.out_prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"OK: wrote batch summary -> {summary_path}")
    print(f"OK: wrote contact sheet -> {contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
