"""Run teacher prompt sweep across multiple styles and select a global lambda."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher prompt sweep for multiple styles.")
    parser.add_argument("--probe-mode", choices=["dry-run", "run-model"], default="dry-run")
    parser.add_argument("--style-ids", default="00,13,27")
    parser.add_argument("--prompt-ids", default="kp0,kp1,kp2,kp3,kp4")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--selected-timesteps", default="700,500,300")
    parser.add_argument("--out-prefix", default="teacher_style_sweep")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--teacher-lambdas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--teacher-highpass-kernel", type=int, default=5)
    parser.add_argument("--sinkhorn-iters", type=int, default=6)
    parser.add_argument("--sinkhorn-tau", type=float, default=0.02)
    parser.add_argument("--runtime-height", type=int, default=512)
    parser.add_argument("--runtime-width", type=int, default=512)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--diff-weight", type=float, default=0.25)
    parser.add_argument("--ratio-weight", type=float, default=0.15)
    parser.add_argument("--stability-weight", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dump-latents",
        action="store_true",
        help="Propagate --dump-latents into each prompt/style teacher probe.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_slug(style_id: str) -> str:
    return f"style{style_id}"


def main() -> int:
    args = parse_args()
    python = sys.executable
    results_dir = Path(args.results_dir)
    style_ids = parse_csv_list(args.style_ids)
    if not style_ids:
        raise ValueError("--style-ids must contain at least one style id")

    runs = []
    combined_pairwise_rows: List[Dict[str, str]] = []

    for style_id in style_ids:
        run_prefix = f"{args.out_prefix}_{style_slug(style_id)}"
        cmd = [
            python,
            "scripts/run_prompt_sweep.py",
            "--probe-mode",
            args.probe_mode,
            "--style-id",
            style_id,
            "--prompt-ids",
            args.prompt_ids,
            "--seed",
            str(args.seed),
            "--selected-timesteps",
            args.selected_timesteps,
            "--out-prefix",
            run_prefix,
            "--results-dir",
            args.results_dir,
            "--teacher-lambdas",
            args.teacher_lambdas,
            "--teacher-highpass-kernel",
            str(args.teacher_highpass_kernel),
            "--sinkhorn-iters",
            str(args.sinkhorn_iters),
            "--sinkhorn-tau",
            str(args.sinkhorn_tau),
            "--min-free-gb",
            str(args.min_free_gb),
            "--diff-weight",
            str(args.diff_weight),
            "--ratio-weight",
            str(args.ratio_weight),
            "--stability-weight",
            str(args.stability_weight),
        ]
        if args.probe_mode == "run-model":
            cmd.extend(["--runtime-height", str(args.runtime_height), "--runtime-width", str(args.runtime_width)])
        else:
            cmd.extend(["--height", str(args.height), "--width", str(args.width)])
        if args.force:
            cmd.append("--force")
        if args.dump_latents:
            cmd.append("--dump-latents")

        run_cmd(cmd, PROJECT_ROOT)

        combined_pairwise_csv = results_dir / f"{run_prefix}_combined_pairwise.csv"
        combined_ranking_json = results_dir / f"{run_prefix}_combined_teacher_ranking.json"
        rows = read_csv(combined_pairwise_csv)
        combined_pairwise_rows.extend(rows)
        runs.append(
            {
                "style_id": style_id,
                "run_prefix": run_prefix,
                "combined_pairwise_csv": str(combined_pairwise_csv),
                "combined_ranking_json": str(combined_ranking_json),
            }
        )

    global_pairwise_csv = results_dir / f"{args.out_prefix}_global_pairwise.csv"
    write_csv(global_pairwise_csv, combined_pairwise_rows)

    global_prefix = results_dir / f"{args.out_prefix}_global"
    select_cmd = [
        python,
        "scripts/select_teacher_lambda.py",
        "--pairwise-csv",
        str(global_pairwise_csv),
        "--out-prefix",
        str(global_prefix),
        "--diff-weight",
        str(args.diff_weight),
        "--ratio-weight",
        str(args.ratio_weight),
        "--stability-weight",
        str(args.stability_weight),
    ]
    run_cmd(select_cmd, PROJECT_ROOT)

    manifest = {
        "mode": "teacher_style_sweep",
        "probe_mode": args.probe_mode,
        "style_ids": style_ids,
        "prompt_ids": parse_csv_list(args.prompt_ids),
        "seed": args.seed,
        "selected_timesteps": args.selected_timesteps,
        "out_prefix": args.out_prefix,
        "teacher_lambdas": args.teacher_lambdas,
        "teacher_highpass_kernel": args.teacher_highpass_kernel,
        "dump_latents": bool(args.dump_latents),
        "global_pairwise_csv": str(global_pairwise_csv),
        "global_ranking_json": str(results_dir / f"{args.out_prefix}_global_teacher_ranking.json"),
        "runs": runs,
    }
    manifest_path = results_dir / f"{args.out_prefix}_style_sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"OK: wrote style sweep manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
