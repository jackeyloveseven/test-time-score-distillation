"""End-to-end teacher-sweep pipeline runner.

Pipeline:
1. Run ``probe_klein9b_residuals.py`` with ``--dump-fields``.
2. Run ``postprocess_teacher_sweep.py`` over the dumped hard/OT residuals.
3. Run ``select_teacher_lambda.py`` over the postprocess pairwise CSV.

Use ``--probe-mode dry-run`` for CPU-only regression, or ``--probe-mode run-model``
when the GPU is available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual dump -> teacher sweep -> lambda selection.")
    parser.add_argument("--probe-mode", choices=["dry-run", "run-model", "postprocess-only"], default="dry-run")
    parser.add_argument("--style-id", default="27")
    parser.add_argument("--prompt-id", default="kp0")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--selected-timesteps", default="700,500,300")
    parser.add_argument("--out-prefix", default="teacher_pipeline")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--teacher-lambdas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--teacher-highpass-kernel", type=int, default=5)
    parser.add_argument("--sinkhorn-iters", type=int, default=6)
    parser.add_argument("--sinkhorn-tau", type=float, default=0.02)
    parser.add_argument("--inject-beta", type=float, default=1.0)
    parser.add_argument("--runtime-height", type=int, default=512)
    parser.add_argument("--runtime-width", type=int, default=512)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=40.0,
        help="For run-model mode, refuse to start if GPU free memory is below this.",
    )
    parser.add_argument(
        "--field-dumps",
        default=None,
        help="Override dump glob. Required for postprocess-only, optional otherwise.",
    )
    parser.add_argument("--diff-weight", type=float, default=0.25)
    parser.add_argument("--ratio-weight", type=float, default=0.15)
    parser.add_argument("--stability-weight", type=float, default=0.10)
    parser.add_argument("--force", action="store_true", help="Run model even when GPU free-memory check fails.")
    parser.add_argument("--no-hyperion-grad", action="store_true", default=True)
    parser.add_argument(
        "--dump-latents",
        action="store_true",
        help="Also dump latents.z_t with each residual field dump for direct score residual training.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def query_gpu_free_gb() -> Optional[float]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return None
    values = [float(line.strip()) for line in out.splitlines() if line.strip()]
    if not values:
        return None
    return max(values) / 1024.0


def ensure_gpu_available(args: argparse.Namespace) -> None:
    if args.probe_mode != "run-model" or args.force:
        return
    free_gb = query_gpu_free_gb()
    if free_gb is None:
        print("WARN: could not query GPU free memory; continuing", flush=True)
        return
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"GPU free memory {free_gb:.2f} GB is below --min-free-gb {args.min_free_gb:.2f}. "
            "Use --force to bypass, or run --probe-mode postprocess-only on existing dumps."
        )


def main() -> int:
    args = parse_args()
    python = sys.executable
    results_dir = Path(args.results_dir)
    dump_dir = results_dir / "field_dumps"
    probe_prefix = args.out_prefix + "_probe"
    teacher_prefix = args.out_prefix + "_teacher"

    ensure_gpu_available(args)

    if args.probe_mode != "postprocess-only":
        probe_cmd = [
            python,
            "scripts/probe_klein9b_residuals.py",
            "--dump-fields",
            "--style-id",
            args.style_id,
            "--prompt-id",
            args.prompt_id,
            "--seed",
            str(args.seed),
            "--selected-timesteps",
            args.selected_timesteps,
            "--out-prefix",
            probe_prefix,
            "--results-dir",
            args.results_dir,
            "--teacher-lambdas",
            "0",
            "--teacher-highpass-kernel",
            str(args.teacher_highpass_kernel),
            "--sinkhorn-iters",
            str(args.sinkhorn_iters),
            "--sinkhorn-tau",
            str(args.sinkhorn_tau),
            "--inject-beta",
            str(args.inject_beta),
        ]
        if args.no_hyperion_grad:
            probe_cmd.append("--no-hyperion-grad")
        if args.dump_latents:
            probe_cmd.append("--dump-latents")
        if args.probe_mode == "run-model":
            probe_cmd.append("--run-model")
            probe_cmd.extend(["--runtime-height", str(args.runtime_height), "--runtime-width", str(args.runtime_width)])
            if args.vram_limit_gb is not None:
                probe_cmd.extend(["--vram-limit-gb", str(args.vram_limit_gb)])
        else:
            probe_cmd.extend(["--dry-run", "--height", str(args.height), "--width", str(args.width)])
        run_cmd(probe_cmd, PROJECT_ROOT)

    field_glob = args.field_dumps
    if field_glob is None:
        field_glob = str(dump_dir / f"{probe_prefix}_{args.prompt_id}_{args.style_id}_t*.pt")

    post_cmd = [
        python,
        "scripts/postprocess_teacher_sweep.py",
        "--field-dumps",
        field_glob,
        "--out-prefix",
        teacher_prefix,
        "--results-dir",
        args.results_dir,
        "--teacher-lambdas",
        args.teacher_lambdas,
        "--teacher-highpass-kernel",
        str(args.teacher_highpass_kernel),
    ]
    run_cmd(post_cmd, PROJECT_ROOT)

    ranking_prefix = str(results_dir / teacher_prefix)
    select_cmd = [
        python,
        "scripts/select_teacher_lambda.py",
        "--pairwise-csv",
        str(results_dir / f"{teacher_prefix}_pairwise.csv"),
        "--out-prefix",
        ranking_prefix,
        "--diff-weight",
        str(args.diff_weight),
        "--ratio-weight",
        str(args.ratio_weight),
        "--stability-weight",
        str(args.stability_weight),
    ]
    run_cmd(select_cmd, PROJECT_ROOT)

    manifest = {
        "mode": "teacher_pipeline",
        "probe_mode": args.probe_mode,
        "style_id": args.style_id,
        "prompt_id": args.prompt_id,
        "seed": args.seed,
        "selected_timesteps": args.selected_timesteps,
        "out_prefix": args.out_prefix,
        "field_dumps": field_glob,
        "dump_latents": bool(args.dump_latents),
        "probe_prefix": probe_prefix,
        "teacher_prefix": teacher_prefix,
        "teacher_lambdas": args.teacher_lambdas,
        "teacher_highpass_kernel": args.teacher_highpass_kernel,
        "ranking_json": str(results_dir / f"{teacher_prefix}_teacher_ranking.json"),
    }
    manifest_path = results_dir / f"{args.out_prefix}_pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"OK: wrote pipeline manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
