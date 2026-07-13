"""Residual field profiling entrypoint.

The default mode is dry-run with synthetic tensors so no heavy models are loaded.
Use --run-model to enter runtime path (currently placeholder).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.profiling import compare_fields, write_csv, write_json_summary


ENV_NOTE = """Recommended environments:
- diffsynth: /home/ubuntu/miniconda3/envs/diffsynth
- yzh-dmd: /home/ubuntu/miniconda3/envs/yzh-dmd

Use --run-model only after these environments are prepared for Klein9B runtime."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile residual fields for hard-SCFG vs OT-CFG.")
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Enable runtime model path. Default is dry-run with random tensors.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run synthetic dry-run path (default).")
    parser.add_argument(
        "--out-prefix",
        default="residual_profile_dryrun",
        help="Output filename prefix under results-dir.",
    )
    parser.add_argument(
        "--env-note",
        action="store_true",
        help="Print recommended runtime environments.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for csv/json outputs.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument(
        "--styles",
        nargs="*",
        default=["00", "10", "27", "28"],
        help="Style IDs to profile in dry-run.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=["kp0", "kp1", "kp2"],
        help="Prompt IDs to profile in dry-run.",
    )
    parser.add_argument(
        "--timesteps",
        nargs="*",
        type=float,
        default=[0.2, 0.5],
        help="Timesteps for profiling samples.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _synthesize_field(
    batch: int,
    lq: int,
    d: int,
    direction_ref: torch.Tensor | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    base = torch.randn(batch, lq, d)
    if direction_ref is None:
        return scale * base

    # Mildly align to reference for controlled cosine behavior.
    ref_n = direction_ref / (torch.linalg.norm(direction_ref, dim=-1, keepdim=True) + 1e-8)
    proj = (base * ref_n).sum(dim=-1, keepdim=True)
    tangent = base - proj * ref_n
    return scale * (0.7 * ref_n + 0.3 * tangent)


def run_dry_run(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lq = args.height * args.width
    records: List = []
    for style_id in args.styles:
        for prompt_id in args.prompts:
            for t in args.timesteps:
                # Synthetic teacher fields.
                hyperion_grad = torch.randn(args.batch_size, lq, args.d)
                hyperion_delta = 0.6 * hyperion_grad + 0.2 * torch.randn_like(hyperion_grad)

                fields = {
                    "hard_scfg": _synthesize_field(args.batch_size, lq, args.d, direction_ref=hyperion_grad, scale=1.1),
                    "otcfg": _synthesize_field(args.batch_size, lq, args.d, direction_ref=hyperion_grad, scale=1.0),
                    "lrs_otcfg": _synthesize_field(args.batch_size, lq, args.d, direction_ref=hyperion_grad, scale=0.8),
                }

                recs = compare_fields(
                    fields=fields,
                    hyperion_grad=hyperion_grad,
                    hyperion_delta=hyperion_delta,
                    height=args.height,
                    width=args.width,
                    style_id=style_id,
                    prompt_id=prompt_id,
                    timestep=float(t),
                )
                records.extend(recs)

    csv_path = results_dir / f"{args.out_prefix}.csv"
    summary_path = results_dir / f"{args.out_prefix}_summary.json"

    write_csv(records, csv_path)
    write_json_summary(records, summary_path)

    print(f"OK: wrote {len(records)} rows to {csv_path}")
    print(f"OK: wrote summary to {summary_path}")
    return 0


def run_model_path(_args: argparse.Namespace) -> int:
    # Runtime integration point for actual Klein9B residual capture.
    # Left as scaffold until runtime hooks/paths are finalized.
    raise NotImplementedError("Model runtime profiling is not implemented yet in this phase.")


def main() -> int:
    args = _parse_args()

    if args.env_note:
        print(ENV_NOTE)

    _set_seed(args.seed)

    # Default behavior: dry-run unless --run-model is explicitly set.
    if args.run_model:
        return run_model_path(args)
    return run_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
