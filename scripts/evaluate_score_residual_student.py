"""Evaluate scale sweeps for direct score residual students."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_score_residual_student import (  # noqa: E402
    ScoreResidualDataset,
    batch_metrics,
    choose_device,
    compose_prediction,
    load_jsonl,
    split_indices_from_rows,
)
from src.score_residual_student import ScoreResidualStudentConfig, build_score_residual_student  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate score student scale sweep.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--student-scales", default="0,0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["random", "style", "prompt", "timestep"],
        default="random",
    )
    parser.add_argument("--holdout-value", default=None)
    parser.add_argument("--spectral-kernel", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def parse_scales(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--student-scales must contain at least one scale.")
    return values


def make_config(raw_config: dict) -> ScoreResidualStudentConfig:
    valid = {field.name for field in fields(ScoreResidualStudentConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid}
    return ScoreResidualStudentConfig(**filtered)


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, Dict[str, int], Dict[str, int], dict]:
    ckpt = torch.load(path, map_location="cpu")
    config = make_config(ckpt["config"])
    model = build_score_residual_student(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, dict(ckpt["style_to_idx"]), dict(ckpt["prompt_to_idx"]), ckpt


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def evaluate_scale(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale: float,
    spectral_kernel: int,
    prediction_mode: str,
    conditioning_mode: str,
    carrier_temperature: float,
    carrier_strength: float,
    carrier_highpass_gain: float,
) -> Dict[str, float]:
    metric_rows: List[Dict[str, float]] = []
    extra_rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            z_t = batch["z_t"].to(device)
            hard = batch["hard"].to(device)
            teacher = batch["teacher"].to(device)
            timestep = batch["timestep"].to(device)
            style_idx = batch["style_idx"].to(device)
            prompt_idx = batch["prompt_idx"].to(device)
            heights = batch["height"].to(device)
            widths = batch["width"].to(device)
            height = int(heights[0].item())
            width = int(widths[0].item())
            if not torch.all(heights == height).item() or not torch.all(widths == width).item():
                raise ValueError("All samples in a batch must share height/width.")

            reference_code = batch["reference_code"].to(device)
            if conditioning_mode == "image_code" and reference_code.shape[-1] == 0:
                raise ValueError("image_code conditioning requires reference_code in every dump")
            raw_correction = model(
                z_t,
                timestep,
                style_idx,
                prompt_idx,
                image_code=reference_code if conditioning_mode == "image_code" else None,
            )
            rendered, _ = compose_prediction(
                raw_correction,
                z_t,
                hard,
                prediction_mode,
                carrier_temperature=carrier_temperature,
                carrier_strength=carrier_strength,
                carrier_highpass_gain=carrier_highpass_gain,
            )
            if prediction_mode == "carrier_anchored":
                pred = float(scale) * rendered
                correction = rendered
                base = torch.zeros_like(teacher)
            elif prediction_mode == "hard_delta":
                pred = hard + float(scale) * raw_correction
                correction = raw_correction
                base = hard
            else:
                pred = float(scale) * raw_correction
                correction = raw_correction
                base = torch.zeros_like(teacher)
            metric_rows.append(batch_metrics(pred, hard, teacher, height, width, spectral_kernel))

            target_delta = teacher - base
            correction_f = flatten_batch(correction.float())
            target_delta_f = flatten_batch(target_delta.float())
            scaled_delta_f = float(scale) * correction_f
            hard_f = flatten_batch(base.float())
            teacher_f = flatten_batch(teacher.float())
            pred_f = flatten_batch(pred.float())
            extra_rows.append(
                {
                    "delta_mse": F.mse_loss(scaled_delta_f, target_delta_f).item(),
                    "raw_delta_mse": F.mse_loss(correction_f, target_delta_f).item(),
                    "delta_cosine": F.cosine_similarity(scaled_delta_f, target_delta_f, dim=-1).mean().item(),
                    "raw_delta_cosine": F.cosine_similarity(correction_f, target_delta_f, dim=-1).mean().item(),
                    "delta_norm_ratio": (
                        torch.linalg.norm(scaled_delta_f, dim=-1).mean()
                        / torch.linalg.norm(target_delta_f, dim=-1).mean().clamp_min(1e-8)
                    ).item(),
                    "raw_delta_norm_ratio": (
                        torch.linalg.norm(correction_f, dim=-1).mean()
                        / torch.linalg.norm(target_delta_f, dim=-1).mean().clamp_min(1e-8)
                    ).item(),
                    "pred_vs_hard_delta_norm": (
                        torch.linalg.norm(pred_f - hard_f, dim=-1).mean()
                        / torch.linalg.norm(teacher_f - hard_f, dim=-1).mean().clamp_min(1e-8)
                    ).item(),
                }
            )
    out = mean_dict(metric_rows)
    out.update(mean_dict(extra_rows))
    out["student_scale"] = float(scale)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.student_scales)
    rows = load_jsonl(Path(args.manifest))
    device = choose_device(args.device)
    model, style_to_idx, prompt_to_idx, ckpt = load_checkpoint(Path(args.checkpoint), device)

    dataset = ScoreResidualDataset(rows, PROJECT_ROOT, style_to_idx, prompt_to_idx)
    train_indices, val_indices, split_meta = split_indices_from_rows(
        rows,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        seed=args.seed,
        holdout_value=args.holdout_value,
    )
    if not val_indices:
        raise ValueError("Scale sweep requires a validation split.")
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}_scale_sweep.csv"
    summary_path = out_dir / f"{args.out_prefix}_scale_sweep_summary.json"

    sweep_rows: List[Dict[str, object]] = []
    for scale in scales:
        metrics = evaluate_scale(
            model,
            loader,
            device=device,
            scale=scale,
            spectral_kernel=args.spectral_kernel,
            prediction_mode=str(ckpt.get("prediction_mode", "hard_delta")),
            conditioning_mode=str(ckpt.get("config", {}).get("conditioning_mode", "style_id")),
            carrier_temperature=float(ckpt.get("carrier", {}).get("temperature", 0.1)),
            carrier_strength=float(ckpt.get("carrier", {}).get("strength", 0.75)),
            carrier_highpass_gain=float(ckpt.get("carrier", {}).get("highpass_gain", 0.0)),
        )
        row = {"checkpoint": str(args.checkpoint), **metrics}
        sweep_rows.append(row)
        print(
            "scale=%.4g val_mse=%.6f hard=%.6f improvement=%.6f delta_norm_ratio=%.3f"
            % (
                scale,
                metrics["mse"],
                metrics["hard_baseline_mse"],
                metrics["hard_mse_improvement"],
                metrics["delta_norm_ratio"],
            ),
            flush=True,
        )

    best = min(sweep_rows, key=lambda row: float(row["mse"]))
    write_csv(csv_path, sweep_rows)
    summary = {
        "mode": "score_residual_student_scale_sweep",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metric": ckpt.get("metric"),
        "prediction_mode": ckpt.get("prediction_mode", "hard_delta"),
        "conditioning_mode": ckpt.get("config", {}).get("conditioning_mode", "style_id"),
        "carrier": ckpt.get("carrier"),
        "split": split_meta,
        "num_samples": len(dataset),
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "scales": scales,
        "best": best,
        "csv_path": str(csv_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"OK: wrote scale sweep -> {csv_path}")
    print(f"OK: wrote summary -> {summary_path}")
    print("BEST scale=%.4g val_mse=%.6f" % (float(best["student_scale"]), float(best["mse"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
