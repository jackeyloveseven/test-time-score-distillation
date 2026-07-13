"""Evaluate a trained residual adapter with student-scale sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_ops import highpass_corrected_teacher, spatial_highpass_tokens
from src.student_adapter import LowRankResidualAdapter, ResidualAdapter, ResidualAdapterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual adapter scale sweep.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-prefix", required=True, help="Output prefix under results-dir unless absolute.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--student-scales", default="0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--spectral-kernel", type=int, default=5)
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def load_dump(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


class ResidualDistillDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        project_root: Path,
        style_to_idx: Dict[str, int],
        prompt_to_idx: Dict[str, int],
    ) -> None:
        self.rows = list(rows)
        self.project_root = project_root
        self.style_to_idx = style_to_idx
        self.prompt_to_idx = prompt_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        dump = load_dump(resolve_path(str(row["dump_path"]), self.project_root))
        fields_d = dump["fields"]
        hard = fields_d["hard_scfg"].detach().float().cpu()
        ot = fields_d["otcfg"].detach().float().cpu()
        if hard.ndim == 3 and hard.shape[0] == 1:
            hard = hard[0]
        if ot.ndim == 3 and ot.shape[0] == 1:
            ot = ot[0]
        teacher_cfg = row.get("teacher", {})
        kind = str(teacher_cfg.get("kind", "highpass_delta"))
        if kind == "hard":
            teacher = hard
        elif kind == "ot":
            teacher = ot
        elif kind == "highpass_delta":
            teacher = highpass_corrected_teacher(
                hard[None, :, :],
                ot[None, :, :],
                height=int(row["height"]),
                width=int(row["width"]),
                lamb=float(teacher_cfg.get("lambda", 1.0)),
                kernel_size=int(teacher_cfg.get("highpass_kernel", 5)),
            )[0]
        else:
            raise ValueError(f"Unsupported teacher kind: {kind}")

        style_id = str(row["style_id"])
        prompt_id = str(row["prompt_id"])
        if style_id not in self.style_to_idx:
            raise KeyError(f"Style id {style_id} missing from checkpoint mapping")
        if prompt_id not in self.prompt_to_idx:
            raise KeyError(f"Prompt id {prompt_id} missing from checkpoint mapping")
        return {
            "hard": hard,
            "teacher": teacher.detach().float().cpu(),
            "timestep": torch.tensor(float(row["timestep"]), dtype=torch.float32),
            "style_idx": torch.tensor(self.style_to_idx[style_id], dtype=torch.long),
            "prompt_idx": torch.tensor(self.prompt_to_idx[prompt_id], dtype=torch.long),
            "height": torch.tensor(int(row["height"]), dtype=torch.long),
            "width": torch.tensor(int(row["width"]), dtype=torch.long),
        }


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[List[int], List[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_n = int(round(n * val_fraction))
    if n > 1 and val_fraction > 0.0:
        val_n = min(max(val_n, 1), n - 1)
    else:
        val_n = 0
    return indices[val_n:], indices[:val_n]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def make_config(raw_config: dict) -> ResidualAdapterConfig:
    valid = {field.name for field in fields(ResidualAdapterConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid}
    return ResidualAdapterConfig(**filtered)


def infer_adapter_type(state: dict) -> str:
    if any(key.startswith("coeff_mlp.") or key == "basis" for key in state):
        return "lowrank"
    return "mlp"


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = make_config(ckpt["config"])
    state = ckpt["model_state"]
    adapter_type = infer_adapter_type(state)
    model = LowRankResidualAdapter(config) if adapter_type == "lowrank" else ResidualAdapter(config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    ckpt["adapter_type"] = adapter_type
    return model, ckpt


def mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def eval_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scales: Sequence[float],
    spectral_kernel: int,
) -> List[dict]:
    by_scale: Dict[float, List[Dict[str, float]]] = {float(scale): [] for scale in scales}
    with torch.no_grad():
        for batch in loader:
            hard = batch["hard"].to(device)
            teacher = batch["teacher"].to(device)
            timestep = batch["timestep"].to(device)
            style_idx = batch["style_idx"].to(device)
            prompt_idx = batch["prompt_idx"].to(device)
            height = int(batch["height"][0].item())
            width = int(batch["width"][0].item())

            pred_delta = model(hard, timestep, style_idx, prompt_idx).float()
            target_delta = teacher.float() - hard.float()
            hard_f = flatten_batch(hard.float())
            teacher_f = flatten_batch(teacher.float())
            baseline_mse = F.mse_loss(hard_f, teacher_f).item()
            baseline_cos = F.cosine_similarity(hard_f, teacher_f, dim=-1).mean().item()

            for scale in scales:
                scale_f = float(scale)
                pred_teacher = hard.float() + scale_f * pred_delta
                pred_f = flatten_batch(pred_teacher)
                mse = F.mse_loss(pred_f, teacher_f).item()
                cosine = F.cosine_similarity(pred_f, teacher_f, dim=-1).mean().item()
                scaled_delta = scale_f * pred_delta
                pred_delta_norm = torch.linalg.norm(flatten_batch(scaled_delta), dim=-1).mean().item()
                target_delta_norm = torch.linalg.norm(flatten_batch(target_delta), dim=-1).mean().item()
                if height * width == hard.shape[1]:
                    pred_hi = spatial_highpass_tokens(
                        scaled_delta,
                        height=height,
                        width=width,
                        kernel_size=spectral_kernel,
                    )
                    target_hi = spatial_highpass_tokens(
                        target_delta,
                        height=height,
                        width=width,
                        kernel_size=spectral_kernel,
                    )
                    hi_mse = F.mse_loss(pred_hi, target_hi).item()
                else:
                    hi_mse = float("nan")
                by_scale[scale_f].append(
                    {
                        "mse": mse,
                        "baseline_mse": baseline_mse,
                        "mse_improvement": baseline_mse - mse,
                        "cosine": cosine,
                        "baseline_cosine": baseline_cos,
                        "delta_norm": pred_delta_norm,
                        "target_delta_norm": target_delta_norm,
                        "delta_norm_ratio": pred_delta_norm / max(target_delta_norm, 1e-8),
                        "highpass_delta_mse": hi_mse,
                    }
                )
    rows = []
    for scale in scales:
        row = {"student_scale": float(scale)}
        row.update(mean_dict(by_scale[float(scale)]))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest, PROJECT_ROOT)
    checkpoint_path = resolve_path(args.checkpoint, PROJECT_ROOT)
    scales = parse_float_list(args.student_scales)
    if not scales:
        raise ValueError("--student-scales must not be empty")

    device = choose_device(args.device)
    model, ckpt = load_model(checkpoint_path, device)
    rows = load_jsonl(manifest_path)
    dataset = ResidualDistillDataset(
        rows,
        PROJECT_ROOT,
        style_to_idx=ckpt["style_to_idx"],
        prompt_to_idx=ckpt["prompt_to_idx"],
    )
    train_indices, val_indices = split_indices(len(rows), args.val_fraction, args.seed)
    split_to_indices = {
        "all": list(range(len(rows))),
        "train": train_indices,
        "val": val_indices,
    }

    results: Dict[str, List[dict]] = {}
    result_dir = Path(args.results_dir)
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = result_dir / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for split_name, indices in split_to_indices.items():
        if args.split != "all" and split_name != args.split:
            continue
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        split_rows = eval_loader(
            model,
            loader,
            device,
            scales=scales,
            spectral_kernel=args.spectral_kernel,
        )
        for row in split_rows:
            row["split"] = split_name
            row["num_samples"] = len(indices)
        results[split_name] = split_rows
        write_csv(out_prefix.parent / f"{out_prefix.name}_{split_name}_scale_sweep.csv", split_rows)

    all_rows = [row for split_rows in results.values() for row in split_rows]
    write_csv(out_prefix.parent / f"{out_prefix.name}_scale_sweep.csv", all_rows)
    val_rows = results.get("val") or results.get("all") or all_rows
    best_mse = min(val_rows, key=lambda row: row["mse"])
    best_cosine = max(val_rows, key=lambda row: row["cosine"])
    summary = {
        "mode": "evaluate_residual_adapter",
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "adapter_type": ckpt["adapter_type"],
        "checkpoint_epoch": ckpt.get("epoch"),
        "num_samples": len(rows),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "student_scales": scales,
        "best_by_val_mse": best_mse,
        "best_by_val_cosine": best_cosine,
        "csv": str(out_prefix.parent / f"{out_prefix.name}_scale_sweep.csv"),
    }
    with (out_prefix.parent / f"{out_prefix.name}_scale_sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"OK: wrote {out_prefix.parent / f'{out_prefix.name}_scale_sweep.csv'}")
    print(f"OK: wrote {out_prefix.parent / f'{out_prefix.name}_scale_sweep_summary.json'}")
    print(f"BEST_MSE scale={best_mse['student_scale']} mse={best_mse['mse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
