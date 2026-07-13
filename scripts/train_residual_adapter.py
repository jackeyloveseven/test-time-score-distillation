"""Train a small adapter from hard CFG residuals to selected teacher residuals."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
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
    parser = argparse.ArgumentParser(description="Train residual adapter on a distillation manifest.")
    parser.add_argument("--manifest", required=True, help="JSONL manifest from build_distill_manifest.py.")
    parser.add_argument("--out-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter-type", choices=["mlp", "lowrank"], default="mlp")
    parser.add_argument("--low-rank-dim", type=int, default=32)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--cosine-weight", type=float, default=0.05)
    parser.add_argument("--delta-weight", type=float, default=0.0)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--spectral-kernel", type=int, default=5)
    parser.add_argument("--delta-norm-weight", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


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
        fields = dump["fields"]
        hard = fields["hard_scfg"].detach().float().cpu()
        ot = fields["otcfg"].detach().float().cpu()
        if hard.ndim == 3 and hard.shape[0] == 1:
            hard = hard[0]
        if ot.ndim == 3 and ot.shape[0] == 1:
            ot = ot[0]
        if hard.shape != ot.shape:
            raise ValueError(f"hard/ot shape mismatch for row {idx}: {tuple(hard.shape)} vs {tuple(ot.shape)}")

        teacher_cfg = row.get("teacher", {})
        kind = str(teacher_cfg.get("kind", "highpass_delta"))
        if kind == "hard":
            teacher = hard
        elif kind == "ot":
            teacher = ot
        elif kind == "highpass_delta":
            teacher_3d = highpass_corrected_teacher(
                hard[None, :, :],
                ot[None, :, :],
                height=int(row["height"]),
                width=int(row["width"]),
                lamb=float(teacher_cfg.get("lambda", 1.0)),
                kernel_size=int(teacher_cfg.get("highpass_kernel", 5)),
            )
            teacher = teacher_3d[0]
        else:
            raise ValueError(f"Unsupported teacher kind: {kind}")

        style_id = str(row["style_id"])
        prompt_id = str(row["prompt_id"])
        return {
            "hard": hard,
            "teacher": teacher.detach().float().cpu(),
            "timestep": torch.tensor(float(row["timestep"]), dtype=torch.float32),
            "style_idx": torch.tensor(self.style_to_idx[style_id], dtype=torch.long),
            "prompt_idx": torch.tensor(self.prompt_to_idx[prompt_id], dtype=torch.long),
            "sample_id": torch.tensor(int(row.get("sample_id", idx)), dtype=torch.long),
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


def batch_metrics(
    pred_teacher: torch.Tensor,
    hard: torch.Tensor,
    teacher: torch.Tensor,
    height: int,
    width: int,
    spectral_kernel: int,
) -> Dict[str, float]:
    pred_f = flatten_batch(pred_teacher.float())
    hard_f = flatten_batch(hard.float())
    teacher_f = flatten_batch(teacher.float())
    pred_delta = pred_teacher.float() - hard.float()
    target_delta = teacher.float() - hard.float()
    pred_mse = F.mse_loss(pred_f, teacher_f).item()
    base_mse = F.mse_loss(hard_f, teacher_f).item()
    cosine = F.cosine_similarity(pred_f, teacher_f, dim=-1).mean().item()
    base_cosine = F.cosine_similarity(hard_f, teacher_f, dim=-1).mean().item()
    pred_delta_f = flatten_batch(pred_delta)
    target_delta_f = flatten_batch(target_delta)
    pred_delta_norm = torch.linalg.norm(pred_delta_f, dim=-1).mean().item()
    target_delta_norm = torch.linalg.norm(target_delta_f, dim=-1).mean().item()
    norm_ratio = pred_delta_norm / max(target_delta_norm, 1e-8)
    if height * width == pred_teacher.shape[1]:
        pred_hi = spatial_highpass_tokens(
            pred_delta,
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
        hi_mse = F.mse_loss(pred_hi.float(), target_hi.float()).item()
    else:
        hi_mse = float("nan")
    return {
        "mse": pred_mse,
        "baseline_mse": base_mse,
        "mse_improvement": base_mse - pred_mse,
        "cosine": cosine,
        "baseline_cosine": base_cosine,
        "delta_norm": pred_delta_norm,
        "target_delta_norm": target_delta_norm,
        "delta_norm_ratio": norm_ratio,
        "highpass_delta_mse": hi_mse,
    }


def mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def run_epoch(
    model: ResidualAdapter,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cosine_weight: float,
    delta_weight: float,
    spectral_weight: float,
    spectral_kernel: int,
    delta_norm_weight: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_rows: List[Dict[str, float]] = []
    loss_rows: List[Dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
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
                raise ValueError("All samples in a batch must share height/width for spectral loss.")

            pred_delta = model(hard, timestep, style_idx, prompt_idx)
            pred_teacher = hard + pred_delta
            target_delta = teacher - hard
            mse = F.mse_loss(pred_teacher.float(), teacher.float())
            delta_mse = F.mse_loss(pred_delta.float(), target_delta.float())
            if height * width == hard.shape[1]:
                pred_hi = spatial_highpass_tokens(
                    pred_delta.float(),
                    height=height,
                    width=width,
                    kernel_size=spectral_kernel,
                )
                target_hi = spatial_highpass_tokens(
                    target_delta.float(),
                    height=height,
                    width=width,
                    kernel_size=spectral_kernel,
                )
                spectral_mse = F.mse_loss(pred_hi, target_hi)
            else:
                spectral_mse = pred_delta.new_tensor(0.0)
            pred_delta_norm = torch.linalg.norm(flatten_batch(pred_delta.float()), dim=-1)
            target_delta_norm = torch.linalg.norm(flatten_batch(target_delta.float()), dim=-1)
            delta_norm_loss = F.mse_loss(
                torch.log(pred_delta_norm.clamp_min(1e-8)),
                torch.log(target_delta_norm.clamp_min(1e-8)),
            )
            cosine_loss = 1.0 - F.cosine_similarity(
                flatten_batch(pred_teacher.float()),
                flatten_batch(teacher.float()),
                dim=-1,
            ).mean()
            loss = (
                mse
                + float(cosine_weight) * cosine_loss
                + float(delta_weight) * delta_mse
                + float(spectral_weight) * spectral_mse
                + float(delta_norm_weight) * delta_norm_loss
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            metric_rows.append(
                batch_metrics(
                    pred_teacher.detach(),
                    hard.detach(),
                    teacher.detach(),
                    height=height,
                    width=width,
                    spectral_kernel=spectral_kernel,
                )
            )
            loss_rows.append(
                {
                    "loss": float(loss.detach().item()),
                    "mse": float(mse.detach().item()),
                    "delta_mse": float(delta_mse.detach().item()),
                    "spectral_mse": float(spectral_mse.detach().item()),
                    "delta_norm_loss": float(delta_norm_loss.detach().item()),
                    "cosine_loss": float(cosine_loss.detach().item()),
                }
            )
    out = mean_dict(metric_rows)
    for key, value in mean_dict(loss_rows).items():
        out[f"loss_{key}"] = value
    return out


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    manifest_path = resolve_path(args.manifest, PROJECT_ROOT)
    rows = load_jsonl(manifest_path)
    style_ids = sorted({str(row["style_id"]) for row in rows})
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    style_to_idx = {value: idx for idx, value in enumerate(style_ids)}
    prompt_to_idx = {value: idx for idx, value in enumerate(prompt_ids)}

    first_shape = rows[0]["field_shape"]
    latent_dim = int(first_shape[-1])
    if any(int(row["field_shape"][-1]) != latent_dim for row in rows):
        raise ValueError("All rows must share latent_dim for the first adapter smoke.")

    train_indices, val_indices = split_indices(len(rows), args.val_fraction, args.seed)
    dataset = ResidualDistillDataset(rows, PROJECT_ROOT, style_to_idx, prompt_to_idx)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = choose_device(args.device)
    config = ResidualAdapterConfig(
        latent_dim=latent_dim,
        num_styles=len(style_ids),
        num_prompts=len(prompt_ids),
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        time_embed_dim=args.time_embed_dim,
        dropout=args.dropout,
        low_rank_dim=args.low_rank_dim,
        use_gate=not args.no_gate,
    )
    model = (LowRankResidualAdapter(config) if args.adapter_type == "lowrank" else ResidualAdapter(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = resolve_path(args.out_dir, PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "args": vars(args),
        "manifest": str(manifest_path),
        "num_samples": len(rows),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "style_to_idx": style_to_idx,
        "prompt_to_idx": prompt_to_idx,
        "model": asdict(config),
        "adapter_type": args.adapter_type,
        "teacher": rows[0].get("teacher", {}),
        "device": str(device),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    logs: List[dict] = []
    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            cosine_weight=args.cosine_weight,
            delta_weight=args.delta_weight,
            spectral_weight=args.spectral_weight,
            spectral_kernel=args.spectral_kernel,
            delta_norm_weight=args.delta_norm_weight,
        )
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                cosine_weight=args.cosine_weight,
                delta_weight=args.delta_weight,
                spectral_weight=args.spectral_weight,
                spectral_kernel=args.spectral_kernel,
                delta_norm_weight=args.delta_norm_weight,
            )
            if val_indices
            else {}
        )
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        logs.append(row)

        val_key = "val_mse" if val_indices else "train_mse"
        current = float(row[val_key])
        if current < best_val:
            best_val = current
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "style_to_idx": style_to_idx,
                    "prompt_to_idx": prompt_to_idx,
                    "epoch": epoch,
                    "metrics": row,
                },
                out_dir / "adapter_best.pt",
            )
        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs):
            msg = f"epoch={epoch} train_mse={row['train_mse']:.6f}"
            if val_indices:
                msg += f" val_mse={row['val_mse']:.6f} val_cos={row['val_cosine']:.6f}"
            print(msg, flush=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "style_to_idx": style_to_idx,
            "prompt_to_idx": prompt_to_idx,
            "epoch": args.epochs,
            "metrics": logs[-1],
        },
        out_dir / "adapter_last.pt",
    )
    write_csv(out_dir / "train_log.csv", logs)
    final_payload = {
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "final": logs[-1],
        "log_path": str(out_dir / "train_log.csv"),
        "best_checkpoint": str(out_dir / "adapter_best.pt"),
        "last_checkpoint": str(out_dir / "adapter_last.pt"),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"OK: wrote {out_dir / 'adapter_best.pt'}")
    print(f"OK: wrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
