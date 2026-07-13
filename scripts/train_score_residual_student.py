"""Train a direct student from z_t to style score/velocity residuals."""

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

from src.carrier_score import affinity_transport_cosine, content_affinity_transport
from src.latent_ops import highpass_corrected_teacher, spatial_highpass_tokens
from src.score_residual_student import ScoreResidualStudentConfig, build_score_residual_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct z_t -> style residual student.")
    parser.add_argument("--manifest", required=True, help="JSONL manifest from build_score_residual_manifest.py.")
    parser.add_argument("--out-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--cond-dim", type=int, default=128)
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--model-type",
        choices=["token_mlp", "token_mixer"],
        default="token_mlp",
        help="Student architecture. token_mixer adds spatial mixing on the latent grid.",
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=["style_id", "image_code"],
        default="style_id",
        help="Use a memorized style ID or a frozen reference-image code.",
    )
    parser.add_argument("--mixer-layers", type=int, default=4)
    parser.add_argument("--mixer-kernel", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["random", "style", "prompt", "timestep"],
        default="random",
        help="Validation split strategy. Group modes hold out all rows matching --holdout-value.",
    )
    parser.add_argument(
        "--holdout-value",
        default=None,
        help=(
            "Held-out group value for --split-mode style/prompt/timestep. "
            "For timestep, use the printed decimal value or an integer prefix such as 363."
        ),
    )
    parser.add_argument("--cosine-weight", type=float, default=0.05)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--spectral-kernel", type=int, default=5)
    parser.add_argument(
        "--prediction-mode",
        choices=["residual", "hard_delta", "carrier_anchored"],
        default="residual",
        help=(
            "residual: model predicts the full teacher residual. "
            "hard_delta: model predicts a z_t-conditioned correction added to hard_scfg. "
            "carrier_anchored: model predicts native goal, then content affinity renders it spatially."
        ),
    )
    parser.add_argument("--carrier-temperature", type=float, default=0.1)
    parser.add_argument("--carrier-strength", type=float, default=0.75)
    parser.add_argument("--carrier-highpass-gain", type=float, default=0.0)
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


class ScoreResidualDataset(Dataset):
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
        latents = dump.get("latents")
        if not isinstance(latents, dict) or "z_t" not in latents:
            raise ValueError(f"Row {idx} dump missing latents.z_t: {row['dump_path']}")

        z_t = latents["z_t"].detach().float().cpu()
        reference_code = dump.get("reference_code")
        if reference_code is None:
            reference_code = torch.zeros(0, dtype=torch.float32)
        else:
            reference_code = reference_code.detach().float().cpu().reshape(-1)
        target_cfg = row.get("target", row.get("teacher", {}))
        kind = str(target_cfg.get("kind", "highpass_delta"))
        target_name = str(target_cfg.get("name", ""))
        if target_name in fields:
            teacher = fields[target_name].detach().float().cpu()
        elif kind in {"native_delta_velocity", "native_delta_score"}:
            teacher = fields[kind].detach().float().cpu()
        else:
            teacher = None

        if teacher is None:
            hard = fields["hard_scfg"].detach().float().cpu()
            ot = fields["otcfg"].detach().float().cpu()
        else:
            hard = fields.get("hard_scfg", torch.zeros_like(teacher)).detach().float().cpu()
            ot = fields.get("otcfg", torch.zeros_like(teacher)).detach().float().cpu()
        if z_t.ndim == 3 and z_t.shape[0] == 1:
            z_t = z_t[0]
        if hard.ndim == 3 and hard.shape[0] == 1:
            hard = hard[0]
        if ot.ndim == 3 and ot.shape[0] == 1:
            ot = ot[0]
        if teacher is not None and teacher.ndim == 3 and teacher.shape[0] == 1:
            teacher = teacher[0]
        if z_t.shape != hard.shape or hard.shape != ot.shape or (teacher is not None and teacher.shape != z_t.shape):
            raise ValueError(
                f"shape mismatch for row {idx}: z_t={tuple(z_t.shape)} "
                f"hard={tuple(hard.shape)} ot={tuple(ot.shape)} "
                f"teacher={None if teacher is None else tuple(teacher.shape)}"
            )

        if teacher is not None:
            pass
        elif kind == "hard":
            teacher = hard
        elif kind == "ot":
            teacher = ot
        elif kind == "highpass_delta":
            teacher_3d = highpass_corrected_teacher(
                hard[None, :, :],
                ot[None, :, :],
                height=int(row["height"]),
                width=int(row["width"]),
                lamb=float(target_cfg.get("lambda", 1.0)),
                kernel_size=int(target_cfg.get("highpass_kernel", 5)),
            )
            teacher = teacher_3d[0]
        else:
            raise ValueError(f"Unsupported target kind: {kind}")

        style_id = str(row["style_id"])
        prompt_id = str(row["prompt_id"])
        return {
            "z_t": z_t,
            "hard": hard,
            "teacher": teacher.detach().float().cpu(),
            "timestep": torch.tensor(float(row["timestep"]), dtype=torch.float32),
            "style_idx": torch.tensor(self.style_to_idx[style_id], dtype=torch.long),
            "prompt_idx": torch.tensor(self.prompt_to_idx[prompt_id], dtype=torch.long),
            "sample_id": torch.tensor(int(row.get("sample_id", idx)), dtype=torch.long),
            "height": torch.tensor(int(row["height"]), dtype=torch.long),
            "width": torch.tensor(int(row["width"]), dtype=torch.long),
            "reference_code": reference_code,
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


def row_group_value(row: dict, split_mode: str) -> str:
    if split_mode == "style":
        return str(row["style_id"])
    if split_mode == "prompt":
        return str(row["prompt_id"])
    if split_mode == "timestep":
        return f"{float(row['timestep']):.6f}"
    raise ValueError(f"Unsupported group split mode: {split_mode}")


def split_indices_from_rows(
    rows: Sequence[dict],
    split_mode: str,
    val_fraction: float,
    seed: int,
    holdout_value: str | None,
) -> tuple[List[int], List[int], Dict[str, object]]:
    if split_mode == "random":
        train_indices, val_indices = split_indices(len(rows), val_fraction, seed)
        return train_indices, val_indices, {
            "split_mode": split_mode,
            "holdout_value": None,
            "groups": {},
        }

    groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row_group_value(row, split_mode), []).append(idx)
    if not groups:
        raise ValueError("No groups available for split.")

    if holdout_value is None:
        selected = sorted(groups.keys())[-1]
    else:
        selected = str(holdout_value)
        if split_mode == "timestep" and selected not in groups:
            matches = [key for key in groups if key.startswith(selected)]
            if len(matches) == 1:
                selected = matches[0]
        if selected not in groups:
            raise ValueError(f"Holdout value {holdout_value!r} not found in groups: {sorted(groups)}")

    val_indices = sorted(groups[selected])
    val_set = set(val_indices)
    train_indices = [idx for idx in range(len(rows)) if idx not in val_set]
    if not train_indices or not val_indices:
        raise ValueError(f"Invalid split: train={len(train_indices)} val={len(val_indices)}")
    return train_indices, val_indices, {
        "split_mode": split_mode,
        "holdout_value": selected,
        "groups": {key: len(value) for key, value in sorted(groups.items())},
    }


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def compose_prediction(
    raw_pred: torch.Tensor,
    z_t: torch.Tensor,
    hard: torch.Tensor,
    prediction_mode: str,
    *,
    carrier_temperature: float = 0.1,
    carrier_strength: float = 0.75,
    carrier_highpass_gain: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rendered prediction and the field used by the model loss.

    ``carrier_anchored`` predicts a native goal field first. The detached
    content affinity is then applied as the spatial renderer, so the reference
    teacher supplies what to draw while ``z_t`` supplies where to draw it.
    """

    if prediction_mode == "residual":
        return raw_pred, raw_pred
    if prediction_mode == "hard_delta":
        return hard + raw_pred, raw_pred
    if prediction_mode == "carrier_anchored":
        rendered, _ = content_affinity_transport(
            raw_pred,
            z_t,
            temperature=carrier_temperature,
            strength=carrier_strength,
            highpass_gain=carrier_highpass_gain,
        )
        return rendered, rendered
    raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")


def batch_metrics(
    pred: torch.Tensor,
    hard: torch.Tensor,
    teacher: torch.Tensor,
    height: int,
    width: int,
    spectral_kernel: int,
) -> Dict[str, float]:
    pred_f = flatten_batch(pred.float())
    hard_f = flatten_batch(hard.float())
    teacher_f = flatten_batch(teacher.float())
    zero_f = torch.zeros_like(teacher_f)
    pred_mse = F.mse_loss(pred_f, teacher_f).item()
    hard_mse = F.mse_loss(hard_f, teacher_f).item()
    zero_mse = F.mse_loss(zero_f, teacher_f).item()
    cosine = F.cosine_similarity(pred_f, teacher_f, dim=-1).mean().item()
    hard_cosine = F.cosine_similarity(hard_f, teacher_f, dim=-1).mean().item()
    pred_norm = torch.linalg.norm(pred_f, dim=-1).mean().item()
    teacher_norm = torch.linalg.norm(teacher_f, dim=-1).mean().item()
    norm_ratio = pred_norm / max(teacher_norm, 1e-8)
    if height * width == pred.shape[1]:
        pred_hi = spatial_highpass_tokens(pred, height=height, width=width, kernel_size=spectral_kernel)
        teacher_hi = spatial_highpass_tokens(teacher, height=height, width=width, kernel_size=spectral_kernel)
        hi_mse = F.mse_loss(pred_hi.float(), teacher_hi.float()).item()
    else:
        hi_mse = float("nan")
    return {
        "mse": pred_mse,
        "hard_baseline_mse": hard_mse,
        "zero_baseline_mse": zero_mse,
        "hard_mse_improvement": hard_mse - pred_mse,
        "zero_mse_improvement": zero_mse - pred_mse,
        "cosine": cosine,
        "hard_baseline_cosine": hard_cosine,
        "pred_norm": pred_norm,
        "teacher_norm": teacher_norm,
        "norm_ratio": norm_ratio,
        "highpass_mse": hi_mse,
    }


def mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cosine_weight: float,
    spectral_weight: float,
    spectral_kernel: int,
    prediction_mode: str,
    conditioning_mode: str,
    carrier_temperature: float,
    carrier_strength: float,
    carrier_highpass_gain: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_rows: List[Dict[str, float]] = []
    loss_rows: List[Dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
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
                raise ValueError("All samples in a batch must share height/width for spectral loss.")

            reference_code = batch["reference_code"].to(device)
            if conditioning_mode == "image_code" and reference_code.shape[-1] == 0:
                raise ValueError("image_code conditioning requires reference_code in every dump")
            raw_pred = model(
                z_t,
                timestep,
                style_idx,
                prompt_idx,
                image_code=reference_code if conditioning_mode == "image_code" else None,
            )
            pred, train_pred = compose_prediction(
                raw_pred,
                z_t,
                hard,
                prediction_mode,
                carrier_temperature=carrier_temperature,
                carrier_strength=carrier_strength,
                carrier_highpass_gain=carrier_highpass_gain,
            )
            target = teacher if prediction_mode != "hard_delta" else teacher - hard

            mse = F.mse_loss(pred.float(), teacher.float())
            train_target = teacher if prediction_mode != "hard_delta" else target
            cos_loss = 1.0 - F.cosine_similarity(
                flatten_batch(train_pred.float()),
                flatten_batch(train_target.float()),
                dim=-1,
            ).mean()
            loss = mse + float(cosine_weight) * cos_loss

            spectral_loss = torch.tensor(0.0, device=device)
            if spectral_weight > 0.0 and height * width == pred.shape[1]:
                pred_hi = spatial_highpass_tokens(train_pred, height=height, width=width, kernel_size=spectral_kernel)
                teacher_hi = spatial_highpass_tokens(train_target, height=height, width=width, kernel_size=spectral_kernel)
                spectral_loss = F.mse_loss(pred_hi.float(), teacher_hi.float())
                loss = loss + float(spectral_weight) * spectral_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            metric_rows.append(batch_metrics(pred.detach(), hard, teacher, height, width, spectral_kernel))
            loss_rows.append(
                {
                    "loss": float(loss.detach().item()),
                    "mse_loss": float(mse.detach().item()),
                    "cosine_loss": float(cos_loss.detach().item()),
                    "spectral_loss": float(spectral_loss.detach().item()),
                    "raw_pred_norm": float(torch.linalg.norm(flatten_batch(raw_pred.detach().float()), dim=-1).mean().item()),
                    "carrier_transport_cosine": (
                        float(affinity_transport_cosine(raw_pred.detach(), pred.detach()).mean().item())
                        if prediction_mode == "carrier_anchored"
                        else float("nan")
                    ),
                }
            )

    out = mean_dict(metric_rows)
    out.update(mean_dict(loss_rows))
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest)
    rows = load_jsonl(manifest_path)
    target_kinds = {
        str(row.get("target", row.get("teacher", {})).get("kind", "unknown"))
        for row in rows
    }
    if len(target_kinds) != 1:
        raise ValueError(f"Manifest must use one target kind per run, got {sorted(target_kinds)}")
    target_kind = next(iter(target_kinds))
    style_ids = sorted({str(row["style_id"]) for row in rows})
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    style_to_idx = {style_id: idx for idx, style_id in enumerate(style_ids)}
    prompt_to_idx = {prompt_id: idx for idx, prompt_id in enumerate(prompt_ids)}

    first_dump = load_dump(resolve_path(str(rows[0]["dump_path"]), PROJECT_ROOT))
    z_t = first_dump.get("latents", {}).get("z_t")
    if z_t is None:
        raise ValueError("First dump missing latents.z_t; use build_score_residual_manifest.py.")
    latent_dim = int(z_t.shape[-1])
    max_tokens = int(z_t.shape[-2])
    grid_height = int(rows[0]["height"])
    grid_width = int(rows[0]["width"])
    first_reference_code = first_dump.get("reference_code")
    if args.conditioning_mode == "image_code":
        if first_reference_code is None:
            raise ValueError("image_code conditioning requires dumps with reference_code")
        image_code_dim = int(first_reference_code.numel())
        if image_code_dim <= 0:
            raise ValueError("reference_code must be non-empty")
    else:
        image_code_dim = 256

    dataset = ScoreResidualDataset(rows, PROJECT_ROOT, style_to_idx, prompt_to_idx)
    train_indices, val_indices, split_meta = split_indices_from_rows(
        rows,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        seed=args.seed,
        holdout_value=args.holdout_value,
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if val_indices
        else None
    )

    device = choose_device(args.device)
    config = ScoreResidualStudentConfig(
        latent_dim=latent_dim,
        max_tokens=max_tokens,
        num_styles=len(style_ids),
        num_prompts=len(prompt_ids),
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        time_embed_dim=args.time_embed_dim,
        dropout=args.dropout,
        model_type=args.model_type,
        mixer_layers=args.mixer_layers,
        mixer_kernel=args.mixer_kernel,
        grid_height=grid_height,
        grid_width=grid_width,
        conditioning_mode=args.conditioning_mode,
        image_code_dim=image_code_dim,
    )
    model = build_score_residual_student(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows: List[Dict[str, object]] = []
    best_metric = float("inf")
    best_epoch = -1

    run_config = {
        "manifest": str(manifest_path),
        "num_samples": len(dataset),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "split": split_meta,
        "style_to_idx": style_to_idx,
        "prompt_to_idx": prompt_to_idx,
        "device": str(device),
        "model": asdict(config),
        "args": vars(args),
        "target_kind": target_kind,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            cosine_weight=args.cosine_weight,
            spectral_weight=args.spectral_weight,
            spectral_kernel=args.spectral_kernel,
            prediction_mode=args.prediction_mode,
            conditioning_mode=args.conditioning_mode,
            carrier_temperature=args.carrier_temperature,
            carrier_strength=args.carrier_strength,
            carrier_highpass_gain=args.carrier_highpass_gain,
        )
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                cosine_weight=args.cosine_weight,
                spectral_weight=args.spectral_weight,
                spectral_kernel=args.spectral_kernel,
                prediction_mode=args.prediction_mode,
                conditioning_mode=args.conditioning_mode,
                carrier_temperature=args.carrier_temperature,
                carrier_strength=args.carrier_strength,
                carrier_highpass_gain=args.carrier_highpass_gain,
            )
            if val_loader is not None
            else {}
        )
        row: Dict[str, object] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        log_rows.append(row)

        metric = float(val_metrics.get("mse", train_metrics["mse"]))
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "style_to_idx": style_to_idx,
                    "prompt_to_idx": prompt_to_idx,
                    "epoch": epoch,
                    "metric": best_metric,
                    "prediction_mode": args.prediction_mode,
                    "target_kind": target_kind,
                    "carrier": {
                        "temperature": args.carrier_temperature,
                        "strength": args.carrier_strength,
                        "highpass_gain": args.carrier_highpass_gain,
                    },
                },
                out_dir / "student_best.pt",
            )

        if epoch % args.log_every == 0 or epoch == args.epochs:
            val_text = f" val_mse={val_metrics['mse']:.6f}" if val_metrics else ""
            print(
                f"epoch={epoch} train_mse={train_metrics['mse']:.6f}"
                f" train_hard_base={train_metrics['hard_baseline_mse']:.6f}{val_text}",
                flush=True,
            )

    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "style_to_idx": style_to_idx,
            "prompt_to_idx": prompt_to_idx,
            "epoch": args.epochs,
            "metric": best_metric,
            "prediction_mode": args.prediction_mode,
            "target_kind": target_kind,
            "carrier": {
                "temperature": args.carrier_temperature,
                "strength": args.carrier_strength,
                "highpass_gain": args.carrier_highpass_gain,
            },
        },
        out_dir / "student_last.pt",
    )
    write_csv(out_dir / "train_log.csv", log_rows)

    final = log_rows[-1]
    metrics = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "final": final,
        "num_samples": len(dataset),
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "split": split_meta,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"OK: wrote checkpoints and metrics -> {out_dir}")
    print(f"best_epoch={best_epoch} best_metric={best_metric:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
