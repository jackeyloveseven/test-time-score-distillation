"""Residual field profiling and serialization utilities.

This module is intentionally lightweight and pure tensor based.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, Iterable, List, Mapping, Optional

import torch

from . import field_metrics


@dataclass
class ResidualFieldRecord:
    style_id: str
    prompt_id: str
    timestep: float
    variant: str
    residual_norm: float
    rho: float
    cos_to_hyperion_grad: Optional[float]
    cos_to_hyperion_delta: Optional[float]
    hi_freq_ratio: Optional[float]
    low_freq_ratio: Optional[float]


@dataclass
class ResidualPairRecord:
    style_id: str
    prompt_id: str
    timestep: float
    variant_a: str
    variant_b: str
    cosine: float
    diff_norm: float
    norm_ratio_b_over_a: float
    diff_hi_freq_ratio: Optional[float]
    diff_low_freq_ratio: Optional[float]


def record_to_dict(record: ResidualFieldRecord) -> Dict[str, object]:
    """Convert a record to plain dict for csv/json serialization."""
    return {
        "style_id": record.style_id,
        "prompt_id": record.prompt_id,
        "timestep": record.timestep,
        "variant": record.variant,
        "residual_norm": record.residual_norm,
        "rho": record.rho,
        "cos_to_hyperion_grad": record.cos_to_hyperion_grad,
        "cos_to_hyperion_delta": record.cos_to_hyperion_delta,
        "hi_freq_ratio": record.hi_freq_ratio,
        "low_freq_ratio": record.low_freq_ratio,
    }


def pair_record_to_dict(record: ResidualPairRecord) -> Dict[str, object]:
    """Convert a pairwise record to plain dict for csv/json serialization."""
    return {
        "style_id": record.style_id,
        "prompt_id": record.prompt_id,
        "timestep": record.timestep,
        "variant_a": record.variant_a,
        "variant_b": record.variant_b,
        "cosine": record.cosine,
        "diff_norm": record.diff_norm,
        "norm_ratio_b_over_a": record.norm_ratio_b_over_a,
        "diff_hi_freq_ratio": record.diff_hi_freq_ratio,
        "diff_low_freq_ratio": record.diff_low_freq_ratio,
    }


def write_csv(records: Iterable[ResidualFieldRecord], path: Path | str) -> None:
    """Write all records into csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [record_to_dict(r) for r in records]
    if len(rows) == 0:
        with path.open("w", encoding="utf-8") as f:
            f.write(
                "style_id,prompt_id,timestep,variant,residual_norm,rho,"
                "cos_to_hyperion_grad,cos_to_hyperion_delta,hi_freq_ratio,low_freq_ratio\n"
            )
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_csv(records: Iterable[ResidualPairRecord], path: Path | str) -> None:
    """Write pairwise residual comparison records into csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [pair_record_to_dict(r) for r in records]
    fieldnames = [
        "style_id",
        "prompt_id",
        "timestep",
        "variant_a",
        "variant_b",
        "cosine",
        "diff_norm",
        "norm_ratio_b_over_a",
        "diff_hi_freq_ratio",
        "diff_low_freq_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_summary(records: Iterable[ResidualFieldRecord], path: Path | str) -> Dict[str, object]:
    """Aggregate summary by variant and write JSON summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[ResidualFieldRecord]] = {}
    for r in records:
        grouped.setdefault(r.variant, []).append(r)

    keys = [
        "residual_norm",
        "rho",
        "cos_to_hyperion_grad",
        "cos_to_hyperion_delta",
        "hi_freq_ratio",
        "low_freq_ratio",
    ]

    summary: Dict[str, object] = {"by_variant": {}, "overall": {}}
    overall_vals: Dict[str, List[float]] = {k: [] for k in keys}

    for variant, items in grouped.items():
        variant_summary: Dict[str, object] = {"n": len(items)}
        for key in keys:
            vals = [getattr(i, key) for i in items if getattr(i, key) is not None]
            if vals:
                variant_summary[key] = {
                    "mean": mean(vals),
                    "std": (pvariance(vals) ** 0.5) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
                overall_vals[key].extend(vals)
            else:
                variant_summary[key] = {"mean": None, "std": None, "n": 0}
        summary["by_variant"][variant] = variant_summary

    for key, vals in overall_vals.items():
        if vals:
            summary["overall"][key] = {
                "mean": mean(vals),
                "std": (pvariance(vals) ** 0.5) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        else:
            summary["overall"][key] = {"mean": None, "std": None, "n": 0}

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def write_pairwise_json_summary(records: Iterable[ResidualPairRecord], path: Path | str) -> Dict[str, object]:
    """Aggregate pairwise residual comparison records and write JSON summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[ResidualPairRecord]] = {}
    for r in records:
        pair_key = f"{r.variant_a}__vs__{r.variant_b}"
        grouped.setdefault(pair_key, []).append(r)

    keys = [
        "cosine",
        "diff_norm",
        "norm_ratio_b_over_a",
        "diff_hi_freq_ratio",
        "diff_low_freq_ratio",
    ]

    summary: Dict[str, object] = {"by_pair": {}, "overall": {}}
    overall_vals: Dict[str, List[float]] = {k: [] for k in keys}

    for pair_key, items in grouped.items():
        pair_summary: Dict[str, object] = {"n": len(items)}
        for key in keys:
            vals = [getattr(i, key) for i in items if getattr(i, key) is not None]
            if vals:
                pair_summary[key] = {
                    "mean": mean(vals),
                    "std": (pvariance(vals) ** 0.5) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
                overall_vals[key].extend(vals)
            else:
                pair_summary[key] = {"mean": None, "std": None, "n": 0}
        summary["by_pair"][pair_key] = pair_summary

    for key, vals in overall_vals.items():
        if vals:
            summary["overall"][key] = {
                "mean": mean(vals),
                "std": (pvariance(vals) ** 0.5) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        else:
            summary["overall"][key] = {"mean": None, "std": None, "n": 0}

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _repeat_if_needed(metric: torch.Tensor, target_size: int) -> torch.Tensor:
    if metric.shape[0] == target_size:
        return metric
    if metric.shape[0] == 1:
        return metric.expand(target_size, *metric.shape[1:])
    raise ValueError(f"batch mismatch: target={target_size}, metric={tuple(metric.shape)}")


def _to_btld(field: torch.Tensor) -> torch.Tensor:
    """Normalize reference field to [B, L, D] for metric functions."""
    if field.ndim == 3:
        return field
    if field.ndim == 4:
        return field.mean(dim=1)
    raise ValueError("field must be [B, L, D] or [B, H, L, D]")


def _resolve_teacher(reference: torch.Tensor, residual_batch: int) -> torch.Tensor:
    aligned = _to_btld(reference)
    return _repeat_if_needed(aligned, residual_batch)


def compare_fields(
    fields: Mapping[str, torch.Tensor],
    hyperion_grad: Optional[torch.Tensor] = None,
    hyperion_delta: Optional[torch.Tensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    style_id: str = "unknown",
    prompt_id: str = "unknown",
    timestep: float = 0.0,
) -> List[ResidualFieldRecord]:
    """Compute profiling metrics for each residual field variant.

    Args:
        fields: Dict variant name -> residual field tensor.
        hyperion_grad: optional teacher gradient field.
        hyperion_delta: optional Hyperion displacement field.
        height/width: optional spatial shape for spatial/freq metrics.
        style_id/prompt_id/timestep: metadata attached to each record.
    """
    records: List[ResidualFieldRecord] = []

    for variant, field in fields.items():
        if field.ndim < 2:
            raise ValueError(
                f"field for variant '{variant}' must have at least shape [B, ...], got {tuple(field.shape)}"
            )

        batch = field.shape[0]
        residual_norm = field_metrics.field_norm(field)
        rho = field_metrics.spatial_neighbor_cosine(field, height=height, width=width)
        freq = field_metrics.frequency_energy_ratio(field, height=height, width=width)

        if hyperion_grad is not None:
            grad_ref = _resolve_teacher(hyperion_grad, batch)
            cos_grad = field_metrics.cosine_to(field, grad_ref)
        else:
            cos_grad = None

        if hyperion_delta is not None:
            delta_ref = _resolve_teacher(hyperion_delta, batch)
            cos_delta = field_metrics.cosine_to(field, delta_ref)
        else:
            cos_delta = None

        low_ratio = freq["low_freq_ratio"]
        high_ratio = freq["high_freq_ratio"]

        for i in range(batch):
            records.append(
                ResidualFieldRecord(
                    style_id=style_id,
                    prompt_id=prompt_id,
                    timestep=float(timestep),
                    variant=variant,
                    residual_norm=float(residual_norm[i].detach().cpu().item()),
                    rho=float(rho[i].detach().cpu().item()),
                    cos_to_hyperion_grad=None if cos_grad is None else float(cos_grad[i].detach().cpu().item()),
                    cos_to_hyperion_delta=None if cos_delta is None else float(cos_delta[i].detach().cpu().item()),
                    hi_freq_ratio=float(high_ratio[i].detach().cpu().item()),
                    low_freq_ratio=float(low_ratio[i].detach().cpu().item()),
                )
            )

    return records


def compare_field_pairs(
    fields: Mapping[str, torch.Tensor],
    pairs: Iterable[tuple[str, str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    style_id: str = "unknown",
    prompt_id: str = "unknown",
    timestep: float = 0.0,
) -> List[ResidualPairRecord]:
    """Compute direct pairwise diagnostics between residual variants."""
    records: List[ResidualPairRecord] = []

    for variant_a, variant_b in pairs:
        if variant_a not in fields:
            raise KeyError(f"Unknown variant_a: {variant_a}")
        if variant_b not in fields:
            raise KeyError(f"Unknown variant_b: {variant_b}")

        field_a = fields[variant_a]
        field_b = fields[variant_b]
        if field_a.shape[0] != field_b.shape[0]:
            raise ValueError(
                f"batch mismatch for {variant_a}/{variant_b}: "
                f"{tuple(field_a.shape)} vs {tuple(field_b.shape)}"
            )

        cosine = field_metrics.cosine_to(field_a, field_b)
        norm_a = field_metrics.field_norm(field_a)
        norm_b = field_metrics.field_norm(field_b)
        diff = field_b - field_a
        diff_norm = field_metrics.field_norm(diff)
        diff_freq = field_metrics.frequency_energy_ratio(diff, height=height, width=width)
        ratio = norm_b / (norm_a + 1e-12)

        for i in range(field_a.shape[0]):
            records.append(
                ResidualPairRecord(
                    style_id=style_id,
                    prompt_id=prompt_id,
                    timestep=float(timestep),
                    variant_a=variant_a,
                    variant_b=variant_b,
                    cosine=float(cosine[i].detach().cpu().item()),
                    diff_norm=float(diff_norm[i].detach().cpu().item()),
                    norm_ratio_b_over_a=float(ratio[i].detach().cpu().item()),
                    diff_hi_freq_ratio=float(diff_freq["high_freq_ratio"][i].detach().cpu().item()),
                    diff_low_freq_ratio=float(diff_freq["low_freq_ratio"][i].detach().cpu().item()),
                )
            )

    return records
