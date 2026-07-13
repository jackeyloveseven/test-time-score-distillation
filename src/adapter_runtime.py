"""Runtime helpers for applying trained residual adapters."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Dict, Sequence

import torch

from .student_adapter import LowRankResidualAdapter, ResidualAdapter, ResidualAdapterConfig


Tensor = torch.Tensor


def _make_config(raw_config: dict) -> ResidualAdapterConfig:
    valid = {field.name for field in fields(ResidualAdapterConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid}
    return ResidualAdapterConfig(**filtered)


def _infer_adapter_type(state: dict) -> str:
    if any(key.startswith("coeff_mlp.") or key == "basis" for key in state):
        return "lowrank"
    return "mlp"


def _as_batch_ids(value: str | Sequence[str], batch_size: int) -> list[str]:
    if isinstance(value, str):
        return [value] * batch_size
    out = [str(item) for item in value]
    if len(out) != batch_size:
        raise ValueError(f"Expected {batch_size} ids, got {len(out)}")
    return out


class ResidualAdapterRuntime:
    """Load and apply a residual adapter checkpoint with id mappings."""

    def __init__(
        self,
        model: torch.nn.Module,
        style_to_idx: Dict[str, int],
        prompt_to_idx: Dict[str, int],
        device: torch.device,
        student_scale: float = 0.75,
        adapter_type: str = "mlp",
    ) -> None:
        self.model = model
        self.style_to_idx = dict(style_to_idx)
        self.prompt_to_idx = dict(prompt_to_idx)
        self.device = device
        self.student_scale = float(student_scale)
        self.adapter_type = adapter_type

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        student_scale: float = 0.75,
    ) -> "ResidualAdapterRuntime":
        dev = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = _make_config(ckpt["config"])
        state = ckpt["model_state"]
        adapter_type = _infer_adapter_type(state)
        model = LowRankResidualAdapter(config) if adapter_type == "lowrank" else ResidualAdapter(config)
        model.load_state_dict(state)
        model.to(dev)
        model.eval()
        return cls(
            model=model,
            style_to_idx=ckpt["style_to_idx"],
            prompt_to_idx=ckpt["prompt_to_idx"],
            device=dev,
            student_scale=student_scale,
            adapter_type=adapter_type,
        )

    def _ids_to_tensors(
        self,
        style_id: str | Sequence[str],
        prompt_id: str | Sequence[str],
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        style_ids = _as_batch_ids(style_id, batch_size)
        prompt_ids = _as_batch_ids(prompt_id, batch_size)
        try:
            style_idx = [self.style_to_idx[item] for item in style_ids]
        except KeyError as exc:
            raise KeyError(f"Unknown style id {exc.args[0]!r}; known={sorted(self.style_to_idx)}") from exc
        try:
            prompt_idx = [self.prompt_to_idx[item] for item in prompt_ids]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt id {exc.args[0]!r}; known={sorted(self.prompt_to_idx)}") from exc
        return (
            torch.tensor(style_idx, dtype=torch.long, device=self.device),
            torch.tensor(prompt_idx, dtype=torch.long, device=self.device),
        )

    def predict_delta(
        self,
        hard_residual: Tensor,
        timestep: float | Tensor,
        style_id: str | Sequence[str],
        prompt_id: str | Sequence[str],
    ) -> Tensor:
        squeeze = False
        if hard_residual.ndim == 2:
            hard_residual = hard_residual[None, :, :]
            squeeze = True
        if hard_residual.ndim != 3:
            raise ValueError("hard_residual must be [L, D] or [B, L, D]")
        hard = hard_residual.to(self.device)
        batch_size = hard.shape[0]
        if isinstance(timestep, Tensor):
            time = timestep.to(self.device).float().reshape(-1)
        else:
            time = torch.full((batch_size,), float(timestep), dtype=torch.float32, device=self.device)
        if time.numel() == 1 and batch_size > 1:
            time = time.expand(batch_size)
        if time.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} timesteps, got {time.numel()}")
        style_idx, prompt_idx = self._ids_to_tensors(style_id, prompt_id, batch_size)
        with torch.no_grad():
            delta = self.model(hard, time, style_idx, prompt_idx)
        return delta[0] if squeeze else delta

    def apply(
        self,
        hard_residual: Tensor,
        timestep: float | Tensor,
        style_id: str | Sequence[str],
        prompt_id: str | Sequence[str],
        student_scale: float | None = None,
    ) -> Tensor:
        scale = self.student_scale if student_scale is None else float(student_scale)
        delta = self.predict_delta(hard_residual, timestep, style_id, prompt_id)
        return hard_residual.to(delta.device) + scale * delta


__all__ = ["ResidualAdapterRuntime"]
