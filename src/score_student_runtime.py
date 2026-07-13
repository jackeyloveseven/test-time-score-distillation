"""Runtime helpers for direct score/velocity residual students."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import nn

from .score_residual_student import ScoreResidualStudentConfig, build_score_residual_student


Tensor = torch.Tensor


def _make_config(raw_config: dict) -> ScoreResidualStudentConfig:
    valid = {field.name for field in fields(ScoreResidualStudentConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid}
    return ScoreResidualStudentConfig(**filtered)


def _as_batch_ids(value: str | Sequence[str], batch_size: int) -> list[str]:
    if isinstance(value, str):
        return [value] * batch_size
    out = [str(item) for item in value]
    if len(out) != batch_size:
        raise ValueError(f"Expected {batch_size} ids, got {len(out)}")
    return out


class ScoreResidualStudentRuntime:
    """Load and apply a direct z_t-conditioned residual correction student."""

    def __init__(
        self,
        model: nn.Module,
        style_to_idx: Dict[str, int],
        prompt_to_idx: Dict[str, int],
        device: torch.device,
        student_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.style_to_idx = dict(style_to_idx)
        self.prompt_to_idx = dict(prompt_to_idx)
        self.device = device
        self.student_scale = float(student_scale)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        student_scale: float = 1.0,
    ) -> "ScoreResidualStudentRuntime":
        dev = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = _make_config(ckpt["config"])
        model = build_score_residual_student(config)
        model.load_state_dict(ckpt["model"])
        model.to(dev)
        model.eval()
        return cls(
            model=model,
            style_to_idx=ckpt["style_to_idx"],
            prompt_to_idx=ckpt["prompt_to_idx"],
            device=dev,
            student_scale=student_scale,
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

    def predict_correction(
        self,
        z_t: Tensor,
        timestep: float | Tensor,
        style_id: str | Sequence[str],
        prompt_id: str | Sequence[str],
    ) -> Tensor:
        squeeze = False
        if z_t.ndim == 2:
            z_t = z_t[None, :, :]
            squeeze = True
        if z_t.ndim != 3:
            raise ValueError("z_t must be [L, D] or [B, L, D]")
        z = z_t.to(self.device)
        batch_size = z.shape[0]
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
            correction = self.model(z, time, style_idx, prompt_idx)
        return correction[0] if squeeze else correction

    def apply_hard_delta(
        self,
        z_t: Tensor,
        hard_residual: Tensor,
        timestep: float | Tensor,
        style_id: str | Sequence[str],
        prompt_id: str | Sequence[str],
        student_scale: float | None = None,
    ) -> Tensor:
        scale = self.student_scale if student_scale is None else float(student_scale)
        correction = self.predict_correction(z_t, timestep, style_id, prompt_id)
        return hard_residual.to(correction.device) + scale * correction


__all__ = ["ScoreResidualStudentRuntime"]
