"""Gradient-scale audit helpers for TPAMI-style latent optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch


Tensor = torch.Tensor
SupportedAuditMode = Literal["score_match", "ca_only", "ca_dmd", "kv_match", "carrier_score", "score_carrier"]


@dataclass
class StyleGradientCalibrationState:
    """Frozen run-level state for optional first-step style gradient matching."""

    target_rms: float | None
    calibration_eps: float
    scale: float = 1.0
    effective_style_weight: float | None = None
    frozen: bool = False
    measured_once: bool = False
    measured_outer_step: int | None = None
    measured_inner_iter: int | None = None
    measured_style_mode: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.calibration_eps)) or float(self.calibration_eps) <= 0.0:
            raise ValueError(f"calibration_eps must be finite and > 0, got {self.calibration_eps!r}")
        if self.target_rms is not None:
            if not math.isfinite(float(self.target_rms)) or float(self.target_rms) <= 0.0:
                raise ValueError(f"target_rms must be finite and > 0, got {self.target_rms!r}")


@dataclass(frozen=True)
class GradientAuditResult:
    weighted_style_loss: Tensor
    weighted_dist_loss: Tensor
    weighted_content_loss: Tensor
    scale_applied: float
    effective_style_weight: float
    log_row: dict[str, object] = field(default_factory=dict)


def grad_rms(grad: Tensor | None) -> float:
    if grad is None:
        return 0.0
    grad_f = grad.float()
    if grad_f.numel() == 0:
        return 0.0
    return float(grad_f.square().mean().sqrt().item())


def grad_cosine(a: Tensor | None, b: Tensor | None, eps: float = 1e-12) -> float | None:
    if a is None or b is None:
        return None
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    denom = float(a_f.norm().item() * b_f.norm().item())
    if denom <= eps:
        return None
    return float(torch.dot(a_f, b_f).item() / denom)


def _component_grad(loss: Tensor, latents: Tensor) -> Tensor | None:
    grad = torch.autograd.grad(
        loss,
        latents,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )[0]
    if grad is None:
        return None
    return grad.detach()


def compute_style_calibration_scale(
    *,
    raw_style_grad_rms: float,
    style_weight: float,
    target_rms: float,
    calibration_eps: float,
) -> float:
    if not math.isfinite(float(target_rms)) or float(target_rms) <= 0.0:
        raise ValueError(f"target_rms must be finite and > 0, got {target_rms!r}")
    if not math.isfinite(float(calibration_eps)) or float(calibration_eps) <= 0.0:
        raise ValueError(f"calibration_eps must be finite and > 0, got {calibration_eps!r}")
    denom = abs(float(style_weight)) * float(raw_style_grad_rms)
    if denom <= float(calibration_eps):
        raise RuntimeError(
            "Cannot calibrate style gradient RMS because the first-step weighted style gradient "
            f"is too small: |style_weight|*raw_style_grad_rms={denom:.12g} <= eps={float(calibration_eps):.12g}"
        )
    return float(target_rms) / denom


def apply_gradient_audit(
    *,
    style_mode: SupportedAuditMode,
    latents: Tensor,
    style_loss: Tensor,
    dist_loss: Tensor,
    content_loss: Tensor,
    measurement_latents: Tensor | None = None,
    measurement_style_loss: Tensor | None = None,
    measurement_dist_loss: Tensor | None = None,
    measurement_content_loss: Tensor | None = None,
    style_weight: float,
    dist_weight: float,
    content_weight: float,
    calibration_state: StyleGradientCalibrationState,
    should_measure: bool,
    outer_step: int,
    inner_iter: int,
) -> GradientAuditResult:
    if style_mode not in {"score_match", "ca_only", "ca_dmd", "kv_match", "carrier_score", "score_carrier"}:
        raise ValueError(f"Unsupported style_mode for gradient audit: {style_mode}")

    scale_applied = float(calibration_state.scale)
    effective_style_weight = float(style_weight) * scale_applied
    log_row: dict[str, object] = {}

    if should_measure:
        calibration_state.measured_once = True
        calibration_state.measured_outer_step = int(outer_step)
        calibration_state.measured_inner_iter = int(inner_iter)
        calibration_state.measured_style_mode = str(style_mode)
        grad_latents = measurement_latents if measurement_latents is not None else latents
        grad_style_loss = measurement_style_loss if measurement_style_loss is not None else style_loss
        grad_dist_loss = measurement_dist_loss if measurement_dist_loss is not None else dist_loss
        grad_content_loss = measurement_content_loss if measurement_content_loss is not None else content_loss
        style_grad = _component_grad(grad_style_loss, grad_latents)
        dist_grad = _component_grad(grad_dist_loss, grad_latents)
        content_grad = _component_grad(grad_content_loss, grad_latents)

        raw_style_grad_rms = grad_rms(style_grad)
        raw_dist_grad_rms = grad_rms(dist_grad)
        raw_content_grad_rms = grad_rms(content_grad)
        weighted_style_grad_rms_pre = abs(float(style_weight)) * raw_style_grad_rms

        if calibration_state.target_rms is not None and not calibration_state.frozen:
            calibration_state.scale = compute_style_calibration_scale(
                raw_style_grad_rms=raw_style_grad_rms,
                style_weight=style_weight,
                target_rms=float(calibration_state.target_rms),
                calibration_eps=float(calibration_state.calibration_eps),
            )
            calibration_state.effective_style_weight = float(style_weight) * float(calibration_state.scale)
            calibration_state.frozen = True

        scale_applied = float(calibration_state.scale)
        effective_style_weight = float(style_weight) * scale_applied

        log_row = {
            "gradient_audit_measured": True,
            "gradient_audit_style_mode": style_mode,
            "style_grad_rms": raw_style_grad_rms,
            "style_weighted_grad_rms_pre_calibration": weighted_style_grad_rms_pre,
            "style_weighted_grad_rms_applied": abs(effective_style_weight) * raw_style_grad_rms,
            "dist_weighted_grad_rms": abs(float(dist_weight)) * raw_dist_grad_rms,
            "content_weighted_grad_rms": abs(float(content_weight)) * raw_content_grad_rms,
            "calibration_scale": scale_applied,
            "effective_style_weight": effective_style_weight,
            "style_dist_grad_cosine": grad_cosine(style_grad, dist_grad),
            "style_content_grad_cosine": grad_cosine(style_grad, content_grad),
        }

    return GradientAuditResult(
        weighted_style_loss=style_loss * effective_style_weight,
        weighted_dist_loss=dist_loss * float(dist_weight),
        weighted_content_loss=content_loss * float(content_weight),
        scale_applied=scale_applied,
        effective_style_weight=effective_style_weight,
        log_row=log_row,
    )


__all__ = [
    "GradientAuditResult",
    "StyleGradientCalibrationState",
    "apply_gradient_audit",
    "compute_style_calibration_scale",
    "grad_cosine",
    "grad_rms",
]
