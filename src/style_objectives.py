"""Plan A / Plan B style objectives for the TPAMI-style latent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .flux2_score import (
    build_pseudo_target_loss,
    normalize_score_delta,
    velocity_delta_to_score_delta,
    zero_reference_loss,
)


Tensor = torch.Tensor
NormalizationMode = Literal["raw", "coeff"]
LossKind = Literal["mse", "smooth_l1"]
StyleMode = Literal["score_match", "ca_only", "ca_dmd", "kv_match"]


@dataclass(frozen=True)
class ScoreMatchLossResult:
    loss: Tensor
    delta_velocity: Tensor
    delta_score: Tensor
    delta_score_for_loss: Tensor
    score_delta_coefficient: Tensor


@dataclass(frozen=True)
class PseudoGuidanceResult:
    loss: Tensor
    delta_ca_velocity: Tensor
    delta_ca_score: Tensor
    delta_dm_velocity: Tensor | None
    delta_dm_score: Tensor | None
    pseudo_grad_score: Tensor
    pseudo_direction: Tensor
    x0_target: Tensor
    score_delta_coefficient: Tensor


def validate_style_mode(mode: str, backend_kind: str | None) -> None:
    valid_modes = {"score_match", "ca_only", "ca_dmd", "kv_match"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported style mode {mode!r}; expected one of {sorted(valid_modes)}")
    if mode == "ca_dmd" and backend_kind in {None, "", "none"}:
        raise ValueError(
            "style_mode='ca_dmd' requires a real fake-score backend. "
            "Use 'ca_only' until a DMD2-compatible guidance backend is wired."
        )


def score_match_loss_from_velocities(
    v_pre: Tensor,
    v_style_detached: Tensor,
    sigma: Tensor | float,
    *,
    normalization: NormalizationMode = "coeff",
    loss_kind: LossKind = "mse",
    sigma_eps: float = 1e-4,
    coeff_eps: float = 1e-6,
) -> ScoreMatchLossResult:
    """Plan A objective with the shared state term analytically removed.

    The detach boundary is applied before score conversion:

    ``delta_v = v_pre - stopgrad(v_style)``
    ``delta_s = -(1 - sigma) / sigma * delta_v``
    """

    delta_velocity = v_pre - v_style_detached.detach()
    delta_score, coeff = velocity_delta_to_score_delta(
        delta_velocity,
        sigma,
        sigma_eps=sigma_eps,
    )
    if normalization == "raw":
        delta_for_loss = delta_score
    elif normalization == "coeff":
        delta_for_loss = normalize_score_delta(
            delta_score,
            coeff,
            coeff_eps=coeff_eps,
        )
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization}")
    loss = zero_reference_loss(delta_for_loss, loss_kind=loss_kind)
    return ScoreMatchLossResult(
        loss=loss,
        delta_velocity=delta_velocity,
        delta_score=delta_score,
        delta_score_for_loss=delta_for_loss,
        score_delta_coefficient=coeff,
    )


def pseudo_guidance_from_velocities(
    mode: StyleMode,
    *,
    x0_pred: Tensor,
    v_with_detached: Tensor,
    v_without_detached: Tensor,
    sigma: Tensor | float,
    cfg_scale: float = 1.0,
    lambda_ca: float = 1.0,
    lambda_dm: float = 1.0,
    pseudo_step_size: float = 0.05,
    v_fake_detached: Tensor | None = None,
    sigma_eps: float = 1e-4,
    normalize_grad_mode: str = "rms",
    grad_eps: float = 1e-6,
) -> PseudoGuidanceResult:
    """Plan B pseudo-gradient objective in score space."""

    if mode not in {"ca_only", "ca_dmd"}:
        raise ValueError(f"Pseudo guidance is only defined for Plan B modes, got {mode!r}")

    delta_ca_velocity = v_with_detached.detach() - v_without_detached.detach()
    delta_ca_score, coeff = velocity_delta_to_score_delta(
        delta_ca_velocity,
        sigma,
        sigma_eps=sigma_eps,
    )

    delta_dm_velocity = None
    delta_dm_score = None
    pseudo_grad = float(lambda_ca) * (float(cfg_scale) - 1.0) * delta_ca_score

    if mode == "ca_dmd":
        if v_fake_detached is None:
            raise ValueError(
                "style_mode='ca_dmd' requires v_fake_detached from a real online fake-score backend. "
                "Use 'ca_only' if no backend is available."
            )
        delta_dm_velocity = v_with_detached.detach() - v_fake_detached.detach()
        delta_dm_score, _ = velocity_delta_to_score_delta(
            delta_dm_velocity,
            sigma,
            sigma_eps=sigma_eps,
        )
        pseudo_grad = pseudo_grad + float(lambda_dm) * delta_dm_score

    loss, x0_target, pseudo_direction = build_pseudo_target_loss(
        x0_pred,
        pseudo_grad,
        step_size=pseudo_step_size,
        normalize_grad_mode=normalize_grad_mode,
        grad_eps=grad_eps,
    )
    return PseudoGuidanceResult(
        loss=loss,
        delta_ca_velocity=delta_ca_velocity,
        delta_ca_score=delta_ca_score,
        delta_dm_velocity=delta_dm_velocity,
        delta_dm_score=delta_dm_score,
        pseudo_grad_score=pseudo_grad,
        pseudo_direction=pseudo_direction,
        x0_target=x0_target,
        score_delta_coefficient=coeff,
    )


__all__ = [
    "PseudoGuidanceResult",
    "ScoreMatchLossResult",
    "pseudo_guidance_from_velocities",
    "score_match_loss_from_velocities",
    "validate_style_mode",
]
