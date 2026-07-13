"""FLUX.2 flow/velocity <-> score conversion helpers.

The scheduler contract used here is the one implemented by DiffSynth's
``FlowMatchScheduler``:

``z_sigma = (1 - sigma) * x0 + sigma * eps``
``v = eps - x0``

Therefore:

``x0_hat = z_sigma - sigma * v``
``eps_hat = z_sigma + (1 - sigma) * v``
``score = -(z_sigma + (1 - sigma) * v) / sigma``

For same-state velocity differences, the shared ``-z_sigma / sigma`` term must
be analytically removed before any detach boundary to avoid injecting an
identity-gradient through the optimized latent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class Flux2SigmaInfo:
    """Resolved FLUX.2 sigma/coefficient for a concrete scheduler timestep."""

    timestep_index: int
    sigma: Tensor
    sigma_safe: Tensor
    score_delta_coefficient: Tensor


def _to_tensor(value: Tensor | float, *, ref: Tensor) -> Tensor:
    if torch.is_tensor(value):
        out = value.to(device=ref.device, dtype=ref.dtype)
    else:
        out = torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
    return out


def _broadcast_like(value: Tensor, ref: Tensor) -> Tensor:
    while value.ndim < ref.ndim:
        value = value.unsqueeze(-1)
    return value


def resolve_sigma_info(
    scheduler: Any,
    timestep: Tensor | float,
    *,
    ref_tensor: Tensor,
    sigma_eps: float = 1e-4,
) -> Flux2SigmaInfo:
    """Resolve sigma by nearest scheduler timestep, never by ``t / 1000``."""

    scheduler_timesteps = scheduler.timesteps
    scheduler_sigmas = scheduler.sigmas
    timestep_tensor = _to_tensor(timestep, ref=ref_tensor).reshape(-1)[0]
    timestep_cpu = timestep_tensor.detach().to(device="cpu", dtype=scheduler_timesteps.dtype)
    timestep_index = int(torch.argmin((scheduler_timesteps - timestep_cpu).abs()).item())
    sigma = scheduler_sigmas[timestep_index].to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    sigma_safe = sigma.clamp_min(float(sigma_eps))
    coeff = -((1.0 - sigma) / sigma_safe)
    return Flux2SigmaInfo(
        timestep_index=timestep_index,
        sigma=sigma,
        sigma_safe=sigma_safe,
        score_delta_coefficient=coeff,
    )


def velocity_to_x0(z_sigma: Tensor, velocity: Tensor, sigma: Tensor | float) -> Tensor:
    sigma_t = _broadcast_like(_to_tensor(sigma, ref=z_sigma), z_sigma)
    return z_sigma - sigma_t * velocity


def velocity_to_eps(z_sigma: Tensor, velocity: Tensor, sigma: Tensor | float) -> Tensor:
    sigma_t = _broadcast_like(_to_tensor(sigma, ref=z_sigma), z_sigma)
    return z_sigma + (1.0 - sigma_t) * velocity


def velocity_to_score(
    z_sigma: Tensor,
    velocity: Tensor,
    sigma: Tensor | float,
    *,
    sigma_eps: float = 1e-4,
) -> Tensor:
    sigma_t = _broadcast_like(_to_tensor(sigma, ref=z_sigma), z_sigma)
    sigma_safe = sigma_t.clamp_min(float(sigma_eps))
    eps_hat = velocity_to_eps(z_sigma, velocity, sigma_t)
    return -(eps_hat / sigma_safe)


def velocity_delta_to_score_delta(
    delta_velocity: Tensor,
    sigma: Tensor | float,
    *,
    sigma_eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Convert a same-state velocity difference to a score difference.

    ``delta_velocity`` must already have removed any shared state-only terms.
    """

    sigma_ref = _to_tensor(sigma, ref=delta_velocity)
    sigma_safe = sigma_ref.clamp_min(float(sigma_eps))
    coeff = -((1.0 - sigma_ref) / sigma_safe)
    coeff_full = _broadcast_like(coeff, delta_velocity)
    return coeff_full * delta_velocity, coeff


def normalize_score_delta(
    delta_score: Tensor,
    coefficient: Tensor | float,
    *,
    coeff_eps: float = 1e-6,
) -> Tensor:
    """Remove only the timestep-dependent scalar coefficient magnitude.

    This preserves direction and relative token/channel energy while avoiding
    endpoint blow-ups when ``sigma -> 0``.
    """

    coeff = _to_tensor(coefficient, ref=delta_score)
    denom = coeff.abs().clamp_min(float(coeff_eps))
    denom_full = _broadcast_like(denom, delta_score)
    normalized = delta_score / denom_full
    mask = _broadcast_like((coeff.abs() < float(coeff_eps)), delta_score)
    return torch.where(mask, torch.zeros_like(normalized), normalized)


def zero_reference_loss(
    delta: Tensor,
    *,
    loss_kind: str = "mse",
) -> Tensor:
    zeros = torch.zeros_like(delta)
    if loss_kind == "mse":
        return F.mse_loss(delta.float(), zeros.float(), reduction="mean")
    if loss_kind == "smooth_l1":
        return F.smooth_l1_loss(delta.float(), zeros.float(), reduction="mean")
    raise ValueError(f"Unsupported loss_kind: {loss_kind}")


def l2_normalize_field(field: Tensor, *, eps: float = 1e-6) -> Tensor:
    flat = field.float().reshape(field.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(float(eps))
    norm_full = norms.reshape(field.shape[0], *([1] * (field.ndim - 1)))
    return (field.float() / norm_full).to(dtype=field.dtype)


def rms_normalize_field(field: Tensor, *, eps: float = 1e-6) -> Tensor:
    flat = field.float().reshape(field.shape[0], -1)
    rms = flat.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(float(eps))
    rms_full = rms.reshape(field.shape[0], *([1] * (field.ndim - 1)))
    return (field.float() / rms_full).to(dtype=field.dtype)


def build_pseudo_target_loss(
    x0_pred: Tensor,
    score_pseudo_grad: Tensor,
    *,
    step_size: float,
    normalize_grad_mode: str = "rms",
    grad_eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create the detached pseudo-target used by Plan B.

    Returns ``(loss, x0_target, pseudo_direction)``.
    """

    direction = score_pseudo_grad.detach()
    if normalize_grad_mode == "none":
        pass
    elif normalize_grad_mode == "rms":
        direction = rms_normalize_field(direction, eps=grad_eps)
    elif normalize_grad_mode == "l2":
        direction = l2_normalize_field(direction, eps=grad_eps)
    else:
        raise ValueError(f"Unsupported normalize_grad_mode: {normalize_grad_mode}")
    x0_target = (x0_pred.detach() + float(step_size) * direction).detach()
    loss = 0.5 * F.mse_loss(x0_pred.float(), x0_target.float(), reduction="mean")
    return loss, x0_target, direction


__all__ = [
    "Flux2SigmaInfo",
    "build_pseudo_target_loss",
    "l2_normalize_field",
    "normalize_score_delta",
    "rms_normalize_field",
    "resolve_sigma_info",
    "velocity_delta_to_score_delta",
    "velocity_to_eps",
    "velocity_to_score",
    "velocity_to_x0",
    "zero_reference_loss",
]
