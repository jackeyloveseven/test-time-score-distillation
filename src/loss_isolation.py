"""Helpers for warm-start/final-AdaIN loss-isolation bookkeeping."""

from __future__ import annotations

import torch

from .latent_ops import adain_1d


Tensor = torch.Tensor


def final_adain_if_enabled(
    latents: Tensor,
    style_latent: Tensor,
    *,
    enabled: bool,
) -> Tensor:
    if not enabled:
        return latents
    return adain_1d(latents, style_latent)


def is_true_loss_only(
    *,
    disable_slt: bool,
    adain_alpha: float,
    disable_final_adain: bool,
) -> bool:
    return bool(disable_slt and float(adain_alpha) == 0.0 and disable_final_adain)


def build_loss_isolation_contract(
    *,
    disable_slt: bool,
    adain_alpha: float,
    disable_final_adain: bool,
) -> dict[str, object]:
    true_loss_only = is_true_loss_only(
        disable_slt=disable_slt,
        adain_alpha=adain_alpha,
        disable_final_adain=disable_final_adain,
    )
    return {
        "true_loss_only": true_loss_only,
        "disable_slt": bool(disable_slt),
        "adain_alpha": float(adain_alpha),
        "final_adain_enabled": not bool(disable_final_adain),
        "note": (
            "true loss-only: SLT disabled, warm-start AdaIN disabled, final decode-time AdaIN disabled; dist/content regularizers remain enabled"
            if true_loss_only
            else "not true loss-only: at least one of SLT, warm-start AdaIN, or final decode-time AdaIN remains enabled"
        ),
    }


__all__ = [
    "build_loss_isolation_contract",
    "final_adain_if_enabled",
    "is_true_loss_only",
]
