"""Shared Klein9B model-loading configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class DualModelVramContract:
    requested_main_vram_limit_gb: float | None
    requested_teacher_vram_limit_gb: float | None
    effective_main_vram_limit_gb: float | None
    effective_teacher_vram_limit_gb: float | None
    main_requires_full_cuda: bool
    teacher_requires_full_cuda: bool
    teacher_model_required: bool
    notes: tuple[str, ...]


def resolve_dual_model_vram_contract(
    style_mode: str,
    *,
    legacy_vram_limit_gb: float | None = None,
    main_vram_limit_gb: float | None = None,
    teacher_vram_limit_gb: float | None = None,
) -> DualModelVramContract:
    requested_main = main_vram_limit_gb if main_vram_limit_gb is not None else legacy_vram_limit_gb
    requested_teacher = teacher_vram_limit_gb if teacher_vram_limit_gb is not None else legacy_vram_limit_gb
    effective_main = requested_main
    effective_teacher = requested_teacher
    notes: list[str] = []
    main_requires_full_cuda = style_mode == "kv_match"
    teacher_requires_full_cuda = style_mode == "score_match"
    teacher_model_required = style_mode != "kv_match"
    if main_requires_full_cuda and effective_main is not None:
        notes.append(
            "kv_match requires the distilled main DiT to stay full CUDA for latent backward; "
            f"overriding main_vram_limit_gb={effective_main} to None"
        )
        effective_main = None
    if teacher_requires_full_cuda and effective_teacher is not None:
        notes.append(
            "score_match requires a full-CUDA teacher DiT for latent backward; "
            f"overriding teacher_vram_limit_gb={effective_teacher} to None"
        )
        effective_teacher = None
    if not teacher_model_required and requested_teacher is not None:
        notes.append(
            "kv_match does not load an external teacher model; teacher_vram_limit_gb is ignored"
        )
    return DualModelVramContract(
        requested_main_vram_limit_gb=requested_main,
        requested_teacher_vram_limit_gb=requested_teacher,
        effective_main_vram_limit_gb=effective_main,
        effective_teacher_vram_limit_gb=effective_teacher,
        main_requires_full_cuda=main_requires_full_cuda,
        teacher_requires_full_cuda=teacher_requires_full_cuda,
        teacher_model_required=teacher_model_required,
        notes=tuple(notes),
    )


def model_config_offload_kwargs(vram_limit_gb: float | None) -> Dict[str, object]:
    """Enable lossless BF16 CPU offload when a VRAM limit is requested.

    DiffSynth only activates its layer-wise VRAM manager when a ModelConfig has
    both offload dtype and device. Passing ``vram_limit`` to the pipeline alone
    does not enable offload and still loads the full text encoder and DiT onto
    CUDA at the same time.
    """

    if vram_limit_gb is None:
        return {}
    return {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def model_loading_mode(vram_limit_gb: float | None) -> str:
    return "full_cuda" if vram_limit_gb is None else "cpu_offload_bf16"


__all__ = [
    "DualModelVramContract",
    "model_config_offload_kwargs",
    "model_loading_mode",
    "resolve_dual_model_vram_contract",
]
