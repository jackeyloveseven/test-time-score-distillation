"""Tensor field diagnostics used by residual profiling.

Utilities here are intentionally pure and do not depend on model/runtime objects.
"""

from __future__ import annotations

from math import isqrt
from typing import Dict, Optional

import torch

Tensor = torch.Tensor


def flatten_field(x: Tensor) -> Tensor:
    """Flatten spatial/token/token-embedding field for batchwise reduction.

    Returns a tensor of shape ``[B, -1]``.
    """
    if x.ndim < 2:
        raise ValueError("field must have at least 2 dims with batch dimension")
    return x.reshape(x.shape[0], -1)


def cosine_to(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Batchwise cosine similarity for flattened fields.

    Returns tensor of shape ``[B]``.
    """
    a_f = flatten_field(a)
    b_f = flatten_field(b)

    if a_f.shape != b_f.shape:
        raise ValueError(
            f"a and b must have broadcast-compatible shapes, got {tuple(a_f.shape)} and {tuple(b_f.shape)}"
        )

    numerator = (a_f * b_f).sum(dim=1)
    denom = torch.linalg.norm(a_f, dim=1) * torch.linalg.norm(b_f, dim=1)
    return numerator / (denom + eps)


def field_norm(x: Tensor) -> Tensor:
    """Batchwise L2 norm of a field.

    Returns tensor of shape ``[B]``.
    """
    x_f = flatten_field(x)
    return torch.linalg.norm(x_f, dim=1)


def _to_btld(field: Tensor) -> Tensor:
    """Normalize tensor to ``[B, L, D]`` for metric helpers."""
    if field.ndim == 3:
        # [B, L, D]
        return field
    if field.ndim == 4:
        # [B, H, L, D]
        return field.mean(dim=1)
    raise ValueError("field must be [B, L, D] or [B, H, L, D]")


def _pairwise_cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Batchwise cosine for token-aligned neighbor field arrays."""
    num = (a * b).sum(dim=-1)
    den = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)
    return num / (den + eps)


def spatial_neighbor_cosine(field: Tensor, height: Optional[int] = None, width: Optional[int] = None) -> Tensor:
    """Batchwise average cosine of spatial neighbors.

    Supports ``[B, L, D]`` and ``[B, H, L, D]``.

    If ``height``/``width`` are omitted, the function requires ``L`` be a perfect
    square and infers 2D shape. Returns ``[B]``.
    """
    btld = _to_btld(field)
    b, lq, d = btld.shape

    if height is None and width is None:
        h = isqrt(lq)
        if h * h != lq:
            raise ValueError("height/width not provided and L is not a perfect square")
        height = h
        width = h

    if height is None or width is None:
        raise ValueError("both height and width must be provided together")

    if height * width != lq:
        raise ValueError(
            f"field token count L={lq} does not match height*width={height * width}"
        )

    x = btld.reshape(b, height, width, d)

    if height <= 1 and width <= 1:
        return torch.zeros(b, dtype=field.dtype, device=field.device)

    # Right and down neighbors only; this already covers all unique adjacency pairs.
    right = _pairwise_cosine(x[:, :, :-1, :], x[:, :, 1:, :])
    down = _pairwise_cosine(x[:, :-1, :, :], x[:, 1:, :, :])

    vals = [right, down]
    return sum(v.mean(dim=(-1, -2)) for v in vals) / len(vals)


def frequency_energy_ratio(
    field: Tensor,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Compute low/high frequency energy ratio with a simple square window.

    Supports ``[B, L, D]`` and ``[B, H, L, D]``. Returns:

    ``{"low_freq_ratio": ..., "high_freq_ratio": ...}`` with batchwise tensors.
    """
    btld = _to_btld(field)
    b, lq, d = btld.shape

    if height is None and width is None:
        h = isqrt(lq)
        if h * h != lq:
            raise ValueError("height/width not provided and L is not a perfect square")
        height = h
        width = h

    if height is None or width is None:
        raise ValueError("both height and width must be provided together")
    if height * width != lq:
        raise ValueError(
            f"field token count L={lq} does not match height*width={height * width}"
        )

    x = btld.reshape(b, height, width, d).to(dtype=torch.float32)

    # Sum over channel dimension first: [B, H, W]
    x_sum = x.mean(dim=-1)
    spec = torch.fft.rfft2(x_sum, dim=(1, 2))
    energy = (spec.abs() ** 2)

    total_energy = energy.sum(dim=(1, 2))

    low_h = max(1, height // 4)
    low_w = max(1, width // 4)
    low_mask = torch.zeros_like(energy)
    low_mask[:, :low_h, :low_w] = 1.0

    low_energy = (energy * low_mask).sum(dim=(1, 2))
    high_energy = total_energy - low_energy

    low_ratio = low_energy / (total_energy + 1e-12)
    high_ratio = high_energy / (total_energy + 1e-12)

    return {
        "low_freq_ratio": low_ratio,
        "high_freq_ratio": high_ratio,
    }
