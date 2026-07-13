"""Reference-independent spatial transport for native score residuals."""

from __future__ import annotations

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def content_affinity_transport(
    score_field: Tensor,
    content_tokens: Tensor,
    *,
    temperature: float = 0.1,
    strength: float = 0.75,
    highpass_gain: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Transport a score field with a reference-independent content carrier.

    ``content_tokens`` are the same post-Step1 content latent used by the
    TPAMI inner loop. Their normalized token affinity defines a spatial
    transport matrix; the reference image never enters this operation. The
    optional high-pass term keeps local style variation after transport.

    Returns ``(transported_field, affinity)``. The affinity is detached so the
    teacher carrier cannot become an unintended second optimization objective.
    """

    if score_field.ndim != 3 or content_tokens.ndim != 3:
        raise ValueError("score_field and content_tokens must be [B, L, C]")
    if score_field.shape[:2] != content_tokens.shape[:2]:
        raise ValueError("score_field and content_tokens must share batch/token dimensions")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if float(strength) < 0.0:
        raise ValueError("strength must be non-negative")

    content = F.normalize(content_tokens.float(), dim=-1)
    logits = torch.bmm(content, content.transpose(1, 2)) / float(temperature)
    affinity = torch.softmax(logits, dim=-1).detach()
    transported = torch.bmm(affinity, score_field.float())
    highpass = score_field.float() - transported
    field = transported + float(highpass_gain) * highpass
    field = score_field.float() + float(strength) * (field - score_field.float())
    return field.to(dtype=score_field.dtype), affinity


def affinity_transport_cosine(before: Tensor, after: Tensor) -> Tensor:
    """Return per-batch cosine similarity between two transported fields."""

    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    before_flat = before.float().flatten(1)
    after_flat = after.float().flatten(1)
    return F.cosine_similarity(before_flat, after_flat, dim=-1)


__all__ = ["affinity_transport_cosine", "content_affinity_transport"]
