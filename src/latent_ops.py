"""Pure tensor ops for sinkhorn transport and AdaIN style operations.

All functions here are self-contained and do not depend on global pipeline/model
state.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def spatial_lowpass_tokens(
    field: Tensor,
    height: int,
    width: int,
    kernel_size: int = 5,
) -> Tensor:
    """Apply channelwise spatial average pooling to a ``[B, H*W, D]`` token field."""
    if field.ndim != 3:
        raise ValueError("field must be [B, L, D]")
    if height * width != field.shape[1]:
        raise ValueError(
            f"field token count L={field.shape[1]} does not match height*width={height * width}"
        )
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")

    k = int(kernel_size)
    if k % 2 == 0:
        k += 1

    b, _, d = field.shape
    x = field.float().reshape(b, height, width, d).permute(0, 3, 1, 2)
    low = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
    low = low.permute(0, 2, 3, 1).reshape(b, height * width, d)
    return low.to(dtype=field.dtype)


def spatial_highpass_tokens(
    field: Tensor,
    height: int,
    width: int,
    kernel_size: int = 5,
) -> Tensor:
    """Return high-frequency residual ``field - lowpass(field)`` for token fields."""
    return field - spatial_lowpass_tokens(field, height=height, width=width, kernel_size=kernel_size)


def highpass_corrected_teacher(
    hard_residual: Tensor,
    ot_residual: Tensor,
    height: int,
    width: int,
    lamb: float = 1.0,
    kernel_size: int = 5,
) -> Tensor:
    """Build ``hard + lambda * highpass(ot - hard)`` teacher residual."""
    if hard_residual.shape != ot_residual.shape:
        raise ValueError(
            f"hard_residual and ot_residual shapes must match, got "
            f"{tuple(hard_residual.shape)} and {tuple(ot_residual.shape)}"
        )
    correction = spatial_highpass_tokens(
        ot_residual - hard_residual,
        height=height,
        width=width,
        kernel_size=kernel_size,
    )
    return hard_residual + float(lamb) * correction


def _align_style_tokens(
    k_style: Tensor,
    v_style: Tensor,
) -> tuple[Tensor, Tensor]:
    """Align style K/V lengths for sinkhorn matmul.

    The content query length must be preserved in the output.
    """
    n = min(k_style.shape[-2], v_style.shape[-2])
    if n <= 0:
        return k_style.new_zeros(k_style.shape[:-2] + (0, k_style.shape[-1])), v_style.new_zeros(v_style.shape[:-2] + (0, v_style.shape[-1]))
    return k_style[:, :, :n, :], v_style[:, :, :n, :]


def sinkhorn_attn(
    Q: Tensor,
    K_style: Tensor,
    V_style: Tensor,
    n_iters: int = 6,
    tau: float = 0.1,
) -> Tensor:
    """Entropy-regularized attention transport with Sinkhorn normalization."""
    if Q.ndim != 4:
        raise ValueError("Q must be [B, H, Lq, D]")
    if K_style.ndim != 4 or V_style.ndim != 4:
        raise ValueError("K_style and V_style must be [B, H, L, D]")
    if Q.shape[:2] != K_style.shape[:2] or Q.shape[:2] != V_style.shape[:2]:
        raise ValueError("Q, K_style, V_style must share batch/head dims")

    k_t, v_t = _align_style_tokens(K_style, V_style)
    if k_t.shape[-2] == 0:
        return torch.zeros_like(Q)

    logits = torch.matmul(Q, k_t.transpose(-1, -2)) / (Q.shape[-1] ** 0.5 * tau)
    log_p = logits
    for _ in range(int(n_iters)):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    p = torch.softmax(log_p, dim=-1)
    return torch.matmul(p, v_t)


def sinkhorn_attn_freq(
    Q: Tensor,
    K_style: Tensor,
    V_style: Tensor,
    n_iters: int = 6,
    tau: float = 0.1,
    hi_boost: float = 2.0,
) -> Tensor:
    """Frequency-aware Sinkhorn attention transport."""
    if Q.ndim != 4:
        raise ValueError("Q must be [B, H, Lq, D]")
    if K_style.ndim != 4 or V_style.ndim != 4:
        raise ValueError("K_style and V_style must be [B, H, L, D]")
    if Q.shape[:2] != K_style.shape[:2] or Q.shape[:2] != V_style.shape[:2]:
        raise ValueError("Q, K_style, V_style must share batch/head dims")

    k_t, v_t = _align_style_tokens(K_style, V_style)
    if k_t.shape[-2] == 0:
        return torch.zeros_like(Q)

    q_lo = Q.mean(dim=2, keepdim=True).expand_as(Q)
    k_lo = k_t.mean(dim=2, keepdim=True).expand_as(k_t)
    q_hi = Q - q_lo
    k_hi = k_t - k_lo

    d = Q.shape[-1]
    logits_hi = torch.matmul(q_hi, k_hi.transpose(-1, -2)) / (d ** 0.5 * tau)
    logits_lo = torch.matmul(q_lo, k_lo.transpose(-1, -2)) / (d ** 0.5 * tau)
    logits = logits_lo + hi_boost * logits_hi

    log_p = logits
    for _ in range(int(n_iters)):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    p = torch.softmax(log_p, dim=-1)
    return torch.matmul(p, v_t)


def sinkhorn_style_pattern(
    K_style: Tensor,
    V_content: Tensor,
    n_iters: int = 6,
    tau: float = 0.1,
    txt_len: int = 0,
) -> Tensor:
    """Transfer style self-attention pattern to content values."""
    k_img = K_style[:, :, txt_len:, :]
    if k_img.shape[2] == 0:
        return V_content

    n = min(k_img.shape[-2], V_content.shape[-2])
    if n == 0:
        return V_content

    k_t = k_img[:, :, :n, :]
    v_t = V_content[:, :, :n, :]
    logits = torch.matmul(k_t, k_t.transpose(-1, -2)) / (K_style.shape[-1] ** 0.5 * tau)
    log_p = logits
    for _ in range(int(n_iters)):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    p = torch.softmax(log_p, dim=-1)
    out = torch.matmul(p, v_t)

    if out.shape[2] < V_content.shape[2]:
        out = F.pad(out, (0, 0, 0, V_content.shape[2] - out.shape[2]))
    return out[:, :, : V_content.shape[2], :]


def feature_adain(x: Tensor, y: Tensor, alpha: float = 1.0, eps: float = 1e-6) -> Tensor:
    """Per-token AdaIN-like style matching and interpolation."""
    mean_x = x.mean(dim=2, keepdim=True)
    std_x = x.std(dim=2, keepdim=True) + eps
    mean_y = y.mean(dim=2, keepdim=True)
    std_y = y.std(dim=2, keepdim=True) + eps
    res = (x - mean_x) / std_x * std_y + mean_y
    return alpha * res + (1.0 - alpha) * x


def adain_1d(source: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Global 1D AdaIN for token latents."""
    src_mean = source.mean(dim=1, keepdim=True)
    src_std = source.std(dim=1, keepdim=True) + eps
    tgt_mean = target.mean(dim=(0, 1), keepdim=True)
    tgt_std = target.std(dim=(0, 1), keepdim=True) + eps
    return (source - src_mean) / src_std * tgt_std + tgt_mean


def sinkhorn_latent_transport(
    content_lat: Tensor,
    style_lat_noisy: Tensor,
    style_lat_clean: Tensor,
    alpha: float = 0.5,
    n_iters: int = 10,
    tau: float = 0.1,
    grag_delta: float = 1.0,
    mode: Literal['ot_adain', 'blend'] = 'ot_adain',
) -> Tensor:
    """OT transport in latent token space."""
    if content_lat.ndim != 3:
        raise ValueError("content_lat should be [B, N, D]")

    style_noisy_g = style_lat_noisy
    style_clean_g = style_lat_clean
    if grag_delta < 1.0:
        s_mean = style_lat_noisy.mean(dim=1, keepdim=True)
        style_noisy_g = s_mean + grag_delta * (style_lat_noisy - s_mean)
        s_clean_mean = style_lat_clean.mean(dim=1, keepdim=True)
        style_clean_g = s_clean_mean + grag_delta * (style_lat_clean - s_clean_mean)

    c_hi = content_lat.float() - content_lat.mean(dim=1, keepdim=True)
    s_hi = style_noisy_g.float() - style_noisy_g.mean(dim=1, keepdim=True)
    c_norm = F.normalize(c_hi, dim=-1)
    s_norm = F.normalize(s_hi, dim=-1)

    logits = torch.bmm(c_norm, s_norm.transpose(1, 2)) / tau
    log_p = logits
    for _ in range(int(n_iters)):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    transport = torch.softmax(log_p, dim=-1)

    if mode == "blend":
        transported = torch.bmm(transport, style_clean_g.float()).to(content_lat.dtype)
        return (1.0 - alpha) * content_lat + alpha * transported

    style_f = style_clean_g.float()
    matched_mean = torch.bmm(transport, style_f)
    matched_var = torch.bmm(transport, style_f ** 2) - matched_mean**2
    matched_std = (matched_var.clamp(min=1e-6)).sqrt()

    c_mean = content_lat.float().mean(dim=1, keepdim=True)
    c_std = content_lat.float().std(dim=1, keepdim=True).clamp(min=1e-6)
    normed = (content_lat.float() - c_mean) / c_std
    styled = normed * matched_std + matched_mean
    return ((1.0 - alpha) * content_lat.float() + alpha * styled).to(content_lat.dtype)
