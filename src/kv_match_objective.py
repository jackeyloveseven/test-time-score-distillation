"""Pure K/V-match style objective helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from .latent_ops import sinkhorn_attn, sinkhorn_attn_freq


Tensor = torch.Tensor
KVMode = Literal["standard", "freq"]


@dataclass(frozen=True)
class KVMatchLossResult:
    loss: Tensor
    tau: float
    layer_count: int
    ot_layer_count: int
    target_norm: Tensor
    self_out_norm: Tensor


def apply_reference_kv_grag(
    ref_k_list: Sequence[Tensor],
    ref_v_list: Sequence[Tensor],
    *,
    txt_len: int,
    shallow: float,
    deep: float,
) -> tuple[list[Tensor], list[Tensor]]:
    """Apply per-layer GRAG compression to the image-token part of reference K/V."""

    out_k = [tensor.detach().clone() for tensor in ref_k_list]
    out_v = [tensor.detach().clone() for tensor in ref_v_list]
    if not out_k:
        return out_k, out_v
    for idx, (ref_k, ref_v) in enumerate(zip(out_k, out_v)):
        depth_frac = idx / max(len(out_k) - 1, 1)
        delta = float(shallow) + (float(deep) - float(shallow)) * depth_frac
        if delta >= 1.0 or ref_k.shape[2] <= txt_len:
            continue
        k_txt, k_img = ref_k[:, :, :txt_len, :], ref_k[:, :, txt_len:, :]
        v_txt, v_img = ref_v[:, :, :txt_len, :], ref_v[:, :, txt_len:, :]
        k_mean = k_img.mean(dim=2, keepdim=True)
        v_mean = v_img.mean(dim=2, keepdim=True)
        out_k[idx] = torch.cat([k_txt, k_mean + delta * (k_img - k_mean)], dim=2)
        out_v[idx] = torch.cat([v_txt, v_mean + delta * (v_img - v_mean)], dim=2)
    return out_k, out_v


def kv_match_loss_from_attention_layers(
    *,
    self_out_list: Sequence[Tensor],
    q_list: Sequence[Tensor],
    ref_k_list: Sequence[Tensor],
    ref_v_list: Sequence[Tensor],
    sinkhorn_tau: float,
    sinkhorn_iters: int = 6,
    ot_layer_frac: float = 0.5,
    sinkhorn_mode: KVMode = "freq",
    freq_hi_boost: float = 2.0,
) -> KVMatchLossResult:
    """Reproduce the original summed-L1 K/V style objective on cached layers."""

    if not self_out_list or not q_list:
        raise ValueError("self_out_list and q_list must be non-empty")
    if not (len(self_out_list) == len(q_list) == len(ref_k_list) == len(ref_v_list)):
        raise ValueError("self_out/q/ref_k/ref_v layer counts must match")

    n_layers = len(q_list)
    n_ot_layers = max(1, int(n_layers * float(ot_layer_frac)))
    ref_self_out_list: list[Tensor] = []

    for idx, (q, ref_k, ref_v) in enumerate(zip(q_list, ref_k_list, ref_v_list)):
        q_det = q.detach()
        use_ot = idx >= n_layers - n_ot_layers
        if use_ot:
            if sinkhorn_mode == "freq":
                ref_self_out = sinkhorn_attn_freq(
                    q_det,
                    ref_k,
                    ref_v,
                    n_iters=sinkhorn_iters,
                    tau=sinkhorn_tau,
                    hi_boost=freq_hi_boost,
                )
            elif sinkhorn_mode == "standard":
                ref_self_out = sinkhorn_attn(
                    q_det,
                    ref_k,
                    ref_v,
                    n_iters=sinkhorn_iters,
                    tau=sinkhorn_tau,
                )
            else:
                raise ValueError(f"Unsupported sinkhorn_mode: {sinkhorn_mode}")
        else:
            ref_self_out = F.scaled_dot_product_attention(q_det, ref_k, ref_v)
        ref_self_out_list.append(ref_self_out)

    style_loss = sum(F.l1_loss(self_out, ref_self_out.detach(), reduction="mean") for self_out, ref_self_out in zip(self_out_list, ref_self_out_list))
    target_norm = torch.stack([ref_self_out.float().norm() for ref_self_out in ref_self_out_list]).mean()
    self_out_norm = torch.stack([self_out.float().norm() for self_out in self_out_list]).mean()
    return KVMatchLossResult(
        loss=style_loss,
        tau=float(sinkhorn_tau),
        layer_count=n_layers,
        ot_layer_count=n_ot_layers,
        target_norm=target_norm,
        self_out_norm=self_out_norm,
    )


__all__ = [
    "KVMatchLossResult",
    "apply_reference_kv_grag",
    "kv_match_loss_from_attention_layers",
]
