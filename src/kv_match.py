"""Matched K/V-replacement teacher orchestration for the TPAMI latent loop."""

from __future__ import annotations

from typing import Sequence

import torch

from .attention_hooks import Klein9BCache, Klein9BController
from .kv_match_objective import KVMatchLossResult, KVMode, apply_reference_kv_grag, kv_match_loss_from_attention_layers


Tensor = torch.Tensor


def capture_reference_kv(
    *,
    dit,
    cache: Klein9BCache,
    controller: Klein9BController,
    style_latent_t: Tensor,
    model_timestep: Tensor,
    null_prompt_embeds: Tensor,
    null_text_ids: Tensor,
    style_image_ids: Tensor,
    kv_grag_shallow: float = 1.0,
    kv_grag_deep: float = 1.0,
) -> tuple[list[Tensor], list[Tensor], int]:
    """Cache reference K/V from the same timestep's noisy style latent."""

    cache.clear()
    controller.step()
    with torch.no_grad():
        _ = dit(
            hidden_states=style_latent_t,
            timestep=model_timestep,
            guidance=None,
            encoder_hidden_states=null_prompt_embeds,
            txt_ids=null_text_ids,
            img_ids=style_image_ids,
        )
    _, ref_k_list, ref_v_list, _ = cache.get()
    txt_len = int(controller.txt_len)
    cache.clear()
    controller.step()
    ref_k_list, ref_v_list = apply_reference_kv_grag(
        ref_k_list,
        ref_v_list,
        txt_len=txt_len,
        shallow=kv_grag_shallow,
        deep=kv_grag_deep,
    )
    return ref_k_list, ref_v_list, txt_len


def compute_kv_match_style_loss(
    *,
    dit,
    cache: Klein9BCache,
    controller: Klein9BController,
    latents: Tensor,
    style_latent_t: Tensor,
    model_timestep: Tensor,
    null_prompt_embeds: Tensor,
    null_text_ids: Tensor,
    image_ids: Tensor,
    style_image_ids: Tensor,
    sinkhorn_tau: float,
    sinkhorn_iters: int = 6,
    ot_layer_frac: float = 0.5,
    sinkhorn_mode: KVMode = "freq",
    freq_hi_boost: float = 2.0,
    kv_grag_shallow: float = 1.0,
    kv_grag_deep: float = 1.0,
) -> KVMatchLossResult:
    """Compute original K/V-replacement style loss at a post-Step1 latent."""

    ref_k_list, ref_v_list, _txt_len = capture_reference_kv(
        dit=dit,
        cache=cache,
        controller=controller,
        style_latent_t=style_latent_t,
        model_timestep=model_timestep,
        null_prompt_embeds=null_prompt_embeds,
        null_text_ids=null_text_ids,
        style_image_ids=style_image_ids,
        kv_grag_shallow=kv_grag_shallow,
        kv_grag_deep=kv_grag_deep,
    )

    cache.clear()
    controller.step()
    with torch.enable_grad():
        _ = dit(
            hidden_states=latents,
            timestep=model_timestep,
            guidance=None,
            encoder_hidden_states=null_prompt_embeds,
            txt_ids=null_text_ids,
            img_ids=image_ids,
        )
    q_list, _, _, self_out_list = cache.get()
    result = kv_match_loss_from_attention_layers(
        self_out_list=self_out_list,
        q_list=q_list,
        ref_k_list=ref_k_list,
        ref_v_list=ref_v_list,
        sinkhorn_tau=sinkhorn_tau,
        sinkhorn_iters=sinkhorn_iters,
        ot_layer_frac=ot_layer_frac,
        sinkhorn_mode=sinkhorn_mode,
        freq_hi_boost=freq_hi_boost,
    )
    cache.clear()
    controller.step()
    return result


__all__ = [
    "KVMatchLossResult",
    "capture_reference_kv",
    "compute_kv_match_style_loss",
    "kv_match_loss_from_attention_layers",
]
