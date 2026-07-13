"""Attention hook utilities for Klein9B/Klein style experiments.

Only hook wiring and state handling are included. No model loading or experiment
control logic is placed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .latent_ops import sinkhorn_attn, sinkhorn_style_pattern


KVTensor = torch.Tensor


class Klein9BCache:
    """Collect q/k/v/attn outputs for selected self-attention layers."""

    def __init__(self) -> None:
        self.q: List[KVTensor] = []
        self.k: List[KVTensor] = []
        self.v: List[KVTensor] = []
        self.out: List[KVTensor] = []

    def clear(self) -> None:
        self.q.clear()
        self.k.clear()
        self.v.clear()
        self.out.clear()

    def add(self, q: KVTensor, k: KVTensor, v: KVTensor, out: KVTensor) -> None:
        self.q.append(q)
        self.k.append(k)
        self.v.append(v)
        self.out.append(out)

    def get(self):
        return self.q.copy(), self.k.copy(), self.v.copy(), self.out.copy()


@dataclass
class Klein9BController:
    """Runtime state consumed by the hooked attention modules."""

    self_layers: Tuple[int, int] = (0, 16)
    mode: str = "cache"
    cur_self_layer: int = 0
    txt_len: int = 0

    inject_ref_k: List[KVTensor] | None = None
    inject_ref_v: List[KVTensor] | None = None
    inject_beta: float = 0.0
    inject_idx: int = 0

    sinkhorn_tau: float = 0.1
    sinkhorn_iters: int = 6
    sinkhorn_only_last_n: int = 0
    sinkhorn_use_pattern: bool = False

    def __post_init__(self) -> None:
        self.inject_ref_k = []
        self.inject_ref_v = []

    def step(self) -> None:
        self.cur_self_layer = 0
        self.inject_idx = 0

    def set_inject(self, ref_k_list: List[KVTensor], ref_v_list: List[KVTensor], beta: float) -> None:
        self.mode = "inject"
        self.inject_ref_k = ref_k_list
        self.inject_ref_v = ref_v_list
        self.inject_beta = float(beta)
        self.inject_idx = 0

    def set_sinkhorn(
        self,
        ref_k_list: List[KVTensor],
        ref_v_list: List[KVTensor],
        tau: float = 0.1,
        n_iters: int = 6,
        beta: float = 1.0,
        only_last_n: int = 0,
        use_pattern: bool = False,
    ) -> None:
        self.mode = "sinkhorn"
        self.inject_ref_k = ref_k_list
        self.inject_ref_v = ref_v_list
        self.sinkhorn_tau = float(tau)
        self.sinkhorn_iters = int(n_iters)
        self.inject_beta = float(beta)
        self.sinkhorn_only_last_n = int(only_last_n)
        self.sinkhorn_use_pattern = bool(use_pattern)
        self.inject_idx = 0

    def reset_mode(self) -> None:
        self.mode = "cache"
        self.inject_ref_k = []
        self.inject_ref_v = []
        self.inject_beta = 0.0
        self.inject_idx = 0


def _resolve_txt_len(
    controller: Klein9BController,
    encoder_hidden_states: Optional[torch.Tensor],
    computed_txt_len: int,
) -> int:
    if computed_txt_len > 0:
        controller.txt_len = computed_txt_len
    return int(controller.txt_len)


def _style_blend(
    attn_out_t: KVTensor,
    q_t: KVTensor,
    v_t: KVTensor,
    controller: Klein9BController,
    txt_len: int,
    ref_k: KVTensor,
    ref_v: KVTensor,
) -> KVTensor:
    if q_t.shape[2] <= txt_len:
        return attn_out_t

    q_img = q_t[:, :, txt_len:, :]
    beta = controller.inject_beta

    if controller.mode == "inject":
        style_out = F.scaled_dot_product_attention(q_img, ref_k, ref_v)
    elif controller.mode == "sinkhorn":
        use_its = len(controller.inject_ref_k)
        should_apply_ot = controller.sinkhorn_only_last_n <= 0 or controller.inject_idx >= use_its - controller.sinkhorn_only_last_n
        if not should_apply_ot:
            return attn_out_t

        if controller.sinkhorn_use_pattern:
            v_img = v_t[:, :, txt_len:, :]
            style_out = sinkhorn_style_pattern(
                ref_k,
                v_img,
                n_iters=controller.sinkhorn_iters,
                tau=controller.sinkhorn_tau,
                txt_len=txt_len,
            )
        else:
            style_out = sinkhorn_attn(
                q_img,
                ref_k,
                ref_v,
                n_iters=controller.sinkhorn_iters,
                tau=controller.sinkhorn_tau,
            )
    else:
        return attn_out_t

    style_len = style_out.shape[2]
    content_len = attn_out_t.shape[2] - txt_len
    if style_len < content_len:
        style_out = F.pad(style_out, (0, 0, 0, content_len - style_len))
    elif style_len > content_len:
        style_out = style_out[:, :, :content_len, :]

    blended = (1.0 - beta) * attn_out_t[:, :, txt_len:, :] + beta * style_out
    out = attn_out_t.clone()
    out[:, :, txt_len:, :] = blended
    return out


def register_klein9b_attn_hooks(dit, controller: Klein9BController, cache: Klein9BCache) -> None:
    """Patch Klein9B attention modules with a cache/inject/sinkhorn hook pair."""

    # Late import to keep this module importable without heavy deps during sanity checks.
    from diffsynth.models.flux2_dit import _get_qkv_projections, apply_rotary_emb

    def make_double_hook(_original_attn):
        def hooked_processor(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn,
                hidden_states,
                encoder_hidden_states,
            )
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            key = attn.norm_k(key)

            txt_len = 0
            if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)
                txt_len = int(encoder_hidden_states.shape[1])
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            query = query.to(hidden_states.dtype)
            key = key.to(hidden_states.dtype)
            value = value.to(hidden_states.dtype)

            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)

            current_txt_len = _resolve_txt_len(controller, encoder_hidden_states, txt_len)

            if controller.cur_self_layer in range(*controller.self_layers):
                cache.add(q_t, k_t, v_t, attn_out_t)
                if controller.mode in {"inject", "sinkhorn"} and controller.inject_idx < len(controller.inject_ref_k):
                    attn_out_t = _style_blend(
                        attn_out_t=attn_out_t,
                        q_t=q_t,
                        v_t=v_t,
                        controller=controller,
                        txt_len=current_txt_len,
                        ref_k=controller.inject_ref_k[controller.inject_idx],
                        ref_v=controller.inject_ref_v[controller.inject_idx],
                    )
                    controller.inject_idx += 1

            controller.cur_self_layer += 1

            attn_output = attn_out_t.transpose(1, 2)
            attn_output = attn_output.flatten(2, 3).to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_out, img_out = attn_output.split([txt_len, attn_output.shape[1] - txt_len], dim=1)
                img_out = attn.to_out[0](img_out)
                img_out = attn.to_out[1](img_out)
                encoder_out = attn.to_add_out(encoder_out)
                return img_out, encoder_out

            attn_output = attn.to_out[0](attn_output)
            attn_output = attn.to_out[1](attn_output)
            return attn_output

        return hooked_processor

    def make_single_hook(_original_attn):
        def hooked_processor(attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            proj_out = attn.to_qkv_mlp_proj(hidden_states)
            qkv, mlp_hidden_states = torch.split(
                proj_out,
                [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
                dim=-1,
            )
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            query = query.to(hidden_states.dtype)
            key = key.to(hidden_states.dtype)
            value = value.to(hidden_states.dtype)

            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)
            attn_out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)

            current_txt_len = int(controller.txt_len)

            if controller.cur_self_layer in range(*controller.self_layers):
                cache.add(q_t, k_t, v_t, attn_out_t)
                if controller.mode in {"inject", "sinkhorn"} and controller.inject_idx < len(controller.inject_ref_k):
                    attn_out_t = _style_blend(
                        attn_out_t=attn_out_t,
                        q_t=q_t,
                        v_t=v_t,
                        controller=controller,
                        txt_len=current_txt_len,
                        ref_k=controller.inject_ref_k[controller.inject_idx],
                        ref_v=controller.inject_ref_v[controller.inject_idx],
                    )
                    controller.inject_idx += 1

            controller.cur_self_layer += 1

            attn_output = attn_out_t.transpose(1, 2).flatten(2, 3).to(query.dtype)
            mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
            hidden_states = attn.to_out(hidden_states)
            return hidden_states

        return hooked_processor

    for block in dit.transformer_blocks:
        block.attn.processor = make_double_hook(block.attn)
    for block in dit.single_transformer_blocks:
        block.attn.processor = make_single_hook(block.attn)
