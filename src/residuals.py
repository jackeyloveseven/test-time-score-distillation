"""Lightweight residual wrappers for Klein9B/FLUX.2 style guidance.

These helpers are intentionally thin: they only orchestrate model calls and simple
algebra around them. The result is a **velocity residual** (not direct score).
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

from .attention_hooks import Klein9BCache, Klein9BController

ModelCallable = Callable[..., torch.Tensor]
ModelKwargs = Optional[Mapping[str, object]]


def _build_model_kwargs(
    guidance: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    image_ids: torch.Tensor,
    additional: ModelKwargs,
) -> Dict[str, object]:
    model_kwargs: Dict[str, object] = {
        "guidance": guidance,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": txt_ids,
        "img_ids": image_ids,
    }
    if additional:
        model_kwargs.update(additional)
    return model_kwargs


def compute_normal_velocity(
    dit: ModelCallable,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    guidance: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    image_ids: torch.Tensor,
    model_kwargs: ModelKwargs = None,
    no_grad: bool = True,
) -> torch.Tensor:
    """Compute normal Klein9B velocity at ``(z_t, t)`` with no style injection."""
    call_kwargs = _build_model_kwargs(guidance, prompt_embeds, txt_ids, image_ids, model_kwargs)
    if no_grad:
        with torch.no_grad():
            return dit(
                hidden_states=latents,
                timestep=timestep,
                **call_kwargs,
            )
    return dit(hidden_states=latents, timestep=timestep, **call_kwargs)


def compute_styled_velocity(
    dit: ModelCallable,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    guidance: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    image_ids: torch.Tensor,
    controller: Klein9BController,
    cache: Klein9BCache,
    ref_k_list: Sequence[torch.Tensor],
    ref_v_list: Sequence[torch.Tensor],
    model_kwargs: ModelKwargs = None,
    mode: str = "inject",
    inject_beta: float = 1.0,
    sinkhorn_tau: float = 0.1,
    sinkhorn_iters: int = 6,
    sinkhorn_only_last_n: int = 0,
    sinkhorn_use_pattern: bool = False,
    set_mode: bool = True,
    reset_mode: bool = True,
    no_grad: bool = True,
) -> torch.Tensor:
    """Compute styled Klein9B velocity with hard inject or sinkhorn branch.

    This function intentionally does not call ``scheduler.step``. It only returns the
    styled prediction evaluated at the same ``(z_t, t)`` as the normal branch.
    """
    if len(ref_k_list) == 0 or len(ref_v_list) == 0:
        raise ValueError("ref_k_list/ref_v_list must be non-empty for styled velocity")

    call_kwargs = _build_model_kwargs(guidance, prompt_embeds, txt_ids, image_ids, model_kwargs)
    cache.clear()
    controller.step()

    if set_mode:
        if mode == "inject":
            controller.set_inject(ref_k_list, ref_v_list, inject_beta)
        elif mode == "sinkhorn":
            controller.set_sinkhorn(
                ref_k_list,
                ref_v_list,
                tau=sinkhorn_tau,
                n_iters=sinkhorn_iters,
                beta=inject_beta,
                only_last_n=sinkhorn_only_last_n,
                use_pattern=sinkhorn_use_pattern,
            )
        else:
            raise ValueError("mode must be one of: inject, sinkhorn")

    if no_grad:
        with torch.no_grad():
            out = dit(
                hidden_states=latents,
                timestep=timestep,
                **call_kwargs,
            )
    else:
        out = dit(hidden_states=latents, timestep=timestep, **call_kwargs)

    if reset_mode:
        controller.reset_mode()
    return out


def compute_style_velocity_residual(
    normal_velocity: torch.Tensor,
    styled_velocity: torch.Tensor,
) -> torch.Tensor:
    """Return Klein9B/FLUX2 style velocity residual: ``styled - normal``."""
    return styled_velocity - normal_velocity


def apply_velocity_guidance(
    normal_velocity: torch.Tensor,
    style_velocity: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply velocity residual guidance without scheduling.

    In Klein9B/FLUX.2 this is a **velocity residual** formulation:
    ``v_guided = v_normal + scale * (v_styled - v_normal)``.
    """
    return normal_velocity + scale * (style_velocity - normal_velocity)


def compute_velocity_tuple(
    dit: ModelCallable,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    guidance: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    image_ids: torch.Tensor,
    controller: Klein9BController,
    cache: Klein9BCache,
    ref_k_list: Sequence[torch.Tensor],
    ref_v_list: Sequence[torch.Tensor],
    model_kwargs: ModelKwargs = None,
    mode: str = "inject",
    guidance_scale: float = 1.0,
    inject_beta: float = 1.0,
    sinkhorn_tau: float = 0.1,
    sinkhorn_iters: int = 6,
    sinkhorn_only_last_n: int = 0,
    sinkhorn_use_pattern: bool = False,
    no_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normal/styled/residual/guided velocity in one call.

    Normal and styled are both evaluated at same ``(z_t, t)``. The caller should call
    ``scheduler.step(guided_velocity, t, z_t)`` exactly once.
    """
    normal_v = compute_normal_velocity(
        dit,
        latents=latents,
        timestep=timestep,
        guidance=guidance,
        prompt_embeds=prompt_embeds,
        txt_ids=txt_ids,
        image_ids=image_ids,
        model_kwargs=model_kwargs,
        no_grad=no_grad,
    )

    styled_v = compute_styled_velocity(
        dit,
        latents=latents,
        timestep=timestep,
        guidance=guidance,
        prompt_embeds=prompt_embeds,
        txt_ids=txt_ids,
        image_ids=image_ids,
        controller=controller,
        cache=cache,
        ref_k_list=ref_k_list,
        ref_v_list=ref_v_list,
        model_kwargs=model_kwargs,
        mode=mode,
        inject_beta=inject_beta,
        sinkhorn_tau=sinkhorn_tau,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_only_last_n=sinkhorn_only_last_n,
        sinkhorn_use_pattern=sinkhorn_use_pattern,
        reset_mode=True,
        set_mode=True,
        no_grad=no_grad,
    )

    residual_v = compute_style_velocity_residual(normal_v, styled_v)
    guided_v = apply_velocity_guidance(normal_v, styled_v, guidance_scale)
    return normal_v, styled_v, residual_v, guided_v
