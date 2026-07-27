"""Native FLUX.2 edit/reference teacher helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


Tensor = torch.Tensor


def freeze_condition_tensor(
    tensor: Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    kwargs: dict[str, object] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    if kwargs:
        tensor = tensor.to(**kwargs)
    return tensor.detach()


def assert_static_condition_tensor(name: str, tensor: Tensor) -> None:
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise RuntimeError(
            f"{name} must be a detached constant condition tensor; "
            f"got requires_grad={tensor.requires_grad} grad_fn={tensor.grad_fn!r}"
        )


@dataclass(frozen=True)
class EditCondition:
    """Prepared native edit-image condition for ``model_fn_flux2``."""

    edit_latents: Tensor
    edit_image_ids: Tensor
    source: str


@dataclass(frozen=True)
class NativeEditTeacherTriplet:
    """Velocity branches evaluated at the same post-Step1 state."""

    pre_velocity: Tensor
    without_velocity: Tensor
    with_velocity: Tensor
    without_shared_with_pre: bool


def prepare_edit_condition(
    pipe: Any,
    edit_image: Any,
    *,
    auto_resize: bool = False,
) -> EditCondition:
    """Encode the reference image through FLUX.2's native edit path."""

    from diffsynth.pipelines.flux2_image import Flux2Unit_EditImageEmbedder

    unit = Flux2Unit_EditImageEmbedder()
    with torch.no_grad():
        encoded = unit.process(pipe, edit_image, edit_image_auto_resize=auto_resize)
    if "edit_latents" not in encoded or "edit_image_ids" not in encoded:
        raise RuntimeError("Native edit image encoding failed to produce edit_latents/edit_image_ids")
    source = str(edit_image if isinstance(edit_image, (str, Path)) else "<in-memory>")
    return EditCondition(
        edit_latents=freeze_condition_tensor(
            encoded["edit_latents"],
            device=pipe.device,
            dtype=pipe.torch_dtype,
        ),
        edit_image_ids=freeze_condition_tensor(
            encoded["edit_image_ids"],
            device=pipe.device,
        ),
        source=source,
    )


def build_layout_free_edit_condition(
    edit_condition: EditCondition,
    *,
    mode: str,
    seed: int,
) -> EditCondition:
    """Produce a "layout-free" variant of ``edit_condition`` by manipulating ONLY the positional
    ``edit_image_ids`` -- never the content ``edit_latents``, never K/V, never any attention code.

    ``edit_image_ids`` has shape ``(B, N, 4)`` where each of the ``N`` reference tokens carries a
    ``(t, h, w, l)`` positional coordinate consumed only by the frozen DiT's RoPE (``img_ids``).
    Because the reference tokens normally sit on a coherent grid matching the generated grid, the
    generated token at position (i, j) aligns to the reference token at (i, j) and the reference's
    spatial LAYOUT/composition is copied. Breaking that positional correspondence keeps the
    reference's APPEARANCE/medium contribution (content-similarity attention) while removing layout.

    Two modes:
    - ``scramble``: deterministically permute the rows (token dim=1) of ``edit_image_ids`` with a
      LOCAL seeded ``torch.Generator`` (never the global RNG). The content ``edit_latents`` stay put,
      so each reference token keeps its content but now claims a *random* position id.
    - ``strip``: broadcast the first row of ``edit_image_ids`` across all ``N`` tokens so every
      reference token is positionally identical (a pure position-invariant bag-of-appearance).

    The returned ``EditCondition`` shares the exact same ``edit_latents`` tensor; only the frozen,
    detached ``edit_image_ids`` differ.
    """

    if mode not in {"scramble", "strip"}:
        raise ValueError(f"Unsupported layout-free mode {mode!r}; expected 'scramble' or 'strip'")
    ids = edit_condition.edit_image_ids
    if ids.ndim != 3:
        raise RuntimeError(
            f"edit_image_ids expected shape (B, N, C); got {tuple(ids.shape)} for layout-free {mode}"
        )
    num_tokens = int(ids.shape[1])
    if mode == "scramble":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(num_tokens, generator=generator).to(device=ids.device)
        new_ids = ids.index_select(1, perm)
    else:  # strip
        new_ids = ids[:, 0:1, :].expand(-1, num_tokens, -1).contiguous()
    return EditCondition(
        edit_latents=edit_condition.edit_latents,
        edit_image_ids=freeze_condition_tensor(new_ids.detach(), device=ids.device),
        source=f"{edit_condition.source}::layout_free_{mode}",
    )


def _default_velocity_fn(
    *,
    dit: Any,
    latents: Tensor,
    timestep: Tensor,
    embedded_guidance: float,
    prompt_embeds: Tensor,
    text_ids: Tensor,
    image_ids: Tensor,
    edit_condition: EditCondition | None,
    no_grad: bool,
) -> Tensor:
    from diffsynth.pipelines.flux2_image import model_fn_flux2

    kwargs: dict[str, Any] = {
        "dit": dit,
        "latents": latents,
        "timestep": timestep,
        "embedded_guidance": embedded_guidance,
        "prompt_embeds": prompt_embeds,
        "text_ids": text_ids,
        "image_ids": image_ids,
    }
    if edit_condition is not None:
        kwargs["edit_latents"] = edit_condition.edit_latents
        kwargs["edit_image_ids"] = edit_condition.edit_image_ids
    if no_grad:
        with torch.no_grad():
            return model_fn_flux2(**kwargs)
    return model_fn_flux2(**kwargs)


def compute_native_velocity(
    *,
    dit: Any,
    latents: Tensor,
    timestep: Tensor,
    embedded_guidance: float,
    prompt_embeds: Tensor,
    text_ids: Tensor,
    image_ids: Tensor,
    edit_condition: EditCondition | None,
    no_grad: bool,
) -> Tensor:
    return _default_velocity_fn(
        dit=dit,
        latents=latents,
        timestep=timestep,
        embedded_guidance=embedded_guidance,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        image_ids=image_ids,
        edit_condition=edit_condition,
        no_grad=no_grad,
    )


def evaluate_native_edit_triplet(
    *,
    dit: Any,
    latents: Tensor,
    timestep: Tensor,
    embedded_guidance: float,
    prompt_embeds: Tensor,
    text_ids: Tensor,
    image_ids: Tensor,
    edit_condition: EditCondition,
    negative_prompt_embeds: Tensor | None = None,
    negative_text_ids: Tensor | None = None,
    require_pre_grad: bool,
    share_without_with_pre: bool = True,
    velocity_fn: Callable[..., Tensor] | None = None,
) -> NativeEditTeacherTriplet:
    """Evaluate Plan A/B branches at the same latent state with explicit conditioning contrast.

    All branches share the same ``z_sigma`` state and timestep. The positive branch may keep the
    full style prompt plus reference condition, while the negative/pre branch may optionally switch
    to a different prompt with no reference condition.
    """

    velocity_impl = velocity_fn or _default_velocity_fn
    if negative_prompt_embeds is None:
        negative_prompt_embeds = prompt_embeds
    if negative_text_ids is None:
        negative_text_ids = text_ids
    pre_velocity = velocity_impl(
        dit=dit,
        latents=latents,
        timestep=timestep,
        embedded_guidance=embedded_guidance,
        prompt_embeds=negative_prompt_embeds,
        text_ids=negative_text_ids,
        image_ids=image_ids,
        edit_condition=None,
        no_grad=not require_pre_grad,
    )
    if share_without_with_pre:
        without_velocity = pre_velocity.detach()
    else:
        without_velocity = velocity_impl(
            dit=dit,
            latents=latents,
            timestep=timestep,
            embedded_guidance=embedded_guidance,
            prompt_embeds=negative_prompt_embeds,
            text_ids=negative_text_ids,
            image_ids=image_ids,
            edit_condition=None,
            no_grad=True,
        )
    with_velocity = velocity_impl(
        dit=dit,
        latents=latents,
        timestep=timestep,
        embedded_guidance=embedded_guidance,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        image_ids=image_ids,
        edit_condition=edit_condition,
        no_grad=True,
    )
    return NativeEditTeacherTriplet(
        pre_velocity=pre_velocity,
        without_velocity=without_velocity.detach(),
        with_velocity=with_velocity.detach(),
        without_shared_with_pre=bool(share_without_with_pre),
    )


__all__ = [
    "EditCondition",
    "NativeEditTeacherTriplet",
    "assert_static_condition_tensor",
    "build_layout_free_edit_condition",
    "compute_native_velocity",
    "evaluate_native_edit_triplet",
    "freeze_condition_tensor",
    "prepare_edit_condition",
]
