"""Quick import + shape sanity tests for minimal project modules."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_shape_smoke(torch) -> None:
    from src.latent_ops import (
        sinkhorn_attn,
        sinkhorn_attn_freq,
        sinkhorn_style_pattern,
        feature_adain,
        adain_1d,
        sinkhorn_latent_transport,
    )

    torch.manual_seed(0)
    b, h, lq, lk, d = 2, 4, 9, 6, 16
    q = torch.randn(b, h, lq, d)
    k = torch.randn(b, h, lk, d)
    v = torch.randn(b, h, lk, d)

    out_attn = sinkhorn_attn(q, k, v)
    out_freq = sinkhorn_attn_freq(q, k, v)
    v_content = torch.randn(b, h, 7, d)
    out_pattern = sinkhorn_style_pattern(k, v_content)

    out_style = feature_adain(q, v[:, :, :lq, :], alpha=0.5)
    out_adain = adain_1d(q.mean(dim=2), v.mean(dim=2))

    content_lat = torch.randn(b, 10, d)
    style_noisy = torch.randn(b, 10, d)
    style_clean = torch.randn(b, 10, d)
    out_transport = sinkhorn_latent_transport(content_lat, style_noisy, style_clean)

    assert out_attn.shape == q.shape
    assert out_freq.shape == q.shape
    assert out_pattern.shape == v_content.shape
    assert out_style.shape == q.shape
    assert out_adain.shape == q.mean(dim=2).shape
    assert out_transport.shape == content_lat.shape


def _run_velocity_smoke(torch) -> None:
    from src.attention_hooks import Klein9BCache, Klein9BController
    from src.residuals import compute_velocity_tuple

    class FakeDiT:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, **kwargs):
            if any(
                name in kwargs
                for name in [
                    "inject_beta",
                    "sinkhorn_tau",
                    "sinkhorn_iters",
                    "sinkhorn_only_last_n",
                    "sinkhorn_use_pattern",
                ]
            ):
                unexpected = [
                    name
                    for name in [
                        "inject_beta",
                        "sinkhorn_tau",
                        "sinkhorn_iters",
                        "sinkhorn_only_last_n",
                        "sinkhorn_use_pattern",
                    ]
                    if name in kwargs
                ]
                raise AssertionError(f"unexpected kwargs leaked to dit: {unexpected}")
            self.calls.append(kwargs)
            hidden_states = kwargs["hidden_states"]
            bias = float(len(self.calls) - 1)
            return hidden_states + bias

    cache = Klein9BCache()
    controller = Klein9BController()
    fake_dit = FakeDiT()

    b, lz, d = 2, 1, 8
    latents = torch.randn(b, lz, d)
    normal_v = torch.randn(b, lz, d)
    timestep = torch.tensor([1.0])
    prompt_embeds = torch.randn(b, 1, d)
    txt_ids = torch.zeros(b, 1, dtype=torch.int64)
    image_ids = torch.zeros(b, 1, dtype=torch.int64)

    normal_out, styled_out, residual_out, guided_out = compute_velocity_tuple(
        dit=fake_dit,
        latents=latents,
        timestep=timestep,
        guidance=None,
        prompt_embeds=prompt_embeds,
        txt_ids=txt_ids,
        image_ids=image_ids,
        controller=controller,
        cache=cache,
        ref_k_list=[torch.randn(b, 4, 1, d)],
        ref_v_list=[torch.randn(b, 4, 1, d)],
        mode="inject",
        guidance_scale=0.35,
        inject_beta=0.8,
        sinkhorn_tau=0.2,
        sinkhorn_iters=4,
        sinkhorn_only_last_n=0,
        sinkhorn_use_pattern=False,
        model_kwargs={"extra_model_kw": "ok"},
    )

    expected_normal = latents + 0.0
    expected_styled = latents + 1.0

    assert torch.allclose(normal_out, expected_normal)
    assert torch.allclose(styled_out, expected_styled)
    assert torch.allclose(residual_out, expected_styled - expected_normal)
    assert torch.allclose(guided_out, expected_normal + 0.35 * (expected_styled - expected_normal))


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("OK (torch not installed; skipped)")
        return

    _run_shape_smoke(torch)
    _run_velocity_smoke(torch)
    print("OK")


if __name__ == "__main__":
    main()
