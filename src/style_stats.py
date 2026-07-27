"""Content-agnostic texture/style-statistics losses for real style transfer.

These match the *distribution* of the generated clean latent x0's per-token feature vectors
to the style reference latent — transferring texture/color statistics everywhere (subject
included) instead of editing toward the conditioned sample (which is what the CA/CFG signal
does). Computed on x0/latent ONLY — never on attention Q/K/V — so it stays clearly distinct
from the reserved first-gen K/V method.

All inputs are (B, N, C) latent-token tensors (B=1 here); stats are over the N spatial tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _tok(x: Tensor) -> Tensor:
    return (x[0] if x.ndim == 3 else x).float()


def gram_loss(gen: Tensor, ref: Tensor, *, center: bool = True) -> Tensor:
    """Match channel-covariance (Gram) over spatial tokens. Layout-agnostic 2nd-order texture."""
    g, r = _tok(gen), _tok(ref)                       # (Ng,C),(Nr,C)
    if center:
        g = g - g.mean(0, keepdim=True)
        r = r - r.mean(0, keepdim=True)
    Gg = (g.t() @ g) / g.shape[0]                     # (C,C)
    Gr = (r.t() @ r) / r.shape[0]
    return F.mse_loss(Gg, Gr)


def mean_std_loss(gen: Tensor, ref: Tensor) -> Tensor:
    """AdaIN-style 1st/2nd moment match (palette + contrast). Cheap companion / floor."""
    g, r = _tok(gen), _tok(ref)
    return F.mse_loss(g.mean(0), r.mean(0)) + F.mse_loss(g.std(0), r.std(0))


def sliced_wasserstein_loss(gen: Tensor, ref: Tensor, *, n_proj: int = 128, eps: float = 1e-8) -> Tensor:
    """Sliced-Wasserstein between the two token-feature distributions.

    Project the C-dim token vectors onto n_proj random unit directions, sort each 1-D marginal,
    and L2-match the sorted marginals (= 1-D optimal transport per slice, averaged). Matches the
    FULL distribution (not just 2nd moment), is content-agnostic, and never touches attention.
    """
    g, r = _tok(gen), _tok(ref)                       # (Ng,C),(Nr,C)
    C = g.shape[1]
    dirs = torch.randn(C, n_proj, device=g.device, dtype=g.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(eps)
    pg = (g @ dirs).sort(dim=0).values                # (Ng,n_proj) sorted per slice
    pr = (r @ dirs).sort(dim=0).values                # (Nr,n_proj)
    if pg.shape[0] != pr.shape[0]:                    # resample ref marginals to gen length
        pr = F.interpolate(pr.t().unsqueeze(0), size=pg.shape[0], mode="linear", align_corners=True).squeeze(0).t()
    return F.mse_loss(pg, pr)


def style_stats_loss(kind: str, gen: Tensor, ref: Tensor, *, n_proj: int = 128) -> Tensor:
    if kind == "gram":
        return gram_loss(gen, ref)
    if kind == "sw":
        return sliced_wasserstein_loss(gen, ref, n_proj=n_proj)
    if kind == "meanstd":
        return mean_std_loss(gen, ref)
    if kind == "sw+gram":
        return sliced_wasserstein_loss(gen, ref, n_proj=n_proj) + gram_loss(gen, ref)
    raise ValueError(f"unknown style-stats kind: {kind!r}")


# --- VGG perceptual (Gatys) style loss on decoded pixels -------------------------------
# The canonical texture-transfer objective: match multi-layer VGG Gram matrices between the
# generated (decoded) image and the style reference. Richer than latent stats. Operates on
# decoded PIXELS + a frozen VGG — never on the diffusion model's Q/K/V (K/V taboo held).
_VGG_CACHE: dict = {}
_VGG_TAPS = (1, 6, 11, 20)   # relu1_1, relu2_1, relu3_1, relu4_1

def _vgg_net(device):
    if "net" not in _VGG_CACHE:
        from torchvision.models import vgg19, VGG19_Weights
        net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        _VGG_CACHE["net"] = net
    return _VGG_CACHE["net"]

def _vgg_grams(x: Tensor):
    """x: (B,3,H,W) in [0,1]. Returns list of Gram matrices at the tap layers."""
    device = x.device
    net = _vgg_net(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    h = (x.float() - mean) / std
    grams, last = [], max(_VGG_TAPS)
    for i, layer in enumerate(net):
        h = layer(h)
        if i in _VGG_TAPS:
            B, C, H, W = h.shape
            f = h.reshape(B, C, H * W)
            grams.append((f @ f.transpose(1, 2)) / (C * H * W))
        if i >= last:
            break
    return grams

def vgg_gram_targets(ref_pix: Tensor):
    with torch.no_grad():
        return [g.detach() for g in _vgg_grams(ref_pix)]

def vgg_gram_loss(gen_pix: Tensor, targets) -> Tensor:
    return sum(F.mse_loss(g, t) for g, t in zip(_vgg_grams(gen_pix), targets))


__all__ = ["gram_loss", "sliced_wasserstein_loss", "mean_std_loss", "style_stats_loss",
           "vgg_gram_targets", "vgg_gram_loss"]
