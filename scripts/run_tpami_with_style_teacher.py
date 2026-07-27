"""Run a TPAMI-style latent loop with the FLUX.2 native edit teacher.

This keeps the original optimization skeleton:

1. standard denoise + ``scheduler.step`` once;
2. prepare the same post-Step1 latent state ``z_sigma`` at ``timestep_style=i+1``;
3. keep AdaIN / SLT warm-start behavior, with optional latent-space GRAG-style
   compression for the warm-start itself;
4. optimize the current latent with AdamW while preserving ``dist_loss`` and
   ``content_loss``;
5. replace only the old ``style_loss`` with a native edit teacher objective.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from tqdm import tqdm


# --- Env-gated Sinkhorn-attention PROBE (negative-study entry; NOT part of TSD) ----------
# Replaces the DiT's own softmax attention with a doubly-stochastic (Sinkhorn-normalized)
# version to test whether de-concentrating attention spreads style off the background onto
# the prompt-fixed subject (the "local-edit" question). Renormalizes ONLY the model's own
# q/k/v attention weights — never injects captured reference K/V (that is the reserved
# first-gen method and stays off-limits). Guarded by $SINKHORN_ATTN.
def _maybe_install_sinkhorn_attention():
    """Install doubly-stochastic (Sinkhorn) attention on the flux2 DiT if $SINKHORN_ATTN is
    truthy. MUST be called after args.diffsynth_root is on sys.path (inside main), before the
    DiT runs. Renormalizes the model's OWN attention only — never injects reference K/V."""
    _sk = os.environ.get("SINKHORN_ATTN", "")
    if not _sk or _sk in ("0", "false", "False", "no", "off"):
        return
    import diffsynth.models.flux2_dit as _f2dit

    import diffsynth.pipelines.flux2_image as _f2img
    _SK_ITERS = int(os.environ.get("SINKHORN_ITERS", "3"))
    _SK_MIX = float(os.environ.get("SINKHORN_MIX", "1.0"))
    _SK_SCOPE = os.environ.get("SINKHORN_SCOPE", "full")   # "full" | "nonedit"
    _f2dit._SK_EDIT_LEN = 0

    def _sinkhorn_attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d",
                                    v_pattern="b n s d", out_pattern="b n s d", dims=None,
                                    attn_mask=None, scale=None, compatibility_mode=False):
        _dims = {} if dims is None else dims
        if q_pattern != "b n s d":
            q = rearrange(q, f"{q_pattern} -> b n s d", **_dims)
        if k_pattern != "b n s d":
            k = rearrange(k, f"{k_pattern} -> b n s d", **_dims)
        if v_pattern != "b n s d":
            v = rearrange(v, f"{v_pattern} -> b n s d", **_dims)
        d = q.shape[-1]
        sc = (d ** -0.5) if scale is None else scale
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sc
        if attn_mask is not None:
            logits = logits + attn_mask
        A = torch.softmax(logits, dim=-1)              # standard row-stochastic attention
        if _SK_MIX > 0.0:
            n_edit = getattr(_f2dit, "_SK_EDIT_LEN", 0)
            if _SK_SCOPE == "nonedit" and n_edit and n_edit > 0 and A.shape[-1] > n_edit:
                # TARGETED: balance ONLY the non-edit key columns; keep each row's edit-vs-nonedit
                # attention mass split (so we never force the model to reproduce the reference).
                ne = A.shape[-1] - n_edit
                A_ne, A_e = A[..., :ne], A[..., ne:]
                row_mass = A_ne.sum(dim=-1, keepdim=True)           # per-row non-edit mass to preserve
                S = A_ne
                for _ in range(_SK_ITERS):
                    S = S / (S.sum(dim=-2, keepdim=True) + 1e-8)     # column (over queries)
                    S = S / (S.sum(dim=-1, keepdim=True) + 1e-8)     # row
                S = S * row_mass                                     # restore original non-edit mass per row
                A = torch.cat([(1.0 - _SK_MIX) * A_ne + _SK_MIX * S, A_e], dim=-1)
            else:                                                    # "full": whole joint attention
                S = A
                for _ in range(_SK_ITERS):
                    S = S / (S.sum(dim=-2, keepdim=True) + 1e-8)     # column normalize
                    S = S / (S.sum(dim=-1, keepdim=True) + 1e-8)     # row normalize
                A = (1.0 - _SK_MIX) * A + _SK_MIX * S
        out = torch.matmul(A.to(v.dtype), v)
        if out_pattern != "b n s d":
            out = rearrange(out, f"b n s d -> {out_pattern}", **_dims)
        return out

    _f2dit.attention_forward = _sinkhorn_attention_forward

    if _SK_SCOPE == "nonedit":     # need edit-token count per forward → wrap model_fn (like v-shuffle)
        _orig_mfn = _f2img.model_fn_flux2
        def _wrapped_mfn(*a, **kw):
            _ed = kw.get("edit_latents", None)
            _f2dit._SK_EDIT_LEN = int(_ed.shape[1]) if _ed is not None else 0
            try:
                return _orig_mfn(*a, **kw)
            finally:
                _f2dit._SK_EDIT_LEN = 0
        _f2img.model_fn_flux2 = _wrapped_mfn

    print(f"[SINKHORN_ATTN] active iters={_SK_ITERS} mix={_SK_MIX} scope={_SK_SCOPE} "
          "(doubly-stochastic OWN attention; NO reference K/V injected)", flush=True)
# ----------------------------------------------------------------------------------------


# ---- V-shuffle probe (env-gated $VSHUFFLE). USER-AUTHORIZED 2026-07-21 relaxation of the
# "don't touch K/V / no-attention-machinery" thesis taboo. V-shuffle (arXiv 2511.06365,
# EXTERNAL-published, NOT the first-gen Sinkhorn K/V-matching) permutes the reference (edit)
# tokens' Value rows inside the joint self-attention, decorrelating the reference's spatial
# structure so generated tokens read style-as-statistic instead of copying reference layout.
# This is a gated ONE-PROBE experiment to see if it removes local-edit — it is attention-V
# manipulation and reverses the paper thesis, so it is NOT part of the method. -------------
def _maybe_install_vshuffle():
    _vs = os.environ.get("VSHUFFLE", "")
    if not _vs or _vs in ("0", "false", "False", "no", "off"):
        return
    import diffsynth.models.flux2_dit as _f2dit
    import diffsynth.pipelines.flux2_image as _f2img
    from diffsynth.core.attention import attention_forward as _orig_attn

    _VS_SEED = int(os.environ.get("VSHUFFLE_SEED", "42"))
    _VS_FRAC = float(os.environ.get("VSHUFFLE_FRAC", "1.0"))  # ρ: fraction of edit-token V rows to
    #   destructure. ρ=1 ⇒ full v-shuffle; 0<ρ<1 ⇒ STRUCTURED partial permutation (a random ρ-subset
    #   of rows is permuted among itself, the rest kept EXACTLY in place). Not a blend: values stay
    #   exact (no averaging/mush), only a fraction of positions move ⇒ ρ is the edit↔transfer knob.
    _f2dit._VS_EDIT_LEN = 0  # N_edit for the CURRENT forward (0 ⇒ no edit tokens ⇒ no-op)

    def _vshuffle_attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d",
                                    v_pattern="b n s d", out_pattern="b n s d", dims=None,
                                    attn_mask=None, scale=None, compatibility_mode=False):
        n_edit = getattr(_f2dit, "_VS_EDIT_LEN", 0)
        if n_edit and n_edit > 0:
            _dims = {} if dims is None else dims
            v_bnsd = v if v_pattern == "b n s d" else rearrange(v, f"{v_pattern} -> b n s d", **_dims)
            S = v_bnsd.shape[2]
            if S > n_edit:                       # edit tokens are the LAST n_edit along the seq axis
                gen = torch.Generator(device=v_bnsd.device).manual_seed(_VS_SEED)
                gidx = torch.arange(n_edit, device=v_bnsd.device)
                if _VS_FRAC >= 1.0:              # full permutation
                    gidx = torch.randperm(n_edit, device=v_bnsd.device, generator=gen)
                else:                            # STRUCTURED partial: permute only a ρ-subset, keep rest exact
                    nk = max(1, int(round(_VS_FRAC * n_edit)))   # NOTE: not 'k' — 'k' is the attention key tensor
                    sel = torch.randperm(n_edit, device=v_bnsd.device, generator=gen)[:nk]
                    gidx[sel] = sel[torch.randperm(nk, device=v_bnsd.device, generator=gen)]
                head = v_bnsd[:, :, : S - n_edit, :]
                tail = v_bnsd[:, :, S - n_edit :, :][:, :, gidx, :]
                v, v_pattern = torch.cat([head, tail], dim=2), "b n s d"
        return _orig_attn(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims,
                          attn_mask=attn_mask, scale=scale, compatibility_mode=compatibility_mode)

    _f2dit.attention_forward = _vshuffle_attention_forward

    _orig_model_fn = _f2img.model_fn_flux2
    def _wrapped_model_fn(*a, **kw):
        _edit = kw.get("edit_latents", None)
        _f2dit._VS_EDIT_LEN = int(_edit.shape[1]) if _edit is not None else 0
        try:
            return _orig_model_fn(*a, **kw)
        finally:
            _f2dit._VS_EDIT_LEN = 0
    _f2img.model_fn_flux2 = _wrapped_model_fn

    print(f"[VSHUFFLE] active seed={_VS_SEED} — permuting reference/edit-token V rows in joint "
          "self-attention (external arXiv 2511.06365; USER-authorized taboo relaxation)", flush=True)
# ----------------------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ca_dmd_backend import (
    FakeConditioning,
    FakeSchedulerLookup,
    KleinFullFakeBackendConfig,
    build_fake_score_backend,
)
from src.attention_hooks import Klein9BCache, Klein9BController, register_klein9b_attn_hooks
from src.csv_rows import csv_fieldnames_for_rows
from src.carrier_score import affinity_transport_cosine, content_affinity_transport
from src.flux2_score import (
    build_pseudo_target_loss,
    resolve_sigma_info,
    velocity_delta_to_score_delta,
    velocity_to_x0,
)
from src.gradient_audit import StyleGradientCalibrationState, apply_gradient_audit, grad_cosine
from src.kv_match import capture_reference_kv
from src.kv_match_objective import kv_match_loss_from_attention_layers
from src.klein9b_loading import (
    model_config_offload_kwargs,
    model_loading_mode,
    resolve_dual_model_vram_contract,
)
from src.loss_isolation import (
    build_loss_isolation_contract,
    final_adain_if_enabled,
    is_true_loss_only,
)
from src.latent_ops import adain_1d, feature_adain, sinkhorn_latent_transport
from src.native_edit_teacher import (
    EditCondition,
    assert_static_condition_tensor,
    build_layout_free_edit_condition,
    compute_native_velocity,
    evaluate_native_edit_triplet,
    freeze_condition_tensor,
    prepare_edit_condition,
)
from src.prompt_text import (
    parse_content_prompt_text as build_content_prompt_text,
    parse_prompt_text as build_prompt_text,
)
from src.style_objectives import (
    layout_repel_velocity_delta,
    pseudo_guidance_from_velocities,
    score_match_loss_from_velocities,
    validate_style_mode,
)


PROMPTS = {
    "kp0": "A panda eating bamboo",
    "kp1": "A castle on a cliff by the sea",
    "kp2": "A cat sitting on a windowsill",
    "kp3": "A field of sunflowers at sunset",
    "kp4": "A portrait of an old fisherman",
    "kp5": "A lighthouse on a rocky coast",
}

FORMAL_DEFAULTS = {
    "num_steps": 25,
    "iters": 5,
    "lr": 0.01,
    "start_time": 850.0,
    "style_weight": 2.0,
    "dist_weight": 0.15,
    "content_weight": 0.2,
    "adain_alpha": 0.5,
    "slt_alpha": 0.2,
    "slt_tau": 0.02,
    "slt_iters": 10,
    "slt_grag": 0.7,
    "grag_shallow": 1.0,
    "grag_deep": 1.0,
    "disable_final_adain": False,
}
STYLE_GRAD_RMS_CALIBRATION_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsynth-root", default="/mnt/local/yzh/prog/DiffSynth-Studio")
    parser.add_argument("--main-model-root", default="/mnt/local/klein9b")
    parser.add_argument("--teacher-model-root", default="/mnt/local/klein9b-base")
    parser.add_argument(
        "--style-root",
        default="/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty",
    )
    parser.add_argument("--style-id", default="27")
    parser.add_argument("--style-image", default=None)
    parser.add_argument("--prompt-id", default="kp0")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--prompt-template",
        default=(
            "Create {content}. Use the visual style of the reference image, "
            "including its medium, palette, texture, and brushwork, but do not "
            "copy the reference objects or composition."
        ),
    )
    parser.add_argument("--content-prompt", default=None)
    parser.add_argument(
        "--content-prompt-template",
        default="Create {content}.",
    )
    parser.add_argument(
        "--without-prompt-mode",
        default="full_style_prompt",
        choices=["full_style_prompt", "content_only"],
    )
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument(
        "--reverse-step-strength",
        type=float,
        default=1.0,
        help=(
            "Scale the frozen scheduler displacement after each denoise prediction: "
            "z <- z + rho * (scheduler.step(...) - z). rho=1 preserves the original "
            "reverse process exactly; rho=0 suppresses its state update while retaining "
            "the scheduler call for matched accounting."
        ),
    )
    parser.add_argument("--embedded-guidance", type=float, default=0.0)
    parser.add_argument("--start-time", type=float, default=850.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--latent-optimizer",
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer for the inner latent updates; Adam remains available for matched ablations.",
    )
    parser.add_argument(
        "--latent-weight-decay",
        type=float,
        default=0.0,
        help="Decoupled AdamW weight decay for latent updates; keep at zero to isolate optimizer effects.",
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--disable-dynamic-iters", action="store_true")
    parser.add_argument(
        "--style-mode",
        default="score_match",
        choices=[
            "score_match", "ca_only", "ca_dmd", "ca_disentangled", "kv_match", "carrier_score", "score_carrier",
        ],
    )
    parser.add_argument("--fake-score-backend", default="none", choices=["none", "dmd2_guidance", "klein_full_fake"])
    parser.add_argument("--fake-model-root", default="/mnt/local/klein4b-base")
    parser.add_argument("--fake-vram-limit-gb", type=float, default=None)
    parser.add_argument("--fake-optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--fake-lr", type=float, default=1e-4)
    parser.add_argument("--fake-grad-clip-norm", type=float, default=None)
    parser.add_argument("--fake-updates-per-inner", type=int, default=1)
    parser.add_argument("--fake-train-timestep-mode", default="uniform", choices=["current", "uniform"])
    parser.add_argument("--fake-train-seed", type=int, default=1064)
    parser.add_argument("--fake-resume-state-path", default=None)
    parser.add_argument("--fake-save-state-path", default=None)
    parser.add_argument("--score-loss", default="mse", choices=["mse", "smooth_l1"])
    parser.add_argument("--score-normalization", default="coeff", choices=["raw", "coeff"])
    parser.add_argument("--sigma-eps", type=float, default=1e-4)
    parser.add_argument("--coeff-eps", type=float, default=1e-6)
    parser.add_argument("--style-weight", type=float, default=2.0)
    parser.add_argument("--log-gradient-components", action="store_true")
    parser.add_argument("--style-grad-rms-target", type=float, default=None)
    parser.add_argument("--dist-weight", type=float, default=0.15)
    parser.add_argument("--content-weight", type=float, default=0.2)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--lambda-ca", type=float, default=1.0)
    parser.add_argument("--lambda-dm", type=float, default=1.0)
    parser.add_argument("--dm-schedule", default="const", choices=["const", "front"],
                        help="ca_dmd DM weight schedule. 'front' = early-on/late-off "
                             "(lambda_dm*(1-progress)): DM stabilizes at high noise, off at "
                             "low noise where its naturalism-pull would kill texture.")
    parser.add_argument("--pseudo-step-size", type=float, default=0.05)
    parser.add_argument("--ca-renoise-tau-frac", type=float, default=0.0,
                        help="CDM L_CA (Rec1/3): if >0, evaluate the ca_only CA contrast at a "
                             "RE-NOISED timestep = current*frac. frac<1 => LOWER noise => texture "
                             "band. 0 disables (plain same-state ca_only).")
    parser.add_argument(
        "--ca-content-repel",
        type=float,
        default=0.0,
        help="ca_disentangled only: b in g_GST=(1-b)*v_cR-v_c0+b*v_cS (weight on d_C=v_cR-v_cS "
             "subtracted from d_R=v_cR-v_c0). b=0 reduces g_GST to d_R, identical to ca_only run "
             "with --without-prompt-mode content_only. Ignored when --ca-project is set.",
    )
    parser.add_argument(
        "--ca-project",
        action="store_true",
        help="ca_disentangled only: replace the content-repel blend with an orthogonal projection "
             "g_GST = d_R - (<d_R,d_C>/(||d_C||^2+eps))*d_C, removing the component of d_R along d_C.",
    )
    # Candidate D: multi-view style marginalization (ca_disentangled only). Averages the with-velocity
    # v_cR over K augmented reference views to marginalize out layout/composition and keep style, then
    # reuses the SAME disentangled combination (--ca-content-repel / --ca-project). K=1 is a byte-identical
    # no-op (view 0 = identity = today's single-view v_cR; no extra forwards).
    parser.add_argument(
        "--ca-marginalize-views",
        type=int,
        default=1,
        help="ca_disentangled only: K views to average v_cR over. K<=1 => no augmentation, no extra "
             "forwards, byte-identical to today. K>1 => view 0 is the identity reference plus (K-1) "
             "augmented crops; costs K with-forwards per inner iter.",
    )
    parser.add_argument(
        "--ca-marginalize-scale",
        type=float,
        default=0.85,
        help="ca_disentangled marginalization: random-crop scale lower bound; each augmented view crops "
             "a square of side s*min(H,W) with s ~ U(scale,1.0), then resizes back to the original size.",
    )
    parser.add_argument(
        "--ca-marginalize-flip",
        action="store_true",
        help="ca_disentangled marginalization: if set, roughly half the augmented views get a "
             "deterministic horizontal flip.",
    )
    parser.add_argument(
        "--ca-marginalize-seed",
        type=int,
        default=12345,
        help="ca_disentangled marginalization: seed for a LOCAL torch.Generator so the augmentation is "
             "deterministic/reproducible and never perturbs the global RNG (denoise noise draws).",
    )
    # Candidate E / G: layout-free reference velocity (ca_disentangled only). Re-runs the with-forward
    # with position-scrambled or position-stripped edit_image_ids (a PURE positional-id manipulation --
    # NEVER touches K/V or attention) so the reference contributes appearance/medium but not spatial
    # layout. --ca-layout-free none is a byte-identical no-op (v_cR_scram is not even computed).
    parser.add_argument(
        "--ca-layout-free",
        default="none",
        choices=["none", "scramble", "strip"],
        help="ca_disentangled only: compute a layout-free reference velocity v_cR_scram by re-running "
             "the with-forward with position-manipulated edit_image_ids. 'scramble' permutes the "
             "reference token positions (seeded, deterministic); 'strip' collapses all reference tokens "
             "to one constant position (bag-of-appearance). 'none' => today's behavior byte-for-byte.",
    )
    parser.add_argument(
        "--ca-layout-seed",
        type=int,
        default=777,
        help="Seed for the LOCAL torch.Generator used by --ca-layout-free scramble; deterministic and "
             "never perturbs the global RNG (denoise noise draws). Unused for strip/none.",
    )
    parser.add_argument(
        "--ca-layout-repel",
        type=float,
        default=0.0,
        help="ca_disentangled layout-free formula selector. 0.0 => Candidate E (direct): v_cR:=v_cR_scram "
             "in the disentangled builder, g=v_cR_scram-v_c0 (content_repel/project apply on top). b>0 => "
             "Candidate G (repel layout): g=d_R-b*d_layout where d_R=v_cR-v_c0, d_layout=v_cR-v_cR_scram.",
    )
    # rms=global (default), rms_token=per-token (lifts crushed subject signal), blend / blend0.3=mix, l2=global l2
    # --normalize-grad-mode is an alias (same dest) exposing build_pseudo_target_loss's normalize_grad_mode
    # on the CLI; passthrough (no re-validation) so "none"/"rms"/"rms_token"/"l2"/"blend"/"blend0.3" all reach it.
    parser.add_argument("--pseudo-grad-normalization", "--normalize-grad-mode", default="rms")
    parser.add_argument("--carrier-temperature", type=float, default=0.1)
    parser.add_argument("--carrier-strength", type=float, default=0.75)
    parser.add_argument("--carrier-highpass-gain", type=float, default=0.0)
    parser.add_argument("--carrier-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--carrier-schedule",
        default="const",
        choices=["const", "back", "front", "cosine", "sqrt"],
        help=(
            "Schedule multiplier on carrier strength + carrier loss weight over "
            "denoise progress (0=early/high-noise, 1=late/near-clean); 'const' "
            "is an exact no-op (multiplier=1.0) that reproduces the flat-carrier "
            "baseline byte-for-byte."
        ),
    )
    parser.add_argument("--adain-alpha", type=float, default=0.5)
    parser.add_argument("--disable-slt", action="store_true")
    parser.add_argument("--disable-final-adain", action="store_true")
    parser.add_argument("--slt-alpha", type=float, default=0.2)
    parser.add_argument("--slt-tau", type=float, default=0.02)
    parser.add_argument("--slt-iters", type=int, default=10)
    parser.add_argument("--slt-grag", type=float, default=0.7)
    parser.add_argument("--grag-shallow", type=float, default=1.0)
    parser.add_argument("--grag-deep", type=float, default=1.0)
    parser.add_argument("--kv-self-layer-start", type=int, default=12)
    parser.add_argument("--kv-self-layer-end", type=int, default=28)
    parser.add_argument("--kv-ot-tau-max", type=float, default=0.15)
    parser.add_argument("--kv-ot-tau-min", type=float, default=0.07)
    parser.add_argument("--kv-ot-iters", type=int, default=6)
    parser.add_argument("--kv-ot-layer-frac", type=float, default=0.5)
    parser.add_argument("--kv-sinkhorn-mode", default="freq", choices=["freq", "standard"])
    parser.add_argument("--kv-freq-hi-boost", type=float, default=2.0)
    parser.add_argument("--kv-grag-shallow", type=float, default=1.0)
    parser.add_argument("--kv-grag-deep", type=float, default=1.0)
    parser.add_argument("--main-vram-limit-gb", type=float, default=None)
    parser.add_argument("--teacher-vram-limit-gb", type=float, default=None)
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-prefix", default="tpami_with_style_teacher")
    parser.add_argument("--export-clean-latent-path", default=None)
    return parser.parse_args()


def install_modelscope_stub() -> None:
    if "modelscope" in sys.modules:
        return
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("modelscope is disabled; use local checkpoints")
    )
    sys.modules["modelscope"] = module


def sorted_files(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    detached = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(str(tuple(detached.shape)).encode("utf-8"))
    digest.update(detached.float().numpy().tobytes())
    return digest.hexdigest()


def manifest_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        record = f"{path.relative_to(root)}:{stat.st_size}\n".encode("utf-8")
        digest.update(record)
    return digest.hexdigest()


def parse_prompt_text(args: argparse.Namespace) -> str:
    return build_prompt_text(
        prompt=args.prompt,
        prompt_id=args.prompt_id,
        prompt_template=args.prompt_template,
        prompt_lookup=PROMPTS,
    )


def parse_content_prompt_text(args: argparse.Namespace) -> str:
    return build_content_prompt_text(
        content_prompt=args.content_prompt,
        prompt=args.prompt,
        prompt_id=args.prompt_id,
        content_prompt_template=args.content_prompt_template,
        prompt_lookup=PROMPTS,
    )


def resolve_style_path(args: argparse.Namespace) -> Path:
    if args.style_image:
        return Path(args.style_image)
    style_root = Path(args.style_root)
    style_id = str(args.style_id)
    direct_path = style_root / f"{style_id}.png"
    if direct_path.exists() or not style_id.isdigit():
        return direct_path
    zero_padded_path = style_root / f"{int(style_id):02d}.png"
    if zero_padded_path.exists():
        return zero_padded_path
    return direct_path


def build_marginalize_edit_conditions(
    pipe,
    style_image: Image.Image,
    *,
    views: int,
    scale: float,
    flip: bool,
    seed: int,
    auto_resize: bool = False,
) -> list[EditCondition]:
    """Candidate D: encode ``views - 1`` augmented reference edit conditions for multi-view style
    marginalization.

    View 0 (the identity / un-augmented reference) is handled by the caller's existing single-view
    ``edit_condition``, so this returns ONLY the augmented views. Each augmented view is a random
    square crop (side = round(s * min(H, W)), s ~ U(scale, 1.0)) at a random position, resized back
    to the original size, with an optional per-view horizontal flip. Crops preserve the reference
    texture/palette/medium (the style) while moving/removing specific content/composition, so
    averaging their with-velocities marginalizes out layout and keeps style.

    Augmentation is deterministic given ``seed`` via a LOCAL ``torch.Generator`` — it never touches
    the global RNG, so the denoise noise draws stay byte-identical to a run without marginalization.
    Returns an empty list when ``views <= 1`` (byte-identical no-op; the local generator is not even
    created). The flip decision is drawn unconditionally so crop positions are stable whether or not
    ``flip`` is enabled.
    """

    if int(views) <= 1:
        return []
    width, height = style_image.size
    generator = torch.Generator().manual_seed(int(seed))
    conditions: list[EditCondition] = []
    for _ in range(int(views) - 1):
        s = float(scale) + (1.0 - float(scale)) * float(torch.rand(1, generator=generator).item())
        side = max(1, int(round(s * min(int(width), int(height)))))
        x0 = int(torch.randint(0, int(width) - side + 1, (1,), generator=generator).item())
        y0 = int(torch.randint(0, int(height) - side + 1, (1,), generator=generator).item())
        do_flip = bool(torch.rand(1, generator=generator).item() < 0.5)
        view = style_image.crop((x0, y0, x0 + side, y0 + side)).resize(
            (int(width), int(height)), Image.Resampling.BICUBIC
        )
        if flip and do_flip:
            view = view.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        conditions.append(prepare_edit_condition(pipe, view, auto_resize=auto_resize))
    return conditions


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_flux2_pipe(model_root: Path, *, vram_limit_gb: float | None):
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    offload_kwargs = model_config_offload_kwargs(vram_limit_gb)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                path=sorted_files(str(model_root / "text_encoder" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(
                path=sorted_files(str(model_root / "transformer" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(
                path=str(model_root / "vae" / "diffusion_pytorch_model.safetensors"),
                **offload_kwargs,
            ),
        ],
        tokenizer_config=ModelConfig(path=str(model_root / "tokenizer")),
        vram_limit=vram_limit_gb,
    )
    pipe.vram_management_enabled = vram_limit_gb is not None
    return pipe


def encode_prompt_for_pipe(pipe, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from diffsynth.pipelines.flux2_image import Flux2Unit_PromptEmbedder, Flux2Unit_Qwen3PromptEmbedder

    if pipe.text_encoder_qwen3 is not None:
        pipe.load_models_to_device(["text_encoder_qwen3"])
        prompt_unit = Flux2Unit_Qwen3PromptEmbedder()
        with torch.no_grad():
            prompt_embeds, text_ids = prompt_unit.encode_prompt(
                pipe.text_encoder_qwen3,
                pipe.tokenizer,
                prompt_text,
                dtype=pipe.torch_dtype,
                device="cuda",
            )
            null_embeds, null_text_ids = prompt_unit.encode_prompt(
                pipe.text_encoder_qwen3,
                pipe.tokenizer,
                "",
                dtype=pipe.torch_dtype,
                device="cuda",
            )
    else:
        pipe.load_models_to_device(["text_encoder"])
        prompt_unit = Flux2Unit_PromptEmbedder()
        with torch.no_grad():
            prompt_embeds, text_ids = prompt_unit.encode_prompt(
                pipe.text_encoder,
                pipe.tokenizer,
                prompt_text,
                dtype=pipe.torch_dtype,
                device="cuda",
            )
            null_embeds, null_text_ids = prompt_unit.encode_prompt(
                pipe.text_encoder,
                pipe.tokenizer,
                "",
                dtype=pipe.torch_dtype,
                device="cuda",
            )
    prompt_embeds = freeze_condition_tensor(prompt_embeds, device="cuda", dtype=pipe.torch_dtype)
    text_ids = freeze_condition_tensor(text_ids, device="cuda")
    null_embeds = freeze_condition_tensor(null_embeds, device="cuda", dtype=pipe.torch_dtype)
    null_text_ids = freeze_condition_tensor(null_text_ids, device="cuda")
    return prompt_embeds, text_ids, null_embeds, null_text_ids


def encode_style_latent(pipe, style_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        style_pixels = pipe.preprocess_image(style_image)
        style_latent_4d = pipe.vae.encode(style_pixels)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C").to(device="cuda", dtype=torch.bfloat16)
    return style_latent, style_latent_4d


def unload_pipe_auxiliary_modules_to_cpu(pipe, *, include_vae: bool = True) -> list[str]:
    moved: list[str] = []
    for module_name in ("text_encoder_qwen3", "text_encoder", "vae"):
        if module_name == "vae" and not include_vae:
            continue
        module = getattr(pipe, module_name, None)
        if module is None:
            continue
        module.to("cpu")
        moved.append(module_name)
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames_for_rows(rows))
        writer.writeheader()
        writer.writerows(rows)


def latent_grag_compress(latents: torch.Tensor, delta: float) -> torch.Tensor:
    if delta >= 1.0:
        return latents
    mean = latents.mean(dim=1, keepdim=True)
    return mean + float(delta) * (latents - mean)


def smoke_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key, expected in FORMAL_DEFAULTS.items():
        value = getattr(args, key)
        if key == "disable_final_adain":
            value = bool(value)
        if value != expected:
            overrides[key] = value
    return overrides


def kv_match_sinkhorn_tau(
    *,
    outer_timestep_value: float,
    start_time: float,
    tau_max: float,
    tau_min: float,
) -> float:
    progress = 1.0 - (float(outer_timestep_value) / max(float(start_time), 1.0))
    progress = min(max(progress, 0.0), 1.0)
    return float(tau_max) - (float(tau_max) - float(tau_min)) * progress


def carrier_schedule_multiplier(progress: float, mode: str) -> float:
    """Schedule multiplier applied to carrier strength / carrier loss weight over
    denoise progress (0.0 = early / high-noise step, 1.0 = late / near-clean step).

    'const' is an exact no-op (always 1.0) so `--carrier-schedule const` stays
    byte-for-byte identical to the pre-existing flat-carrier behavior. The other
    modes average to ~1.0 over progress in [0, 1], so a scheduled run injects
    roughly the same total carrier "dose" as the flat run, just redistributed
    across steps (back/sqrt/cosine push style late; front is the reverse control).
    """
    if mode == "const":
        multiplier = 1.0
    elif mode == "back":
        multiplier = 0.25 + 1.5 * progress
    elif mode == "front":
        multiplier = 1.75 - 1.5 * progress
    elif mode == "sqrt":
        multiplier = 0.25 + 1.5 * (max(progress, 0.0) ** 0.5)
    elif mode == "cosine":
        multiplier = 0.25 + 1.5 * (1.0 - math.cos(math.pi * progress)) / 2.0
    else:
        raise ValueError(f"unknown carrier_schedule mode: {mode!r}")
    return max(multiplier, 0.0)


def optional_file_sha256(path: Path, *, enabled: bool) -> str | None:
    if not enabled:
        return None
    return file_sha256(path)


def optional_manifest_sha256(root: Path, *, enabled: bool) -> str | None:
    if not enabled:
        return None
    return manifest_sha256(root)


def same_state_contract_for_mode(style_mode: str, without_prompt_mode: str = "full_style_prompt") -> str:
    if style_mode == "kv_match":
        return (
            "For every optimized outer step, kv_match captures reference K/V once from the exact "
            "noisy style latent at timestep_style=i+1 with a null prompt, then reuses those cached "
            "targets for all inner Adam iterations on the same post-Step1 optimized latent state."
        )
    if without_prompt_mode == "content_only":
        return (
            "For every optimized outer step, all native-teacher branches use the exact same post-Step1 "
            "latent z_sigma at timestep_style=i+1 with the same timestep, model, and image_ids. "
            "The conditioning contrast is explicit: the positive branch keeps the full style prompt "
            "plus reference edit condition, while the negative/pre branch uses a content-only prompt "
            "with no reference edit condition."
        )
    return (
        "For every optimized outer step, all native-teacher branches use the exact post-Step1 "
        "latent z_sigma at timestep_style=i+1. Only the reference edit condition changes between "
        "with-reference and without-reference evaluations."
    )


def warm_start_note_for_mode(style_mode: str) -> str:
    if style_mode == "kv_match":
        return (
            "Warm-start remains the TPAMI latent-space AdaIN/SLT path only. The kv_match style "
            "objective separately captures reference K/V once per optimized outer step and does "
            "not reuse hook-based attention injection as a warm-start."
        )
    return (
        "Attention K/V GRAG is removed with the old style objective. Warm-start GRAG, when "
        "enabled, acts only through explicit latent compression before AdaIN/SLT."
    )


def without_shared_with_pre_for_mode(style_mode: str) -> bool | None:
    if style_mode == "kv_match":
        return None
    return True


def warm_start_latents(
    latents: torch.Tensor,
    *,
    style_latent_t: torch.Tensor,
    style_latent_clean: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    with torch.no_grad():
        warmed = latents
        warm_start_grag_delta = 0.5 * (float(args.grag_shallow) + float(args.grag_deep))
        style_latent_t_for_warm = latent_grag_compress(style_latent_t, warm_start_grag_delta)
        style_latent_clean_for_warm = latent_grag_compress(style_latent_clean, warm_start_grag_delta)
        if args.adain_alpha > 0:
            warmed = feature_adain(
                warmed.unsqueeze(1),
                style_latent_t_for_warm.unsqueeze(1),
                alpha=args.adain_alpha,
            ).squeeze(1)
        if not args.disable_slt and args.slt_alpha > 0:
            warmed = sinkhorn_latent_transport(
                warmed,
                style_latent_t_for_warm,
                style_latent_clean_for_warm,
                alpha=args.slt_alpha,
                n_iters=args.slt_iters,
                tau=args.slt_tau,
                grag_delta=args.slt_grag,
                mode="ot_adain",
            )
    return warmed.detach()


def compute_regularizer_losses(
    *,
    latents: torch.Tensor,
    style_latent_t: torch.Tensor,
    content_anchor: torch.Tensor,
    inner_idx: int,
    iters_this_step: int,
    loss_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    lat4d = latents.unsqueeze(1)
    sty4d = style_latent_t.unsqueeze(1).detach()
    dist_loss = (
        loss_fn(lat4d.mean(dim=2, keepdim=True), sty4d.mean(dim=2, keepdim=True))
        + loss_fn(lat4d.std(dim=2, keepdim=True), sty4d.std(dim=2, keepdim=True))
    )
    decay = 1.0 - (inner_idx / max(iters_this_step, 1)) * 0.5
    content_loss = loss_fn(latents, content_anchor) * decay
    return dist_loss, content_loss


def tensor_rms_value(tensor: torch.Tensor | None) -> float | None:
    if tensor is None:
        return None
    return float(tensor.detach().float().square().mean().sqrt().item())


def snapshot_kv_controller_state(controller: Klein9BController) -> dict[str, object]:
    return {
        "mode": controller.mode,
        "cur_self_layer": controller.cur_self_layer,
        "txt_len": controller.txt_len,
        "inject_ref_k": list(controller.inject_ref_k or []),
        "inject_ref_v": list(controller.inject_ref_v or []),
        "inject_beta": controller.inject_beta,
        "inject_idx": controller.inject_idx,
        "sinkhorn_tau": controller.sinkhorn_tau,
        "sinkhorn_iters": controller.sinkhorn_iters,
        "sinkhorn_only_last_n": controller.sinkhorn_only_last_n,
        "sinkhorn_use_pattern": controller.sinkhorn_use_pattern,
    }


def restore_kv_controller_state(controller: Klein9BController, snapshot: dict[str, object]) -> None:
    controller.mode = str(snapshot["mode"])
    controller.cur_self_layer = int(snapshot["cur_self_layer"])
    controller.txt_len = int(snapshot["txt_len"])
    controller.inject_ref_k = list(snapshot["inject_ref_k"])  # type: ignore[arg-type]
    controller.inject_ref_v = list(snapshot["inject_ref_v"])  # type: ignore[arg-type]
    controller.inject_beta = float(snapshot["inject_beta"])
    controller.inject_idx = int(snapshot["inject_idx"])
    controller.sinkhorn_tau = float(snapshot["sinkhorn_tau"])
    controller.sinkhorn_iters = int(snapshot["sinkhorn_iters"])
    controller.sinkhorn_only_last_n = int(snapshot["sinkhorn_only_last_n"])
    controller.sinkhorn_use_pattern = bool(snapshot["sinkhorn_use_pattern"])


def snapshot_kv_cache_state(cache: Klein9BCache) -> dict[str, list[torch.Tensor]]:
    q_list, k_list, v_list, out_list = cache.get()
    return {
        "q": list(q_list),
        "k": list(k_list),
        "v": list(v_list),
        "out": list(out_list),
    }


def restore_kv_cache_state(cache: Klein9BCache, snapshot: dict[str, list[torch.Tensor]]) -> None:
    cache.q = list(snapshot["q"])
    cache.k = list(snapshot["k"])
    cache.v = list(snapshot["v"])
    cache.out = list(snapshot["out"])


@contextmanager
def preserved_probe_runtime_state(
    *,
    kv_cache: Klein9BCache | None,
    kv_controller: Klein9BController | None,
):
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    kv_cache_snapshot = snapshot_kv_cache_state(kv_cache) if kv_cache is not None else None
    kv_controller_snapshot = snapshot_kv_controller_state(kv_controller) if kv_controller is not None else None
    if kv_cache is not None:
        kv_cache.clear()
    if kv_controller is not None:
        kv_controller.reset_mode()
        kv_controller.step()
        kv_controller.txt_len = 0
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if kv_cache is not None and kv_cache_snapshot is not None:
            restore_kv_cache_state(kv_cache, kv_cache_snapshot)
        if kv_controller is not None and kv_controller_snapshot is not None:
            restore_kv_controller_state(kv_controller, kv_controller_snapshot)


def build_gradient_audit_probe(
    *,
    teacher_dit,
    latents: torch.Tensor,
    timestep_style_cuda: torch.Tensor,
    sigma_info,
    teacher_prompt_embeds: torch.Tensor,
    teacher_text_ids: torch.Tensor,
    teacher_without_prompt_embeds: torch.Tensor,
    teacher_without_text_ids: torch.Tensor,
    teacher_image_ids: torch.Tensor,
    edit_condition,
    style_latent_t: torch.Tensor,
    content_anchor: torch.Tensor,
    args: argparse.Namespace,
    inner_idx: int,
    iters_this_step: int,
    loss_fn,
    carrier_schedule_mult: float = 1.0,
    triplet=None,
    v_fake: torch.Tensor | None = None,
    kv_cache: Klein9BCache | None = None,
    kv_controller: Klein9BController | None = None,
    kv_ref_k_list: list[torch.Tensor] | None = None,
    kv_ref_v_list: list[torch.Tensor] | None = None,
    kv_null_prompt_embeds: torch.Tensor | None = None,
    kv_null_text_ids: torch.Tensor | None = None,
    kv_image_ids: torch.Tensor | None = None,
    kv_sinkhorn_tau: float | None = None,
    kv_model_timestep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with preserved_probe_runtime_state(kv_cache=kv_cache, kv_controller=kv_controller):
        probe_latents = latents.detach().clone().requires_grad_(True)
        if args.style_mode == "score_match":
            probe_triplet = evaluate_native_edit_triplet(
                dit=teacher_dit,
                latents=probe_latents,
                timestep=timestep_style_cuda,
                embedded_guidance=0.0,
                prompt_embeds=teacher_prompt_embeds,
                text_ids=teacher_text_ids,
                negative_prompt_embeds=teacher_without_prompt_embeds,
                negative_text_ids=teacher_without_text_ids,
                image_ids=teacher_image_ids,
                edit_condition=edit_condition,
                require_pre_grad=True,
                share_without_with_pre=True,
            )
            probe_style_loss = score_match_loss_from_velocities(
                probe_triplet.pre_velocity,
                probe_triplet.with_velocity,
                sigma_info.sigma,
                normalization=args.score_normalization,
                loss_kind=args.score_loss,
                sigma_eps=args.sigma_eps,
                coeff_eps=args.coeff_eps,
            ).loss
        elif args.style_mode == "score_carrier":
            probe_triplet = evaluate_native_edit_triplet(
                dit=teacher_dit,
                latents=probe_latents,
                timestep=timestep_style_cuda,
                embedded_guidance=0.0,
                prompt_embeds=teacher_prompt_embeds,
                text_ids=teacher_text_ids,
                negative_prompt_embeds=teacher_without_prompt_embeds,
                negative_text_ids=teacher_without_text_ids,
                image_ids=teacher_image_ids,
                edit_condition=edit_condition,
                require_pre_grad=True,
                share_without_with_pre=True,
            )
            score_objective = score_match_loss_from_velocities(
                probe_triplet.pre_velocity,
                probe_triplet.with_velocity,
                sigma_info.sigma,
                normalization=args.score_normalization,
                loss_kind=args.score_loss,
                sigma_eps=args.sigma_eps,
                coeff_eps=args.coeff_eps,
            )
            goal_score = -score_objective.delta_score.detach()
            carrier_score, _ = content_affinity_transport(
                goal_score,
                content_anchor,
                temperature=args.carrier_temperature,
                strength=float(args.carrier_strength) * carrier_schedule_mult,
                highpass_gain=args.carrier_highpass_gain,
            )
            carrier_pseudo_grad = float(args.lambda_ca) * (float(args.cfg_scale) - 1.0) * carrier_score
            carrier_loss = build_pseudo_target_loss(
                velocity_to_x0(probe_latents, probe_triplet.pre_velocity, sigma_info.sigma),
                carrier_pseudo_grad,
                step_size=args.pseudo_step_size,
                normalize_grad_mode=args.pseudo_grad_normalization,
                grad_eps=args.coeff_eps,
            )[0]
            probe_style_loss = score_objective.loss + float(args.carrier_loss_weight) * carrier_schedule_mult * carrier_loss
        elif args.style_mode == "kv_match":
            if (
                kv_cache is None
                or kv_controller is None
                or kv_ref_k_list is None
                or kv_ref_v_list is None
                or kv_null_prompt_embeds is None
                or kv_null_text_ids is None
                or kv_image_ids is None
                or kv_sinkhorn_tau is None
                or kv_model_timestep is None
            ):
                raise RuntimeError("kv_match gradient audit probe requires isolated cache/controller state")
            with torch.enable_grad():
                _ = teacher_dit(
                    hidden_states=probe_latents,
                    timestep=kv_model_timestep,
                    guidance=None,
                    encoder_hidden_states=kv_null_prompt_embeds,
                    txt_ids=kv_null_text_ids,
                    img_ids=kv_image_ids,
                )
            q_list, _, _, self_out_list = kv_cache.get()
            probe_style_loss = kv_match_loss_from_attention_layers(
                self_out_list=self_out_list,
                q_list=q_list,
                ref_k_list=kv_ref_k_list,
                ref_v_list=kv_ref_v_list,
                sinkhorn_tau=kv_sinkhorn_tau,
                sinkhorn_iters=args.kv_ot_iters,
                ot_layer_frac=args.kv_ot_layer_frac,
                sinkhorn_mode=args.kv_sinkhorn_mode,
                freq_hi_boost=args.kv_freq_hi_boost,
            ).loss
        elif args.style_mode == "ca_only":
            if triplet is None:
                raise RuntimeError("ca_only gradient audit probe requires a detached teacher triplet")
            sigma_broadcast = sigma_info.sigma.to(device=probe_latents.device, dtype=probe_latents.dtype)
            probe_style_loss = pseudo_guidance_from_velocities(
                "ca_only",
                x0_pred=probe_latents - sigma_broadcast * triplet.without_velocity.detach(),
                v_with_detached=triplet.with_velocity.detach(),
                v_without_detached=triplet.without_velocity.detach(),
                sigma=sigma_info.sigma,
                cfg_scale=args.cfg_scale,
                lambda_ca=args.lambda_ca,
                lambda_dm=args.lambda_dm,
                pseudo_step_size=args.pseudo_step_size,
                sigma_eps=args.sigma_eps,
                normalize_grad_mode=args.pseudo_grad_normalization,
            ).loss
        elif args.style_mode == "carrier_score":
            if triplet is None:
                raise RuntimeError("carrier_score gradient audit probe requires a detached teacher triplet")
            sigma_broadcast = sigma_info.sigma.to(device=probe_latents.device, dtype=probe_latents.dtype)
            delta_ca_velocity = triplet.with_velocity.detach() - triplet.without_velocity.detach()
            delta_ca_score, _ = velocity_delta_to_score_delta(
                delta_ca_velocity,
                sigma_info.sigma,
                sigma_eps=args.sigma_eps,
            )
            carrier_score, _ = content_affinity_transport(
                delta_ca_score,
                content_anchor,
                temperature=args.carrier_temperature,
                strength=float(args.carrier_strength) * carrier_schedule_mult,
                highpass_gain=args.carrier_highpass_gain,
            )
            pseudo_grad_score = float(args.lambda_ca) * (float(args.cfg_scale) - 1.0) * carrier_score
            probe_style_loss = build_pseudo_target_loss(
                probe_latents - sigma_broadcast * triplet.without_velocity.detach(),
                pseudo_grad_score,
                step_size=args.pseudo_step_size,
                normalize_grad_mode=args.pseudo_grad_normalization,
                grad_eps=args.coeff_eps,
            )[0]
        elif args.style_mode == "ca_disentangled":
            raise NotImplementedError(
                "Gradient-audit probe measurement is not wired for ca_disentangled yet; "
                "rerun without --log-gradient-components/--style-grad-rms-target."
            )
        else:
            if triplet is None or v_fake is None:
                raise RuntimeError("ca_dmd gradient audit probe requires detached teacher/fake velocities")
            sigma_broadcast = sigma_info.sigma.to(device=probe_latents.device, dtype=probe_latents.dtype)
            probe_style_loss = pseudo_guidance_from_velocities(
                "ca_dmd",
                x0_pred=probe_latents - sigma_broadcast * triplet.without_velocity.detach(),
                v_with_detached=triplet.with_velocity.detach(),
                v_without_detached=triplet.without_velocity.detach(),
                v_fake_detached=v_fake.detach(),
                sigma=sigma_info.sigma,
                cfg_scale=args.cfg_scale,
                lambda_ca=args.lambda_ca,
                lambda_dm=args.lambda_dm,
                pseudo_step_size=args.pseudo_step_size,
                sigma_eps=args.sigma_eps,
                normalize_grad_mode=args.pseudo_grad_normalization,
            ).loss
        probe_dist_loss, probe_content_loss = compute_regularizer_losses(
            latents=probe_latents,
            style_latent_t=style_latent_t,
            content_anchor=content_anchor,
            inner_idx=inner_idx,
            iters_this_step=iters_this_step,
            loss_fn=loss_fn,
        )
    return probe_latents, probe_style_loss, probe_dist_loss, probe_content_loss


def inner_optimize(
    *,
    teacher_dit,
    latents: torch.Tensor,
    outer_step_idx: int,
    timestep_style: torch.Tensor,
    outer_timestep_value: float,
    sigma_info,
    teacher_prompt_embeds: torch.Tensor,
    teacher_text_ids: torch.Tensor,
    teacher_without_prompt_embeds: torch.Tensor,
    teacher_without_text_ids: torch.Tensor,
    teacher_image_ids: torch.Tensor,
    edit_condition,
    style_latent_t: torch.Tensor,
    content_anchor: torch.Tensor,
    args: argparse.Namespace,
    fake_backend,
    style_latent_clean: torch.Tensor | None = None,
    vgg_decode=None,
    vgg_targets=None,
    cond_for_backend: dict[str, object],
    gradient_calibration_state: StyleGradientCalibrationState,
    gradient_audit_requested: bool,
    kv_cache: Klein9BCache | None = None,
    kv_controller: Klein9BController | None = None,
    kv_ref_k_list: list[torch.Tensor] | None = None,
    kv_ref_v_list: list[torch.Tensor] | None = None,
    kv_null_prompt_embeds: torch.Tensor | None = None,
    kv_null_text_ids: torch.Tensor | None = None,
    kv_image_ids: torch.Tensor | None = None,
    kv_sinkhorn_tau: float | None = None,
    kv_model_timestep: torch.Tensor | None = None,
    scheduler=None,
    marginalize_edit_conditions: list[EditCondition] | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    loss_fn = torch.nn.L1Loss()
    marginalize_edit_conditions = marginalize_edit_conditions or []
    latents = latents.detach().requires_grad_(True)
    if args.latent_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [latents],
            lr=args.lr,
            weight_decay=args.latent_weight_decay,
        )
    else:
        optimizer = torch.optim.Adam([latents], lr=args.lr)
    step_logs: list[dict[str, object]] = []

    timestep_style_cuda = timestep_style.to(device=latents.device, dtype=latents.dtype)
    # progress is needed unconditionally for the carrier schedule (below), even
    # when --disable-dynamic-iters skips using it for iters_this_step.
    progress = 1.0 - (float(outer_timestep_value) / max(float(args.start_time), 1.0))
    carrier_schedule_mult = carrier_schedule_multiplier(progress, args.carrier_schedule)
    if args.disable_dynamic_iters:
        iters_this_step = int(args.iters)
    else:
        iters_this_step = max(1, int(args.iters * (1.0 - 0.6 * progress)))

    for inner_idx in range(iters_this_step):
        optimizer.zero_grad()
        extra: dict[str, float] = {}
        triplet = None
        v_fake = None
        if args.style_mode == "kv_match":
            if (
                kv_cache is None
                or kv_controller is None
                or kv_ref_k_list is None
                or kv_ref_v_list is None
                or kv_null_prompt_embeds is None
                or kv_null_text_ids is None
                or kv_image_ids is None
                or kv_sinkhorn_tau is None
                or kv_model_timestep is None
            ):
                raise RuntimeError("kv_match requires cached ref K/V, null prompt tensors, image ids, and sinkhorn tau")
            kv_cache.clear()
            kv_controller.step()
            with torch.enable_grad():
                _ = teacher_dit(
                    hidden_states=latents,
                    timestep=kv_model_timestep,
                    guidance=None,
                    encoder_hidden_states=kv_null_prompt_embeds,
                    txt_ids=kv_null_text_ids,
                    img_ids=kv_image_ids,
                )
            q_list, _, _, self_out_list = kv_cache.get()
            objective = kv_match_loss_from_attention_layers(
                self_out_list=self_out_list,
                q_list=q_list,
                ref_k_list=kv_ref_k_list,
                ref_v_list=kv_ref_v_list,
                sinkhorn_tau=kv_sinkhorn_tau,
                sinkhorn_iters=args.kv_ot_iters,
                ot_layer_frac=args.kv_ot_layer_frac,
                sinkhorn_mode=args.kv_sinkhorn_mode,
                freq_hi_boost=args.kv_freq_hi_boost,
            )
            kv_cache.clear()
            kv_controller.step()
            style_loss = objective.loss
            extra = {
                "kv_match_tau": float(objective.tau),
                "kv_match_layer_count": float(objective.layer_count),
                "kv_match_ot_layer_count": float(objective.ot_layer_count),
                "kv_match_target_norm": float(objective.target_norm.item()),
                "kv_match_self_out_norm": float(objective.self_out_norm.item()),
            }
        else:
            require_pre_grad = args.style_mode in {"score_match", "score_carrier"}
            triplet = evaluate_native_edit_triplet(
                dit=teacher_dit,
                latents=latents,
                timestep=timestep_style_cuda,
                embedded_guidance=0.0,
                prompt_embeds=teacher_prompt_embeds,
                text_ids=teacher_text_ids,
                negative_prompt_embeds=teacher_without_prompt_embeds,
                negative_text_ids=teacher_without_text_ids,
                image_ids=teacher_image_ids,
                edit_condition=edit_condition,
                require_pre_grad=require_pre_grad,
                share_without_with_pre=True,
            )
            _xdump = os.environ.get("DUMP_XHAT_STEPS", "")
            if _xdump and inner_idx == 0 and int(outer_step_idx) in {int(s) for s in _xdump.split(",") if s.strip().lstrip("-").isdigit()}:
                _dd = "agent_runs/xhat_dump"
                os.makedirs(_dd, exist_ok=True)
                _sig = sigma_info.sigma
                _sig = float(_sig.item()) if torch.is_tensor(_sig) else float(_sig)
                torch.save(
                    {
                        "x0s": velocity_to_x0(latents, triplet.with_velocity, sigma_info.sigma).detach().float().cpu(),
                        "x00": velocity_to_x0(latents, triplet.pre_velocity, sigma_info.sigma).detach().float().cpu(),
                        "outer": int(outer_step_idx),
                        "sigma": _sig,
                    },
                    f"{_dd}/{args.out_prefix}_style{args.style_id}_{args.prompt_id}_o{int(outer_step_idx):02d}.pt",
                )
                print(f"[xhat-dump] outer={int(outer_step_idx)} sigma={_sig:.4f}", flush=True)
            # env-gated: dump the raw CA signal g=v_with-v_without at EVERY inner iter for
            # selected outer steps, to measure whether the subject-region signal is real
            # (coherent across iters) or noise, and how the global rms-norm crushes it.
            _gdump = os.environ.get("DUMP_G_STEPS", "")
            if _gdump and triplet is not None and int(outer_step_idx) in {int(s) for s in _gdump.split(",") if s.strip().lstrip("-").isdigit()}:
                _gd = "agent_runs/g_dump"
                os.makedirs(_gd, exist_ok=True)
                torch.save(
                    {
                        "g": (triplet.with_velocity - triplet.without_velocity).detach().float().cpu(),
                        "outer": int(outer_step_idx),
                        "inner": int(inner_idx),
                    },
                    f"{_gd}/{args.out_prefix}_style{args.style_id}_{args.prompt_id}_o{int(outer_step_idx):02d}_i{int(inner_idx):02d}.pt",
                )
            if args.style_mode == "score_match":
                x0_pred = velocity_to_x0(latents, triplet.pre_velocity, sigma_info.sigma)
                objective = score_match_loss_from_velocities(
                    triplet.pre_velocity,
                    triplet.with_velocity,
                    sigma_info.sigma,
                    normalization=args.score_normalization,
                    loss_kind=args.score_loss,
                    sigma_eps=args.sigma_eps,
                    coeff_eps=args.coeff_eps,
                )
                style_loss = objective.loss
                extra = {
                    "score_match_delta_score_norm": float(objective.delta_score.float().norm().item()),
                    "score_match_delta_velocity_norm": float(objective.delta_velocity.float().norm().item()),
                }
            elif args.style_mode == "score_carrier":
                x0_pred = velocity_to_x0(latents, triplet.pre_velocity, sigma_info.sigma)
                objective = score_match_loss_from_velocities(
                    triplet.pre_velocity,
                    triplet.with_velocity,
                    sigma_info.sigma,
                    normalization=args.score_normalization,
                    loss_kind=args.score_loss,
                    sigma_eps=args.sigma_eps,
                    coeff_eps=args.coeff_eps,
                )
                goal_score = -objective.delta_score.detach()
                effective_carrier_strength = float(args.carrier_strength) * carrier_schedule_mult
                carrier_score, _ = content_affinity_transport(
                    goal_score,
                    content_anchor,
                    temperature=args.carrier_temperature,
                    strength=effective_carrier_strength,
                    highpass_gain=args.carrier_highpass_gain,
                )
                carrier_pseudo_grad = float(args.lambda_ca) * (float(args.cfg_scale) - 1.0) * carrier_score
                carrier_loss, _, _ = build_pseudo_target_loss(
                    x0_pred,
                    carrier_pseudo_grad,
                    step_size=args.pseudo_step_size,
                    normalize_grad_mode=args.pseudo_grad_normalization,
                    grad_eps=args.coeff_eps,
                )
                effective_carrier_loss_weight = float(args.carrier_loss_weight) * carrier_schedule_mult
                style_loss = objective.loss + effective_carrier_loss_weight * carrier_loss
                extra = {
                    "score_carrier_score_match_loss": float(objective.loss.item()),
                    "score_carrier_loss": float(carrier_loss.item()),
                    "score_carrier_delta_score_norm": float(objective.delta_score.float().norm().item()),
                    "score_carrier_transport_norm": float(carrier_score.float().norm().item()),
                    "score_carrier_transport_cosine": float(
                        affinity_transport_cosine(goal_score, carrier_score).mean().item()
                    ),
                    "score_carrier_loss_weight": float(args.carrier_loss_weight),
                    "score_carrier_temperature": float(args.carrier_temperature),
                    "score_carrier_strength": float(args.carrier_strength),
                    "score_carrier_highpass_gain": float(args.carrier_highpass_gain),
                    "carrier_schedule": args.carrier_schedule,
                    "carrier_schedule_multiplier": float(carrier_schedule_mult),
                    "score_carrier_effective_strength": effective_carrier_strength,
                    "score_carrier_effective_loss_weight": effective_carrier_loss_weight,
                }
            elif args.style_mode == "ca_only":
                sigma_broadcast = sigma_info.sigma.to(device=latents.device, dtype=latents.dtype)
                x0_pred = latents - sigma_broadcast * triplet.without_velocity.detach()
                _tau_frac = float(getattr(args, "ca_renoise_tau_frac", 0.0))
                if _tau_frac > 0.0 and scheduler is not None:
                    # CDM L_CA (Rec1/3): evaluate the CA contrast at a RE-NOISED LOW tau (<t => texture band)
                    x0_det = x0_pred.detach()
                    # reuse the STATE-CONSISTENT implied noise eps = z + (1-sigma)*v_without
                    # (survey correctness point: fresh random eps contaminates the contrast).
                    eps = (latents.detach() + (1.0 - sigma_broadcast) * triplet.without_velocity.detach())
                    tau_ts = timestep_style.float() * _tau_frac
                    sigma_tau_info = resolve_sigma_info(scheduler, tau_ts, ref_tensor=latents, sigma_eps=args.sigma_eps)
                    sig_t = sigma_tau_info.sigma.to(device=latents.device, dtype=latents.dtype)
                    z_tau = ((1.0 - sig_t) * x0_det + sig_t * eps).to(device=latents.device, dtype=latents.dtype)
                    tau_ts_cuda = tau_ts.to(device=latents.device, dtype=latents.dtype)
                    tau_trip = evaluate_native_edit_triplet(
                        dit=teacher_dit,
                        latents=z_tau,
                        timestep=tau_ts_cuda,
                        embedded_guidance=0.0,
                        prompt_embeds=teacher_prompt_embeds,
                        text_ids=teacher_text_ids,
                        negative_prompt_embeds=teacher_without_prompt_embeds,
                        negative_text_ids=teacher_without_text_ids,
                        image_ids=teacher_image_ids,
                        edit_condition=edit_condition,
                        require_pre_grad=False,
                        share_without_with_pre=True,
                    )
                    r_tau_v = tau_trip.with_velocity.detach() - tau_trip.without_velocity.detach()
                    r_tau_score, _ = velocity_delta_to_score_delta(r_tau_v, sigma_tau_info.sigma, sigma_eps=args.sigma_eps)
                    w_tau = float(1.0 / (r_tau_v.float().abs().mean().item() + args.coeff_eps))  # CDM inverse-L1
                    pseudo_grad = float(args.lambda_ca) * (float(args.cfg_scale) - 1.0) * w_tau * r_tau_score
                    style_loss, _, _ = build_pseudo_target_loss(
                        x0_pred,
                        pseudo_grad,
                        step_size=args.pseudo_step_size,
                        normalize_grad_mode=args.pseudo_grad_normalization,
                        grad_eps=args.coeff_eps,
                    )
                    extra = {
                        "ca_renoise_tau_frac": _tau_frac,
                        "ca_renoise_tau_ts": float(tau_ts.reshape(-1)[0].item()),
                        "ca_renoise_sigma": float(sigma_tau_info.sigma.item()),
                        "ca_renoise_r_l1": float(r_tau_v.float().abs().mean().item()),
                    }
                else:
                    objective = pseudo_guidance_from_velocities(
                        "ca_only",
                        x0_pred=x0_pred,
                        v_with_detached=triplet.with_velocity,
                        v_without_detached=triplet.without_velocity,
                        sigma=sigma_info.sigma,
                        cfg_scale=args.cfg_scale,
                        lambda_ca=args.lambda_ca,
                        lambda_dm=args.lambda_dm,
                        pseudo_step_size=args.pseudo_step_size,
                        sigma_eps=args.sigma_eps,
                        normalize_grad_mode=args.pseudo_grad_normalization,
                    )
                    style_loss = objective.loss
                    extra = {
                        "delta_ca_score_norm": float(objective.delta_ca_score.float().norm().item()),
                        "delta_ca_velocity_norm": float(objective.delta_ca_velocity.float().norm().item()),
                        "pseudo_grad_score_norm": float(objective.pseudo_grad_score.float().norm().item()),
                    }
            elif args.style_mode == "ca_disentangled":
                # Same-state disentangled CA: v_cR=triplet.with_velocity (full style + reference),
                # v_c0=triplet.without_velocity (content-only prompt, no reference, guaranteed by the
                # --without-prompt-mode=content_only contract enforced in main()), v_cS=full style
                # prompt with no reference, computed fresh at the identical (latents, timestep).
                sigma_broadcast = sigma_info.sigma.to(device=latents.device, dtype=latents.dtype)
                x0_pred = latents - sigma_broadcast * triplet.without_velocity.detach()
                _layout_free_mode = str(args.ca_layout_free)
                _layout_repel = float(args.ca_layout_repel)
                if _layout_free_mode != "none":
                    # === Candidate E / G: layout-free reference velocity ==================================
                    # Re-run the with-forward with position-manipulated edit_image_ids (a PURE positional-
                    # id manipulation -- the content edit_latents are reused byte-for-byte; NO K/V tensor,
                    # NO attention code is touched). This is one extra frozen no_grad forward per inner
                    # iter, exactly like a Candidate D augmented view. Marginalize is ignored here (they do
                    # not stack for now). The layout-free EditCondition is rebuilt each iter deterministically
                    # (seeded LOCAL generator => identical permutation every call; never the global RNG).
                    layout_free_edit_condition = build_layout_free_edit_condition(
                        edit_condition, mode=_layout_free_mode, seed=int(args.ca_layout_seed)
                    )
                    v_cR_scram = compute_native_velocity(
                        dit=teacher_dit,
                        latents=latents,
                        timestep=timestep_style_cuda,
                        embedded_guidance=0.0,
                        prompt_embeds=teacher_prompt_embeds,
                        text_ids=teacher_text_ids,
                        image_ids=teacher_image_ids,
                        edit_condition=layout_free_edit_condition,
                        no_grad=True,
                    ).detach()
                    # Evidence that the position manipulation actually bites: ||v_cR - v_cR_scram|| (the
                    # isolated layout component) must be clearly non-negligible vs ||v_cR - v_c0|| (d_R).
                    _v_cR = triplet.with_velocity.detach()
                    _v_c0 = triplet.without_velocity.detach()
                    d_layout_norm = float((_v_cR - v_cR_scram).float().norm().item())
                    d_R_norm = float((_v_cR - _v_c0).float().norm().item())
                    if _layout_repel > 0.0:
                        # Candidate G (repel layout): g = d_R - b*d_layout, using the ORIGINAL coherent-grid
                        # v_cR for d_R. Routed through the same score-conversion + build_pseudo_target_loss
                        # path as the other CA modes (v_cS is not needed here).
                        delta_ca_velocity = layout_repel_velocity_delta(
                            _v_cR,
                            _v_c0,
                            v_cR_scram,
                            layout_repel=_layout_repel,
                        )
                        delta_ca_score, _lf_score_coeff = velocity_delta_to_score_delta(
                            delta_ca_velocity,
                            sigma_info.sigma,
                            sigma_eps=args.sigma_eps,
                        )
                        pseudo_grad_score = (
                            float(args.lambda_ca) * (float(args.cfg_scale) - 1.0) * delta_ca_score
                        )
                        style_loss, _, _ = build_pseudo_target_loss(
                            x0_pred,
                            pseudo_grad_score,
                            step_size=args.pseudo_step_size,
                            normalize_grad_mode=args.pseudo_grad_normalization,
                            grad_eps=args.coeff_eps,
                        )
                        delta_ca_score_norm = float(delta_ca_score.float().norm().item())
                        delta_ca_velocity_norm = float(delta_ca_velocity.float().norm().item())
                        pseudo_grad_score_norm = float(pseudo_grad_score.float().norm().item())
                    else:
                        # Candidate E (direct): substitute v_cR := v_cR_scram in the disentangled builder,
                        # i.e. g = v_cR_scram - v_c0 (content_repel/project apply on top: d_C = v_cR_scram -
                        # v_cS). Reuses pseudo_guidance_from_velocities unchanged.
                        v_style_no_edit = compute_native_velocity(
                            dit=teacher_dit,
                            latents=latents,
                            timestep=timestep_style_cuda,
                            embedded_guidance=0.0,
                            prompt_embeds=teacher_prompt_embeds,
                            text_ids=teacher_text_ids,
                            image_ids=teacher_image_ids,
                            edit_condition=None,
                            no_grad=True,
                        ).detach()
                        objective = pseudo_guidance_from_velocities(
                            "ca_disentangled",
                            x0_pred=x0_pred,
                            v_with_detached=v_cR_scram,
                            v_without_detached=triplet.without_velocity,
                            v_style_detached=v_style_no_edit,
                            content_repel=args.ca_content_repel,
                            project=args.ca_project,
                            sigma=sigma_info.sigma,
                            cfg_scale=args.cfg_scale,
                            lambda_ca=args.lambda_ca,
                            lambda_dm=args.lambda_dm,
                            pseudo_step_size=args.pseudo_step_size,
                            sigma_eps=args.sigma_eps,
                            normalize_grad_mode=args.pseudo_grad_normalization,
                        )
                        style_loss = objective.loss
                        delta_ca_score_norm = float(objective.delta_ca_score.float().norm().item())
                        delta_ca_velocity_norm = float(objective.delta_ca_velocity.float().norm().item())
                        pseudo_grad_score_norm = float(objective.pseudo_grad_score.float().norm().item())
                    extra = {
                        "delta_ca_score_norm": delta_ca_score_norm,
                        "delta_ca_velocity_norm": delta_ca_velocity_norm,
                        "pseudo_grad_score_norm": pseudo_grad_score_norm,
                        "ca_content_repel": float(args.ca_content_repel),
                        "ca_project": bool(args.ca_project),
                        "ca_marginalize_views_requested": int(args.ca_marginalize_views),
                        "ca_marginalize_views_used": 1,
                        "ca_layout_free": _layout_free_mode,
                        "ca_layout_repel": _layout_repel,
                        "ca_layout_seed": int(args.ca_layout_seed),
                        "ca_layout_d_layout_norm": d_layout_norm,
                        "ca_layout_d_R_norm": d_R_norm,
                    }
                else:
                    v_style_no_edit = compute_native_velocity(
                        dit=teacher_dit,
                        latents=latents,
                        timestep=timestep_style_cuda,
                        embedded_guidance=0.0,
                        prompt_embeds=teacher_prompt_embeds,
                        text_ids=teacher_text_ids,
                        image_ids=teacher_image_ids,
                        edit_condition=None,
                        no_grad=True,
                    ).detach()
                    # Candidate D: marginalize v_cR over K reference views (view 0 = identity =
                    # triplet.with_velocity, plus each precomputed augmented edit condition run at the
                    # SAME latents/timestep/full-style-prompt as the identity with-forward). v_c0/v_cS
                    # stay single-view. When the list is empty (K<=1) this leaves v_cR_marginalized bound
                    # to the exact same triplet.with_velocity object => byte-identical to the single-view
                    # ca_disentangled anchor.
                    marginalize_views_used = len(marginalize_edit_conditions) + 1
                    if marginalize_edit_conditions:
                        with torch.no_grad():
                            v_cR_accum = triplet.with_velocity.detach().float()
                            for _aug_edit_condition in marginalize_edit_conditions:
                                v_cR_accum = v_cR_accum + compute_native_velocity(
                                    dit=teacher_dit,
                                    latents=latents,
                                    timestep=timestep_style_cuda,
                                    embedded_guidance=0.0,
                                    prompt_embeds=teacher_prompt_embeds,
                                    text_ids=teacher_text_ids,
                                    image_ids=teacher_image_ids,
                                    edit_condition=_aug_edit_condition,
                                    no_grad=True,
                                ).detach().float()
                            v_cR_marginalized = (
                                v_cR_accum / float(marginalize_views_used)
                            ).to(dtype=triplet.with_velocity.dtype)
                    else:
                        v_cR_marginalized = triplet.with_velocity
                    objective = pseudo_guidance_from_velocities(
                        "ca_disentangled",
                        x0_pred=x0_pred,
                        v_with_detached=v_cR_marginalized,
                        v_without_detached=triplet.without_velocity,
                        v_style_detached=v_style_no_edit,
                        content_repel=args.ca_content_repel,
                        project=args.ca_project,
                        sigma=sigma_info.sigma,
                        cfg_scale=args.cfg_scale,
                        lambda_ca=args.lambda_ca,
                        lambda_dm=args.lambda_dm,
                        pseudo_step_size=args.pseudo_step_size,
                        sigma_eps=args.sigma_eps,
                        normalize_grad_mode=args.pseudo_grad_normalization,
                    )
                    style_loss = objective.loss
                    extra = {
                        "delta_ca_score_norm": float(objective.delta_ca_score.float().norm().item()),
                        "delta_ca_velocity_norm": float(objective.delta_ca_velocity.float().norm().item()),
                        "pseudo_grad_score_norm": float(objective.pseudo_grad_score.float().norm().item()),
                        "ca_content_repel": float(args.ca_content_repel),
                        "ca_project": bool(args.ca_project),
                        "ca_marginalize_views_requested": int(args.ca_marginalize_views),
                        "ca_marginalize_views_used": int(marginalize_views_used),
                        "ca_layout_free": _layout_free_mode,
                        "ca_layout_repel": _layout_repel,
                        "ca_layout_seed": int(args.ca_layout_seed),
                    }
            elif args.style_mode == "carrier_score":
                sigma_broadcast = sigma_info.sigma.to(device=latents.device, dtype=latents.dtype)
                x0_pred = latents - sigma_broadcast * triplet.without_velocity.detach()
                delta_ca_velocity = triplet.with_velocity.detach() - triplet.without_velocity.detach()
                delta_ca_score, score_coeff = velocity_delta_to_score_delta(
                    delta_ca_velocity,
                    sigma_info.sigma,
                    sigma_eps=args.sigma_eps,
                )
                effective_carrier_strength = float(args.carrier_strength) * carrier_schedule_mult
                carrier_score, _carrier_affinity = content_affinity_transport(
                    delta_ca_score,
                    content_anchor,
                    temperature=args.carrier_temperature,
                    strength=effective_carrier_strength,
                    highpass_gain=args.carrier_highpass_gain,
                )
                pseudo_grad_score = float(args.lambda_ca) * (float(args.cfg_scale) - 1.0) * carrier_score
                style_loss, _, _ = build_pseudo_target_loss(
                    x0_pred,
                    pseudo_grad_score,
                    step_size=args.pseudo_step_size,
                    normalize_grad_mode=args.pseudo_grad_normalization,
                    grad_eps=args.coeff_eps,
                )
                extra = {
                    "carrier_score_delta_score_norm": float(delta_ca_score.float().norm().item()),
                    "carrier_score_transport_norm": float(carrier_score.float().norm().item()),
                    "carrier_score_transport_cosine": float(
                        affinity_transport_cosine(delta_ca_score, carrier_score).mean().item()
                    ),
                    "carrier_score_coeff": float(score_coeff.item()),
                    "carrier_score_pseudo_grad_norm": float(pseudo_grad_score.float().norm().item()),
                    "carrier_score_temperature": float(args.carrier_temperature),
                    "carrier_score_strength": float(args.carrier_strength),
                    "carrier_score_highpass_gain": float(args.carrier_highpass_gain),
                    "carrier_schedule": args.carrier_schedule,
                    "carrier_schedule_multiplier": float(carrier_schedule_mult),
                    "carrier_score_effective_strength": effective_carrier_strength,
                }
            else:
                sigma_broadcast = sigma_info.sigma.to(device=latents.device, dtype=latents.dtype)
                x0_pred = latents - sigma_broadcast * triplet.without_velocity.detach()
                if fake_backend is None:
                    raise RuntimeError("ca_dmd requires a real fake backend; got None after validation")
                backend_train_log = fake_backend.train_step(
                    fake_latents_clean=x0_pred.detach(),
                    z_sigma=latents.detach(),
                    timestep=timestep_style_cuda.detach(),
                    scoring_sigma=sigma_info.sigma.detach(),
                    cond=cond_for_backend,
                )
                v_fake = fake_backend.predict_velocity(
                    z_sigma=latents.detach(),
                    timestep=timestep_style_cuda.detach(),
                    scoring_sigma=sigma_info.sigma.detach(),
                    cond=cond_for_backend,
                )
                _eff_lambda_dm = float(args.lambda_dm)
                if getattr(args, "dm_schedule", "const") == "front":
                    _eff_lambda_dm = float(args.lambda_dm) * max(0.0, 1.0 - progress)  # DM early-on/late-off
                objective = pseudo_guidance_from_velocities(
                    "ca_dmd",
                    x0_pred=x0_pred,
                    v_with_detached=triplet.with_velocity,
                    v_without_detached=triplet.without_velocity,
                    v_fake_detached=v_fake,
                    sigma=sigma_info.sigma,
                    cfg_scale=args.cfg_scale,
                    lambda_ca=args.lambda_ca,
                    lambda_dm=_eff_lambda_dm,
                    pseudo_step_size=args.pseudo_step_size,
                    sigma_eps=args.sigma_eps,
                    normalize_grad_mode=args.pseudo_grad_normalization,
                )
                style_loss = objective.loss
                extra = {
                    "delta_ca_score_norm": float(objective.delta_ca_score.float().norm().item()),
                    "delta_ca_velocity_norm": float(objective.delta_ca_velocity.float().norm().item()),
                    "delta_ca_score_rms": tensor_rms_value(objective.delta_ca_score),
                    "delta_ca_velocity_rms": tensor_rms_value(objective.delta_ca_velocity),
                    "delta_dm_score_norm": float(objective.delta_dm_score.float().norm().item()) if objective.delta_dm_score is not None else 0.0,
                    "delta_dm_velocity_norm": float(objective.delta_dm_velocity.float().norm().item()) if objective.delta_dm_velocity is not None else 0.0,
                    "delta_dm_score_rms": tensor_rms_value(objective.delta_dm_score),
                    "delta_dm_velocity_rms": tensor_rms_value(objective.delta_dm_velocity),
                    "pseudo_grad_score_norm": float(objective.pseudo_grad_score.float().norm().item()),
                    "pseudo_grad_score_rms": tensor_rms_value(objective.pseudo_grad_score),
                    "ca_dm_score_cosine": grad_cosine(objective.delta_ca_score, objective.delta_dm_score),
                    "ca_dm_velocity_cosine": grad_cosine(objective.delta_ca_velocity, objective.delta_dm_velocity),
                    **backend_train_log,
                }

        dist_loss, content_loss = compute_regularizer_losses(
            latents=latents,
            style_latent_t=style_latent_t,
            content_anchor=content_anchor,
            inner_idx=inner_idx,
            iters_this_step=iters_this_step,
            loss_fn=loss_fn,
        )
        should_measure = gradient_audit_requested and not gradient_calibration_state.measured_once
        measurement_kwargs: dict[str, object] = {}
        if should_measure:
            (
                probe_latents,
                probe_style_loss,
                probe_dist_loss,
                probe_content_loss,
            ) = build_gradient_audit_probe(
                teacher_dit=teacher_dit,
                latents=latents,
                timestep_style_cuda=timestep_style_cuda,
                sigma_info=sigma_info,
                teacher_prompt_embeds=teacher_prompt_embeds,
                teacher_text_ids=teacher_text_ids,
                teacher_without_prompt_embeds=teacher_without_prompt_embeds,
                teacher_without_text_ids=teacher_without_text_ids,
                teacher_image_ids=teacher_image_ids,
                edit_condition=edit_condition,
                style_latent_t=style_latent_t,
                content_anchor=content_anchor,
                args=args,
                inner_idx=inner_idx,
                iters_this_step=iters_this_step,
                loss_fn=loss_fn,
                carrier_schedule_mult=carrier_schedule_mult,
                triplet=triplet,
                v_fake=v_fake,
                kv_cache=kv_cache if args.style_mode == "kv_match" else None,
                kv_controller=kv_controller if args.style_mode == "kv_match" else None,
                kv_ref_k_list=kv_ref_k_list,
                kv_ref_v_list=kv_ref_v_list,
                kv_null_prompt_embeds=kv_null_prompt_embeds,
                kv_null_text_ids=kv_null_text_ids,
                kv_image_ids=kv_image_ids,
                kv_sinkhorn_tau=kv_sinkhorn_tau,
                kv_model_timestep=kv_model_timestep,
            )
            measurement_kwargs = {
                "measurement_latents": probe_latents,
                "measurement_style_loss": probe_style_loss,
                "measurement_dist_loss": probe_dist_loss,
                "measurement_content_loss": probe_content_loss,
            }
        audit_result = apply_gradient_audit(
            style_mode=args.style_mode,
            latents=latents,
            style_loss=style_loss,
            dist_loss=dist_loss,
            content_loss=content_loss,
            **measurement_kwargs,
            style_weight=float(args.style_weight),
            dist_weight=float(args.dist_weight),
            content_weight=float(args.content_weight),
            calibration_state=gradient_calibration_state,
            should_measure=should_measure,
            outer_step=outer_step_idx,
            inner_iter=inner_idx,
        )
        total_loss = (
            audit_result.weighted_style_loss
            + audit_result.weighted_dist_loss
            + audit_result.weighted_content_loss
        )
        # env-gated texture-statistics loss (real style transfer, x0/latent only — never K/V).
        # STATS_LOSS in {gram,sw,meanstd,sw+gram}; STATS_WEIGHT scalar. Set --lambda-ca 0 to test
        # the stats signal in isolation. Reference = clean style latent (falls back to noised).
        _statskind = os.environ.get("STATS_LOSS", "")
        if _statskind and triplet is not None:
            _x0_stats = velocity_to_x0(latents, triplet.without_velocity.detach(), sigma_info.sigma)
            _stats_w = float(os.environ.get("STATS_WEIGHT", "1.0"))
            if _statskind == "vgg" and vgg_decode is not None and vgg_targets is not None:
                from src.style_stats import vgg_gram_loss
                _stats_loss = vgg_gram_loss(vgg_decode(_x0_stats), vgg_targets)
            else:
                from src.style_stats import style_stats_loss
                _ref_stats = style_latent_clean if style_latent_clean is not None else style_latent_t
                _stats_loss = style_stats_loss(_statskind, _x0_stats, _ref_stats,
                                               n_proj=int(os.environ.get("STATS_NPROJ", "128")))
            total_loss = total_loss + _stats_w * _stats_loss
            extra["stats_loss"] = float(_stats_loss.detach().item())
        total_loss.backward()
        optimizer.step()

        step_logs.append(
            {
                "inner_iter": inner_idx,
                "sigma": float(sigma_info.sigma.item()),
                "score_delta_coeff": float(sigma_info.score_delta_coefficient.item()),
                "iters_this_step": iters_this_step,
                "style_loss": float(style_loss.item()),
                "dist_loss": float(dist_loss.item()),
                "content_loss": float(content_loss.item()),
                "total_loss": float(total_loss.item()),
                **audit_result.log_row,
                **extra,
            }
        )

    return latents.detach(), step_logs


def main() -> int:
    process_started_at = time.perf_counter()
    args = parse_args()
    if not 0.0 <= float(args.reverse_step_strength) <= 1.0:
        raise ValueError("--reverse-step-strength must lie in [0, 1]")
    validate_style_mode(args.style_mode, args.fake_score_backend)
    fake_backend = None

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    if args.diffsynth_root not in sys.path:
        sys.path.insert(0, args.diffsynth_root)
    install_modelscope_stub()
    _maybe_install_sinkhorn_attention()  # env-gated probe; after stub so diffsynth imports; no-op unless $SINKHORN_ATTN truthy
    _maybe_install_vshuffle()             # env-gated probe; no-op unless $VSHUFFLE truthy (USER-authorized 2026-07-21)

    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from diffsynth.pipelines.flux2_image import Flux2Unit_ImageIDs

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results_dir = PROJECT_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = parse_prompt_text(args)
    content_prompt_text = (
        parse_content_prompt_text(args)
        if (args.without_prompt_mode == "content_only" or args.content_prompt is not None or args.prompt is None)
        else None
    )
    without_prompt_text = (
        content_prompt_text
        if args.without_prompt_mode == "content_only"
        else prompt_text
    )
    style_path = resolve_style_path(args)
    if not style_path.exists():
        raise FileNotFoundError(style_path)
    if args.style_mode == "kv_match" and args.without_prompt_mode != "full_style_prompt":
        raise ValueError("kv_match does not use a native without/pre teacher branch; keep --without-prompt-mode=full_style_prompt")
    if args.style_mode == "ca_disentangled" and args.without_prompt_mode != "content_only":
        raise ValueError(
            "ca_disentangled needs the content-only teacher branch for v_c0 "
            "(g_GST = (1-b)*v_cR - v_c0 + b*v_cS); set --without-prompt-mode=content_only"
        )

    main_model_root = Path(args.main_model_root)
    teacher_model_root = Path(args.teacher_model_root)
    fake_model_root = Path(args.fake_model_root)
    diagnostic_single_model = main_model_root.resolve() == teacher_model_root.resolve()
    vram_contract = resolve_dual_model_vram_contract(
        args.style_mode,
        legacy_vram_limit_gb=args.vram_limit_gb,
        main_vram_limit_gb=args.main_vram_limit_gb,
        teacher_vram_limit_gb=args.teacher_vram_limit_gb,
    )
    main_loading = model_loading_mode(vram_contract.effective_main_vram_limit_gb)
    teacher_loading = (
        model_loading_mode(vram_contract.effective_teacher_vram_limit_gb)
        if vram_contract.teacher_model_required
        else "not_loaded"
    )
    share_runtime_pipe = (
        vram_contract.teacher_model_required
        and
        diagnostic_single_model
        and vram_contract.effective_main_vram_limit_gb == vram_contract.effective_teacher_vram_limit_gb
        and args.style_mode not in {"score_match", "carrier_score", "score_carrier"}
    )
    main_pipe = load_flux2_pipe(main_model_root, vram_limit_gb=vram_contract.effective_main_vram_limit_gb)
    teacher_pipe = None
    if vram_contract.teacher_model_required:
        teacher_pipe = (
            main_pipe
            if share_runtime_pipe
            else load_flux2_pipe(teacher_model_root, vram_limit_gb=vram_contract.effective_teacher_vram_limit_gb)
        )
    for note in vram_contract.notes:
        print(f"[runtime] {note}", flush=True)
    print(
        "[startup] "
        f"main_model_root={main_model_root} requested_vram_limit_gb={vram_contract.requested_main_vram_limit_gb} "
        f"effective_vram_limit_gb={vram_contract.effective_main_vram_limit_gb} loading_mode={main_loading}",
        flush=True,
    )
    print(
        "[startup] "
        f"teacher_model_root={teacher_model_root} requested_vram_limit_gb={vram_contract.requested_teacher_vram_limit_gb} "
        f"effective_vram_limit_gb={vram_contract.effective_teacher_vram_limit_gb} loading_mode={teacher_loading} "
        f"teacher_requires_full_cuda={vram_contract.teacher_requires_full_cuda} "
        f"teacher_model_required={vram_contract.teacher_model_required} shared_runtime_pipe={share_runtime_pipe}",
        flush=True,
    )

    style_image = Image.open(style_path).convert("RGB").resize((args.width, args.height), Image.Resampling.BICUBIC)
    style_latent, style_latent_4d = encode_style_latent(main_pipe, style_image)
    main_prompt_embeds, main_text_ids, _main_null_embeds, _main_null_text_ids = encode_prompt_for_pipe(main_pipe, prompt_text)
    assert_static_condition_tensor("main_prompt_embeds", main_prompt_embeds)
    assert_static_condition_tensor("main_text_ids", main_text_ids)
    assert_static_condition_tensor("main_null_embeds", _main_null_embeds)
    assert_static_condition_tensor("main_null_text_ids", _main_null_text_ids)

    teacher_prompt_embeds = None
    teacher_text_ids = None
    teacher_without_prompt_embeds = None
    teacher_without_text_ids = None
    edit_condition = None
    marginalize_edit_conditions: list[EditCondition] = []
    if teacher_pipe is not None:
        edit_condition = prepare_edit_condition(teacher_pipe, style_image, auto_resize=bool(os.environ.get("EDIT_AUTO_RESIZE", "")))
        if args.style_mode == "ca_disentangled" and int(args.ca_marginalize_views) > 1:
            # Candidate D: encode the (K-1) augmented reference views ONCE (they do not depend on the
            # noised latents/timestep), then reuse across every outer/inner iter's with-forward.
            marginalize_edit_conditions = build_marginalize_edit_conditions(
                teacher_pipe,
                style_image,
                views=int(args.ca_marginalize_views),
                scale=float(args.ca_marginalize_scale),
                flip=bool(args.ca_marginalize_flip),
                seed=int(args.ca_marginalize_seed),
                auto_resize=bool(os.environ.get("EDIT_AUTO_RESIZE", "")),
            )
            for _mec in marginalize_edit_conditions:
                assert_static_condition_tensor("marginalize_edit_latents", _mec.edit_latents)
                assert_static_condition_tensor("marginalize_edit_image_ids", _mec.edit_image_ids)
            print(
                f"[ca-marginalize] views={args.ca_marginalize_views} scale={args.ca_marginalize_scale} "
                f"flip={bool(args.ca_marginalize_flip)} seed={args.ca_marginalize_seed} "
                f"-> {len(marginalize_edit_conditions)} augmented edit conditions "
                f"(+ identity view 0 = {len(marginalize_edit_conditions) + 1} total views averaged)",
                flush=True,
            )
        if args.style_mode == "ca_disentangled" and args.ca_layout_free != "none":
            _lf_formula = "G(repel-layout)" if float(args.ca_layout_repel) > 0.0 else "E(direct)"
            print(
                f"[ca-layout-free] ca_layout_free={args.ca_layout_free} "
                f"ca_layout_repel={float(args.ca_layout_repel)} ca_layout_seed={int(args.ca_layout_seed)} "
                f"formula={_lf_formula} (positional edit_image_ids manipulation ONLY; no K/V / attention)",
                flush=True,
            )
        teacher_prompt_embeds, teacher_text_ids, _teacher_null_embeds, _teacher_null_text_ids = encode_prompt_for_pipe(
            teacher_pipe,
            prompt_text,
        )
        if args.without_prompt_mode == "content_only":
            (
                teacher_without_prompt_embeds,
                teacher_without_text_ids,
                _teacher_without_null_embeds,
                _teacher_without_null_text_ids,
            ) = encode_prompt_for_pipe(
                teacher_pipe,
                without_prompt_text,
            )
            assert_static_condition_tensor("teacher_without_prompt_embeds", teacher_without_prompt_embeds)
            assert_static_condition_tensor("teacher_without_text_ids", teacher_without_text_ids)
            assert_static_condition_tensor("teacher_without_null_embeds", _teacher_without_null_embeds)
            assert_static_condition_tensor("teacher_without_null_text_ids", _teacher_without_null_text_ids)
        else:
            teacher_without_prompt_embeds = teacher_prompt_embeds
            teacher_without_text_ids = teacher_text_ids
        assert_static_condition_tensor("teacher_prompt_embeds", teacher_prompt_embeds)
        assert_static_condition_tensor("teacher_text_ids", teacher_text_ids)
        assert_static_condition_tensor("teacher_null_embeds", _teacher_null_embeds)
        assert_static_condition_tensor("teacher_null_text_ids", _teacher_null_text_ids)
        assert_static_condition_tensor("teacher_edit_latents", edit_condition.edit_latents)
        assert_static_condition_tensor("teacher_edit_image_ids", edit_condition.edit_image_ids)

    main_aux_modules_offloaded: list[str] = []
    if args.style_mode == "kv_match" and vram_contract.effective_main_vram_limit_gb is None:
        main_aux_modules_offloaded = unload_pipe_auxiliary_modules_to_cpu(main_pipe)
        if main_aux_modules_offloaded:
            print(
                "[main-memory] kept main DiT resident on CUDA and moved auxiliary modules to CPU: "
                + ",".join(main_aux_modules_offloaded),
                flush=True,
            )
    teacher_aux_modules_offloaded: list[str] = []
    if teacher_pipe is not None and vram_contract.effective_teacher_vram_limit_gb is None:
        teacher_aux_modules_offloaded = unload_pipe_auxiliary_modules_to_cpu(teacher_pipe)
        if teacher_aux_modules_offloaded:
            print(
                "[teacher-memory] kept teacher DiT resident on CUDA and moved auxiliary modules to CPU: "
                + ",".join(teacher_aux_modules_offloaded),
                flush=True,
            )

    lat_h = args.height // 16
    lat_w = args.width // 16
    image_id_unit = Flux2Unit_ImageIDs()
    main_image_ids = image_id_unit.prepare_latent_ids(lat_h, lat_w).to("cuda")
    teacher_image_ids = image_id_unit.prepare_latent_ids(lat_h, lat_w).to("cuda")
    style_image_ids = image_id_unit.prepare_latent_ids(lat_h, lat_w).to("cuda")
    fake_image_ids = image_id_unit.prepare_latent_ids(lat_h, lat_w).to("cuda")

    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(args.num_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps.clone()

    fake_pipe = None
    fake_dit = None
    fake_aux_modules_offloaded: list[str] = []
    fake_loading_mode = "not_loaded"
    if args.style_mode == "ca_dmd":
        if args.fake_score_backend == "klein_full_fake":
            fake_pipe = load_flux2_pipe(fake_model_root, vram_limit_gb=args.fake_vram_limit_gb)
            fake_prompt_embeds, fake_text_ids, _fake_null_embeds, _fake_null_text_ids = encode_prompt_for_pipe(
                fake_pipe,
                prompt_text,
            )
            fake_edit_condition = prepare_edit_condition(fake_pipe, style_image, auto_resize=bool(os.environ.get("EDIT_AUTO_RESIZE", "")))
            assert_static_condition_tensor("fake_prompt_embeds", fake_prompt_embeds)
            assert_static_condition_tensor("fake_text_ids", fake_text_ids)
            assert_static_condition_tensor("fake_null_embeds", _fake_null_embeds)
            assert_static_condition_tensor("fake_null_text_ids", _fake_null_text_ids)
            assert_static_condition_tensor("fake_edit_latents", fake_edit_condition.edit_latents)
            assert_static_condition_tensor("fake_edit_image_ids", fake_edit_condition.edit_image_ids)
            fake_aux_modules_offloaded = unload_pipe_auxiliary_modules_to_cpu(fake_pipe)
            if fake_aux_modules_offloaded:
                print(
                    "[fake-memory] kept fake DiT resident on CUDA and moved auxiliary modules to CPU: "
                    + ",".join(fake_aux_modules_offloaded),
                    flush=True,
                )
            fake_pipe.load_models_to_device(["dit"])
            fake_dit = fake_pipe.dit.to("cuda")
            fake_dit.requires_grad_(True)
            fake_dit.train()
            fake_loading_mode = "full_cuda_dit_aux_cpu" if fake_aux_modules_offloaded else "full_cuda"
            fake_backend = build_fake_score_backend(
                args.fake_score_backend,
                fake_dit=fake_dit,
                conditioning=FakeConditioning(
                    prompt_embeds=fake_prompt_embeds,
                    text_ids=fake_text_ids,
                    image_ids=fake_image_ids,
                    edit_condition=fake_edit_condition,
                ),
                scheduler_lookup=FakeSchedulerLookup(
                    timesteps=scheduler.timesteps.clone(),
                    sigmas=scheduler.sigmas.clone(),
                ),
                config=KleinFullFakeBackendConfig(
                    fake_model_root=str(fake_model_root),
                    optimizer=args.fake_optimizer,
                    lr=float(args.fake_lr),
                    grad_clip_norm=args.fake_grad_clip_norm,
                    updates_per_inner=int(args.fake_updates_per_inner),
                    train_timestep_mode=args.fake_train_timestep_mode,
                    train_seed=int(args.fake_train_seed),
                    init_checkpoint_root=str(fake_model_root),
                    resume_state_path=args.fake_resume_state_path,
                    save_state_path=args.fake_save_state_path,
                ),
            )
            print(
                "[startup] "
                f"fake_model_root={fake_model_root} requested_vram_limit_gb={args.fake_vram_limit_gb} "
                f"loading_mode={fake_loading_mode} backend={args.fake_score_backend} "
                f"optimizer={args.fake_optimizer} lr={args.fake_lr} updates_per_inner={args.fake_updates_per_inner} "
                f"train_timestep_mode={args.fake_train_timestep_mode} train_seed={args.fake_train_seed} "
                f"grad_clip_norm={args.fake_grad_clip_norm}",
                flush=True,
            )
        else:
            fake_backend = build_fake_score_backend(args.fake_score_backend)

    main_pipe.load_models_to_device(["dit"])
    main_dit = main_pipe.dit
    if vram_contract.effective_main_vram_limit_gb is None:
        main_dit = main_dit.to("cuda")
    main_dit.requires_grad_(False)
    teacher_dit = None
    if teacher_pipe is not None:
        teacher_pipe.load_models_to_device(["dit"])
        teacher_dit = teacher_pipe.dit
        if vram_contract.effective_teacher_vram_limit_gb is None:
            teacher_dit = teacher_dit.to("cuda")
        teacher_dit.requires_grad_(False)
    if args.style_mode != "kv_match" and (
        teacher_dit is None
        or teacher_prompt_embeds is None
        or teacher_text_ids is None
        or edit_condition is None
    ):
        raise RuntimeError(f"{args.style_mode} requires a loaded native teacher model and detached condition tensors")

    kv_cache = None
    kv_controller = None
    if args.style_mode == "kv_match":
        kv_cache = Klein9BCache()
        kv_controller = Klein9BController(
            self_layers=(args.kv_self_layer_start, args.kv_self_layer_end),
        )
        register_klein9b_attn_hooks(main_dit, kv_controller, kv_cache)
        print(
            "[kv-match] registered main DiT cache hooks "
            f"self_layers=({args.kv_self_layer_start},{args.kv_self_layer_end}) "
            f"sinkhorn_mode={args.kv_sinkhorn_mode}",
            flush=True,
        )

    dit_forward_counts = {"shared": 0, "main": 0, "teacher": 0}
    forward_hook_handles = []

    def _count_forward(key: str):
        def _hook(_module, _inputs, _output):
            dit_forward_counts[key] += 1
        return _hook

    if teacher_dit is main_dit:
        forward_hook_handles.append(main_dit.register_forward_hook(_count_forward("shared")))
    else:
        forward_hook_handles.append(main_dit.register_forward_hook(_count_forward("main")))
        if teacher_dit is not None:
            forward_hook_handles.append(teacher_dit.register_forward_hook(_count_forward("teacher")))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_started_at = time.perf_counter()

    noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    latents = rearrange(noise, "B C H W -> B (H W) C")

    row_logs: list[dict[str, object]] = []
    main_scheduler_calls = 0
    optimized_outer_steps = 0
    ref_capture_calls = 0
    gradient_audit_requested = bool(args.log_gradient_components or args.style_grad_rms_target is not None)
    gradient_calibration_state = StyleGradientCalibrationState(
        target_rms=args.style_grad_rms_target,
        calibration_eps=STYLE_GRAD_RMS_CALIBRATION_EPS,
        effective_style_weight=float(args.style_weight),
    )

    cond_for_backend = {
        "prompt_text": prompt_text,
        "style_path": str(style_path),
        "style_id": args.style_id,
        "prompt_id": args.prompt_id,
        "teacher_embedded_guidance_scalar": 0.0,
    }

    # VGG perceptual (Gatys) loss prep: keep VAE on GPU, precompute style-ref Gram targets from
    # the decoded style latent, and build a differentiable x0->[0,1] decode closure. Env STATS_LOSS=vgg.
    _vgg_decode = None
    _vgg_targets = None
    if os.environ.get("STATS_LOSS") == "vgg":
        from src.style_stats import vgg_gram_targets
        main_pipe.load_models_to_device(["vae"])
        try:
            main_pipe.vae = main_pipe.vae.to("cuda")
        except Exception:
            pass

        def _vgg_decode(_x0_bnc):
            _l4 = rearrange(_x0_bnc, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
            _px = main_pipe.vae.decode(_l4)
            return ((_px.float() + 1.0) / 2.0).clamp(0.0, 1.0)

        with torch.no_grad():
            _vgg_targets = vgg_gram_targets(_vgg_decode(style_latent))
        print(f"[VGG] perceptual loss active; {len(_vgg_targets)} gram targets", flush=True)

    pbar = tqdm(enumerate(timesteps), total=len(timesteps), desc=f"TPAMI {args.style_mode}")
    for outer_idx, timestep_cpu in pbar:
        timestep = timestep_cpu.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        main_guidance = None
        if args.embedded_guidance > 0:
            main_guidance = torch.tensor([args.embedded_guidance], device="cuda", dtype=torch.bfloat16)
        if args.style_mode == "kv_match":
            if kv_cache is None or kv_controller is None:
                raise RuntimeError("kv_match requires main DiT cache hooks to be initialized")
            kv_cache.clear()
            kv_controller.step()
        _styled_dg = os.environ.get("STYLED_DENOISE", "")
        _sd_start = int(os.environ.get("STYLED_DENOISE_START", "0"))  # only style the trajectory from this outer step (lets composition form first)
        if _styled_dg and edit_condition is not None and args.style_mode != "kv_match" and int(outer_idx) >= _sd_start:
            # style-guide the reverse trajectory itself: noise_pred = v_without + g*(v_with - v_without),
            # i.e. CFG on the native edit condition IN the denoise, so the trajectory heads toward a
            # styled image instead of re-photorealizing the subject each step (attacks keystone #2).
            from diffsynth.pipelines.flux2_image import model_fn_flux2
            _g_sd = float(_styled_dg)
            with torch.no_grad():
                _v_no = model_fn_flux2(dit=main_dit, latents=latents, timestep=timestep,
                                       embedded_guidance=0.0, prompt_embeds=main_prompt_embeds,
                                       text_ids=main_text_ids, image_ids=main_image_ids)
                _v_ed = model_fn_flux2(dit=main_dit, latents=latents, timestep=timestep,
                                       embedded_guidance=0.0, prompt_embeds=main_prompt_embeds,
                                       text_ids=main_text_ids, image_ids=main_image_ids,
                                       edit_latents=edit_condition.edit_latents,
                                       edit_image_ids=edit_condition.edit_image_ids)
                noise_pred = _v_no + _g_sd * (_v_ed - _v_no)
        else:
            with torch.no_grad():
                noise_pred = main_dit(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=main_guidance,
                    encoder_hidden_states=main_prompt_embeds,
                    txt_ids=main_text_ids,
                    img_ids=main_image_ids,
                )
        if args.style_mode == "kv_match":
            kv_cache.clear()
            kv_controller.step()
        latents_before_reverse = latents
        latents_after_reverse = scheduler.step(noise_pred, timestep_cpu, latents_before_reverse)
        reverse_step_strength = float(args.reverse_step_strength)
        latents = latents_before_reverse + reverse_step_strength * (
            latents_after_reverse - latents_before_reverse
        )
        main_scheduler_calls += 1

        if float(timestep_cpu.item()) >= float(args.start_time):
            row_logs.append(
                {
                    "outer_step": outer_idx,
                    "inner_iter": -1,
                    "timestep_style": None,
                    "sigma": None,
                    "score_delta_coeff": None,
                    "scheduler_calls_so_far": main_scheduler_calls,
                    "skipped": "pre_start_time",
                }
            )
            continue

        if outer_idx + 1 >= len(timesteps):
            row_logs.append(
                {
                    "outer_step": outer_idx,
                    "inner_iter": -1,
                    "timestep_style": None,
                    "sigma": None,
                    "score_delta_coeff": None,
                    "scheduler_calls_so_far": main_scheduler_calls,
                    "skipped": "final_step_no_next_sigma",
                }
            )
            continue

        timestep_style = timesteps[outer_idx + 1 : outer_idx + 2]
        style_noise = torch.randn_like(style_latent)
        style_latent_t = scheduler.add_noise(style_latent, style_noise, timestep_style).to(device="cuda", dtype=torch.bfloat16)
        sigma_info = resolve_sigma_info(
            scheduler,
            timestep_style,
            ref_tensor=latents,
            sigma_eps=args.sigma_eps,
        )

        content_anchor = latents.detach().clone()
        latents = warm_start_latents(
            latents,
            style_latent_t=style_latent_t,
            style_latent_clean=style_latent,
            args=args,
        )
        optimized_outer_steps += 1

        kv_ref_k_list = None
        kv_ref_v_list = None
        kv_sinkhorn_tau = None
        if args.style_mode == "kv_match":
            if kv_cache is None or kv_controller is None:
                raise RuntimeError("kv_match requires main DiT cache hooks to be initialized")
            kv_model_timestep = timestep_style.to(device=latents.device, dtype=latents.dtype) / 1000
            kv_sinkhorn_tau = kv_match_sinkhorn_tau(
                outer_timestep_value=float(timestep_cpu.item()),
                start_time=float(args.start_time),
                tau_max=float(args.kv_ot_tau_max),
                tau_min=float(args.kv_ot_tau_min),
            )
            kv_ref_k_list, kv_ref_v_list, _ = capture_reference_kv(
                dit=main_dit,
                cache=kv_cache,
                controller=kv_controller,
                style_latent_t=style_latent_t,
                model_timestep=kv_model_timestep,
                null_prompt_embeds=_main_null_embeds,
                null_text_ids=_main_null_text_ids,
                style_image_ids=style_image_ids,
                kv_grag_shallow=args.kv_grag_shallow,
                kv_grag_deep=args.kv_grag_deep,
            )
            ref_capture_calls += 1
        else:
            kv_model_timestep = None

        latents, inner_logs = inner_optimize(
            teacher_dit=main_dit if args.style_mode == "kv_match" else teacher_dit,
            latents=latents,
            outer_step_idx=outer_idx,
            timestep_style=timestep_style,
            outer_timestep_value=float(timestep_cpu.item()),
            sigma_info=sigma_info,
            teacher_prompt_embeds=teacher_prompt_embeds if teacher_prompt_embeds is not None else main_prompt_embeds,
            teacher_text_ids=teacher_text_ids if teacher_text_ids is not None else main_text_ids,
            teacher_without_prompt_embeds=(
                teacher_without_prompt_embeds if teacher_without_prompt_embeds is not None else main_prompt_embeds
            ),
            teacher_without_text_ids=(
                teacher_without_text_ids if teacher_without_text_ids is not None else main_text_ids
            ),
            teacher_image_ids=teacher_image_ids,
            edit_condition=edit_condition,
            style_latent_t=style_latent_t,
            style_latent_clean=style_latent,
            vgg_decode=_vgg_decode,
            vgg_targets=_vgg_targets,
            content_anchor=content_anchor,
            args=args,
            fake_backend=fake_backend,
            cond_for_backend=cond_for_backend,
            gradient_calibration_state=gradient_calibration_state,
            gradient_audit_requested=gradient_audit_requested,
            kv_cache=kv_cache,
            kv_controller=kv_controller,
            kv_ref_k_list=kv_ref_k_list,
            kv_ref_v_list=kv_ref_v_list,
            kv_null_prompt_embeds=_main_null_embeds if args.style_mode == "kv_match" else None,
            kv_null_text_ids=_main_null_text_ids if args.style_mode == "kv_match" else None,
            kv_image_ids=main_image_ids if args.style_mode == "kv_match" else None,
            kv_sinkhorn_tau=kv_sinkhorn_tau,
            kv_model_timestep=kv_model_timestep,
            scheduler=scheduler,
            marginalize_edit_conditions=marginalize_edit_conditions,
        )
        for item in inner_logs:
            row_logs.append(
                {
                    "outer_step": outer_idx,
                    "timestep_style": float(timestep_style.item()),
                    "sigma": float(sigma_info.sigma.item()),
                    "score_delta_coeff": float(sigma_info.score_delta_coefficient.item()),
                    "scheduler_calls_so_far": main_scheduler_calls,
                    "ref_capture_calls_so_far": ref_capture_calls,
                    "same_state_contract": same_state_contract_for_mode(args.style_mode, args.without_prompt_mode),
                    "without_shared_with_pre": without_shared_with_pre_for_mode(args.style_mode),
                    **item,
                }
            )
        if inner_logs:
            last = inner_logs[-1]
            pbar.set_postfix(
                sigma=f"{float(sigma_info.sigma.item()):.4f}",
                coeff=f"{float(sigma_info.score_delta_coefficient.item()):.3f}",
                sl=f"{last['style_loss']:.3f}",
                dl=f"{last['dist_loss']:.3f}",
            )

    if main_scheduler_calls != len(timesteps):
        raise RuntimeError(
            f"scheduler.step contract violated: calls={main_scheduler_calls} timesteps={len(timesteps)}"
        )
    if args.style_mode == "kv_match" and ref_capture_calls != optimized_outer_steps:
        raise RuntimeError(
            "kv_match ref capture contract violated: "
            f"ref_capture_calls={ref_capture_calls} optimized_outer_steps={optimized_outer_steps}"
        )

    first_gradient_audit_row = next(
        (row for row in row_logs if row.get("gradient_audit_measured")),
        None,
    )

    clean_latent_export_summary = None
    if args.export_clean_latent_path:
        export_path = Path(args.export_clean_latent_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        clean_latent_cpu = latents.detach().to(device="cpu").clone()
        export_metadata = {
            "shape": list(clean_latent_cpu.shape),
            "dtype": str(clean_latent_cpu.dtype),
            "tensor_sha256": tensor_sha256(clean_latent_cpu),
            "height": args.height,
            "width": args.width,
            "latent_h": lat_h,
            "latent_w": lat_w,
            "style_mode": args.style_mode,
            "style_id": args.style_id,
            "style_path": str(style_path),
            "reference_sha256": file_sha256(style_path),
            "prompt_id": args.prompt_id,
            "prompt_text": prompt_text,
            "content_prompt_text": content_prompt_text,
            "without_prompt_mode": args.without_prompt_mode,
            "seed": args.seed,
            "main_model_manifest_sha256": manifest_sha256(main_model_root),
            "teacher_model_manifest_sha256": optional_manifest_sha256(
                teacher_model_root,
                enabled=vram_contract.teacher_model_required,
            ),
            "fake_init_model_manifest_sha256": optional_manifest_sha256(
                fake_model_root,
                enabled=args.style_mode == "ca_dmd" and args.fake_score_backend == "klein_full_fake",
            ),
            "final_adain_applied_after_export": not args.disable_final_adain,
        }
        torch.save(
            {
                "clean_latent": clean_latent_cpu,
                "metadata": export_metadata,
            },
            export_path,
        )
        clean_latent_export_summary = {
            "path": str(export_path),
            "file_sha256": file_sha256(export_path),
            **export_metadata,
        }

    latents = final_adain_if_enabled(
        latents,
        style_latent,
        enabled=not args.disable_final_adain,
    )
    main_pipe.load_models_to_device(["vae"])
    # In the shared-pipe path (teacher_pipe is main_pipe) the VAE may have been moved to CPU by the
    # TEACHER aux-offload, so check both lists — otherwise decode hits a CPU-weight/CUDA-input mismatch.
    if ("vae" in main_aux_modules_offloaded or "vae" in teacher_aux_modules_offloaded) and not main_pipe.vram_management_enabled:
        main_pipe.vae = main_pipe.vae.to("cuda")
        print("[main-memory] reloaded main VAE to CUDA for decode", flush=True)
    latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    image = main_pipe.vae.decode(latents_4d)
    image = main_pipe.vae_output_to_image(image)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_wall_time_s = time.perf_counter() - inference_started_at
    for handle in forward_hook_handles:
        handle.remove()

    _xdump = os.environ.get("DUMP_XHAT_STEPS", "")
    if _xdump:
        import glob as _glob
        _dd = "agent_runs/xhat_dump"
        for _pt in sorted(_glob.glob(f"{_dd}/{args.out_prefix}_style{args.style_id}_{args.prompt_id}_o*.pt")):
            _d = torch.load(_pt)
            for _key in ("x0s", "x00"):
                _t4 = rearrange(_d[_key].to(device="cuda", dtype=torch.bfloat16), "B (H W) C -> B C H W", H=lat_h, W=lat_w)
                _im = main_pipe.vae_output_to_image(main_pipe.vae.decode(_t4))
                _im.save(results_dir / f"{args.out_prefix}_style{args.style_id}_{args.prompt_id}_o{int(_d['outer']):02d}_{_key}_sig{float(_d['sigma']):.3f}.png")
                print(f"[xhat-decode] o{int(_d['outer']):02d} {_key} sig{float(_d['sigma']):.3f}", flush=True)

    out_prefix = f"{args.out_prefix}_{args.style_mode}_style{args.style_id}_{args.prompt_id}_seed{args.seed}"
    if diagnostic_single_model:
        out_prefix = f"diagnostic_{out_prefix}"
    image_path = results_dir / f"{out_prefix}.png"
    style_copy_path = results_dir / f"{out_prefix}_reference.png"
    csv_path = results_dir / f"{out_prefix}.csv"
    summary_path = results_dir / f"{out_prefix}_summary.json"

    image.save(image_path)
    style_image.save(style_copy_path)
    write_csv(row_logs, csv_path)
    fake_backend_summary = None
    if fake_backend is not None and hasattr(fake_backend, "summary_dict"):
        fake_backend_summary = fake_backend.summary_dict()
    if fake_backend is not None and getattr(args, "fake_save_state_path", None):
        fake_backend.save_state(args.fake_save_state_path)
        if fake_backend_summary is not None:
            fake_backend_summary["saved_state_path"] = args.fake_save_state_path

    backward_calls = sum(
        1 for row in row_logs
        if isinstance(row.get("inner_iter"), int) and int(row["inner_iter"]) >= 0
    )
    process_wall_time_s = time.perf_counter() - process_started_at
    shared_dit_forward_calls = int(dit_forward_counts["shared"])
    main_dit_forward_calls = int(dit_forward_counts["main"])
    teacher_dit_forward_calls = int(dit_forward_counts["teacher"])
    summary = {
        "style_mode": args.style_mode,
        "fake_score_backend": args.fake_score_backend,
        "experimental_scope": (
            fake_backend_summary.get("experimental_scope")
            if isinstance(fake_backend_summary, dict) and fake_backend_summary.get("experimental_scope") is not None
            else ("plumbing_only" if args.style_mode == "ca_dmd" else "standard_runner")
        ),
        "diagnostic_single_model": diagnostic_single_model,
        "main_embedded_guidance_mode": "tensor" if args.embedded_guidance > 0 else "none",
        "main_embedded_guidance_value": float(args.embedded_guidance),
        "teacher_embedded_guidance_scalar": 0.0,
        "smoke_overrides": smoke_overrides(args),
        "main_model_root": str(main_model_root),
        "teacher_model_root": str(teacher_model_root),
        "fake_model_root": str(fake_model_root) if args.style_mode == "ca_dmd" else None,
        "legacy_vram_limit_gb": args.vram_limit_gb,
        "main_vram_limit_gb_requested": vram_contract.requested_main_vram_limit_gb,
        "teacher_vram_limit_gb_requested": vram_contract.requested_teacher_vram_limit_gb,
        "main_vram_limit_gb_effective": vram_contract.effective_main_vram_limit_gb,
        "teacher_vram_limit_gb_effective": vram_contract.effective_teacher_vram_limit_gb,
        "main_loading_mode": main_loading,
        "teacher_loading_mode": teacher_loading,
        "fake_loading_mode": fake_loading_mode,
        "teacher_model_required": vram_contract.teacher_model_required,
        "main_requires_full_cuda": vram_contract.main_requires_full_cuda,
        "teacher_requires_full_cuda": vram_contract.teacher_requires_full_cuda,
        "main_aux_modules_offloaded_to_cpu": main_aux_modules_offloaded,
        "teacher_aux_modules_offloaded_to_cpu": teacher_aux_modules_offloaded,
        "fake_aux_modules_offloaded_to_cpu": fake_aux_modules_offloaded,
        "shared_runtime_pipe": share_runtime_pipe,
        "main_transformer_config_sha256": file_sha256(main_model_root / "transformer" / "config.json"),
        "main_scheduler_config_sha256": file_sha256(main_model_root / "scheduler" / "scheduler_config.json"),
        "main_model_manifest_sha256": manifest_sha256(main_model_root),
        "teacher_transformer_config_sha256": optional_file_sha256(
            teacher_model_root / "transformer" / "config.json",
            enabled=vram_contract.teacher_model_required,
        ),
        "teacher_scheduler_config_sha256": optional_file_sha256(
            teacher_model_root / "scheduler" / "scheduler_config.json",
            enabled=vram_contract.teacher_model_required,
        ),
        "teacher_model_manifest_sha256": optional_manifest_sha256(
            teacher_model_root,
            enabled=vram_contract.teacher_model_required,
        ),
        "fake_transformer_config_sha256": optional_file_sha256(
            fake_model_root / "transformer" / "config.json",
            enabled=args.style_mode == "ca_dmd" and args.fake_score_backend == "klein_full_fake",
        ),
        "fake_scheduler_config_sha256": optional_file_sha256(
            fake_model_root / "scheduler" / "scheduler_config.json",
            enabled=args.style_mode == "ca_dmd" and args.fake_score_backend == "klein_full_fake",
        ),
        "fake_model_manifest_sha256": optional_manifest_sha256(
            fake_model_root,
            enabled=args.style_mode == "ca_dmd" and args.fake_score_backend == "klein_full_fake",
        ),
        "prompt_text": prompt_text,
        "without_prompt_text": without_prompt_text,
        "without_prompt_mode": args.without_prompt_mode,
        "final_adain_enabled": not args.disable_final_adain,
        "loss_isolation_contract": build_loss_isolation_contract(
            disable_slt=args.disable_slt,
            adain_alpha=args.adain_alpha,
            disable_final_adain=args.disable_final_adain,
        ),
        "prompt_id": args.prompt_id,
        "style_id": args.style_id,
        "style_path": str(style_path),
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "reverse_step_strength": float(args.reverse_step_strength),
        "start_time": args.start_time,
        "lr": float(args.lr),
        "latent_optimizer": args.latent_optimizer,
        "latent_weight_decay": float(args.latent_weight_decay),
        "iterations_per_step": int(args.iters),
        "dynamic_iterations_enabled": not bool(args.disable_dynamic_iters),
        "pseudo_step_size": float(args.pseudo_step_size),
        "pseudo_grad_normalization": str(args.pseudo_grad_normalization),
        "cfg_scale": float(args.cfg_scale),
        "lambda_ca": float(args.lambda_ca),
        "lambda_dm": float(args.lambda_dm),
        "wall_time_process_s": float(process_wall_time_s),
        "wall_time_inference_s": float(inference_wall_time_s),
        "dit_forward_calls_total": int(
            shared_dit_forward_calls + main_dit_forward_calls + teacher_dit_forward_calls
        ),
        "shared_dit_forward_calls": (
            shared_dit_forward_calls if teacher_dit is main_dit else None
        ),
        "main_dit_forward_calls": (
            None if teacher_dit is main_dit else main_dit_forward_calls
        ),
        "teacher_dit_forward_calls": (
            None if teacher_dit is main_dit else teacher_dit_forward_calls
        ),
        "latent_backward_calls": int(backward_calls),
        "timesteps": [float(t.item()) for t in timesteps],
        "sigmas": [float(s.item()) for s in scheduler.sigmas],
        "rows": row_logs,
        "scheduler_step_calls_total": main_scheduler_calls,
        "optimized_outer_steps": optimized_outer_steps,
        "ref_capture_calls": ref_capture_calls,
        "denoise_cache_cleared": args.style_mode == "kv_match",
        "same_state_contract": same_state_contract_for_mode(args.style_mode, args.without_prompt_mode),
        "without_shared_with_pre_contract": (
            "not_applicable_for_kv_match"
            if args.style_mode == "kv_match"
            else (
                "without/pre branch uses a content-only negative prompt with no reference while with branch keeps the full style prompt plus reference"
                if args.without_prompt_mode == "content_only"
                else "without branch is shared with the pre branch at the same state, then detached where required"
            )
        ),
        "role_contract": (
            "main_denoiser = distilled main_model_root for denoise/scheduler.step/warm-start; "
            "kv_style_objective = cached Klein9B self-attention K/V matching on the same main DiT with null prompt, "
            "capturing reference K/V once per optimized outer step and reusing them across inner iters."
            if args.style_mode == "kv_match"
            else
            "main_denoiser = distilled main_model_root for denoise/scheduler.step/warm-start; "
            "native_teacher = undistilled teacher_model_root for with/without style scores; "
            "fake_critic = standalone full fake_model_root checkpoint with its own prompt encoder and native edit condition, "
            "trained online as an external critic and never used for final image decode."
            if args.style_mode == "ca_dmd"
            else
            "main_denoiser = distilled main_model_root for denoise/scheduler.step/warm-start; "
            "native_teacher = undistilled teacher_model_root for with/without/pre style objectives."
            if not diagnostic_single_model
            else "diagnostic single-model run: teacher_model_root == main_model_root, so this is not a formal TPAMI loss-only control."
        ),
        "teacher_runtime_contract": (
            "kv_match does not load an external teacher model; main text/VAE modules are moved back to CPU after encoding and only the main DiT stays resident on CUDA until decode."
            if args.style_mode == "kv_match"
            else
            "score_match keeps teacher_dit on full CUDA for latent backward and offloads teacher text/VAE modules after encoding."
            if vram_contract.teacher_requires_full_cuda
            else "Plan B teacher branches are no-grad and may keep using layer-wise offload when requested."
        ),
        "fake_backend_contract": fake_backend_summary,
        "fake_condition_contract": (
            "fake critic scoring uses the same prompt text and the same reference image semantics as the real with-style teacher, "
            "but re-encodes them with fake_model_root's own prompt encoder and native edit condition. "
            "Critic train_step samples an independent noise/timestep pair, then predict_velocity scores the current z_sigma/timestep."
            if args.style_mode == "ca_dmd"
            else None
        ),
        "teacher_conditioning_contract": (
            "positive branch = full style prompt + reference; negative/pre branch = content-only prompt + no reference"
            if args.without_prompt_mode == "content_only"
            else "positive/negative/pre branches share the same full style prompt text; only the native reference condition differs"
        ),
        "style_gradient_protocol": {
            "log_gradient_components": bool(args.log_gradient_components),
            "style_grad_rms_target": args.style_grad_rms_target,
            "calibration_enabled": args.style_grad_rms_target is not None,
            "calibration_eps": STYLE_GRAD_RMS_CALIBRATION_EPS,
            "calibration_scale": float(gradient_calibration_state.scale),
            "calibration_frozen": bool(gradient_calibration_state.frozen),
            "measurement_logged_once": bool(gradient_calibration_state.measured_once),
            "measured_outer_step": gradient_calibration_state.measured_outer_step,
            "measured_inner_iter": gradient_calibration_state.measured_inner_iter,
            "measured_style_mode": gradient_calibration_state.measured_style_mode,
            "original_style_weight": float(args.style_weight),
            "effective_style_weight": float(
                gradient_calibration_state.effective_style_weight
                if gradient_calibration_state.effective_style_weight is not None
                else float(args.style_weight)
            ),
            "formula": (
                "require |style_weight| * raw_style_grad_rms > eps; "
                "scale = target / (|style_weight| * raw_style_grad_rms); "
                "effective_style_weight = style_weight * scale"
            ),
            "protocols": {
                "original_weight": "style_grad_rms_target=None, so effective_style_weight == original style_weight",
                "matched_gradient": "set style_grad_rms_target to freeze a first-step scale that matches the final weighted style gradient RMS",
            },
            "first_active_inner_step_audit": first_gradient_audit_row,
        },
        "warm_start_note": warm_start_note_for_mode(args.style_mode),
        "kv_match_contract": (
            {
                "self_layers": [args.kv_self_layer_start, args.kv_self_layer_end],
                "sinkhorn_mode": args.kv_sinkhorn_mode,
                "ot_tau_max": args.kv_ot_tau_max,
                "ot_tau_min": args.kv_ot_tau_min,
                "ot_iters": args.kv_ot_iters,
                "ot_layer_frac": args.kv_ot_layer_frac,
                "freq_hi_boost": args.kv_freq_hi_boost,
                "kv_grag_shallow": args.kv_grag_shallow,
                "kv_grag_deep": args.kv_grag_deep,
                "ref_capture_calls_must_match_optimized_outer_steps": True,
            }
            if args.style_mode == "kv_match"
            else None
        ),
        "warm_start_contract": {
            "adain_alpha": args.adain_alpha,
            "slt_enabled": not args.disable_slt,
            "slt_alpha": args.slt_alpha,
            "slt_tau": args.slt_tau,
            "slt_iters": args.slt_iters,
            "slt_grag": args.slt_grag,
            "warm_start_grag_delta": 0.5 * (float(args.grag_shallow) + float(args.grag_deep)),
            "note": warm_start_note_for_mode(args.style_mode),
        },
        "image_path": str(image_path),
        "reference_image_path": str(style_copy_path),
        "clean_latent_export": clean_latent_export_summary,
        "cuda_peak_memory_gb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else None
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved image -> {image_path}")
    print(f"Saved reference -> {style_copy_path}")
    print(f"Saved logs -> {csv_path}")
    print(f"Saved summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
