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
import os
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from tqdm import tqdm


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
    assert_static_condition_tensor,
    evaluate_native_edit_triplet,
    freeze_condition_tensor,
    prepare_edit_condition,
)
from src.prompt_text import (
    parse_content_prompt_text as build_content_prompt_text,
    parse_prompt_text as build_prompt_text,
)
from src.style_objectives import (
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
        choices=["score_match", "ca_only", "ca_dmd", "kv_match", "carrier_score", "score_carrier"],
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
    parser.add_argument("--pseudo-step-size", type=float, default=0.05)
    parser.add_argument("--pseudo-grad-normalization", default="rms", choices=["none", "rms", "l2"])
    parser.add_argument("--carrier-temperature", type=float, default=0.1)
    parser.add_argument("--carrier-strength", type=float, default=0.75)
    parser.add_argument("--carrier-highpass-gain", type=float, default=0.0)
    parser.add_argument("--carrier-loss-weight", type=float, default=0.25)
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
                strength=args.carrier_strength,
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
            probe_style_loss = score_objective.loss + float(args.carrier_loss_weight) * carrier_loss
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
                strength=args.carrier_strength,
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
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    loss_fn = torch.nn.L1Loss()
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
    if args.disable_dynamic_iters:
        iters_this_step = int(args.iters)
    else:
        progress = 1.0 - (float(outer_timestep_value) / max(float(args.start_time), 1.0))
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
                carrier_score, _ = content_affinity_transport(
                    goal_score,
                    content_anchor,
                    temperature=args.carrier_temperature,
                    strength=args.carrier_strength,
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
                style_loss = objective.loss + float(args.carrier_loss_weight) * carrier_loss
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
                }
            elif args.style_mode == "ca_only":
                sigma_broadcast = sigma_info.sigma.to(device=latents.device, dtype=latents.dtype)
                x0_pred = latents - sigma_broadcast * triplet.without_velocity.detach()
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
            elif args.style_mode == "carrier_score":
                sigma_broadcast = sigma_info.sigma.to(device=latents.device, dtype=latents.dtype)
                x0_pred = latents - sigma_broadcast * triplet.without_velocity.detach()
                delta_ca_velocity = triplet.with_velocity.detach() - triplet.without_velocity.detach()
                delta_ca_score, score_coeff = velocity_delta_to_score_delta(
                    delta_ca_velocity,
                    sigma_info.sigma,
                    sigma_eps=args.sigma_eps,
                )
                carrier_score, _carrier_affinity = content_affinity_transport(
                    delta_ca_score,
                    content_anchor,
                    temperature=args.carrier_temperature,
                    strength=args.carrier_strength,
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
                objective = pseudo_guidance_from_velocities(
                    "ca_dmd",
                    x0_pred=x0_pred,
                    v_with_detached=triplet.with_velocity,
                    v_without_detached=triplet.without_velocity,
                    v_fake_detached=v_fake,
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
    args = parse_args()
    validate_style_mode(args.style_mode, args.fake_score_backend)
    fake_backend = None

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    if args.diffsynth_root not in sys.path:
        sys.path.insert(0, args.diffsynth_root)
    install_modelscope_stub()

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
    if teacher_pipe is not None:
        edit_condition = prepare_edit_condition(teacher_pipe, style_image, auto_resize=False)
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
            fake_edit_condition = prepare_edit_condition(fake_pipe, style_image, auto_resize=False)
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
        latents = scheduler.step(noise_pred, timestep_cpu, latents)
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
    if "vae" in main_aux_modules_offloaded and not main_pipe.vram_management_enabled:
        main_pipe.vae = main_pipe.vae.to("cuda")
        print("[main-memory] reloaded main VAE to CUDA for decode", flush=True)
    latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
    image = main_pipe.vae.decode(latents_4d)
    image = main_pipe.vae_output_to_image(image)

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
        "start_time": args.start_time,
        "lr": float(args.lr),
        "latent_optimizer": args.latent_optimizer,
        "latent_weight_decay": float(args.latent_weight_decay),
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
