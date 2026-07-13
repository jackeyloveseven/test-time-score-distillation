"""Generate normal / hard / student image samples for Klein9B residual adapter checks."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
import types
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapter_runtime import ResidualAdapterRuntime
from src.attention_hooks import Klein9BCache, Klein9BController, register_klein9b_attn_hooks
from src.klein9b_loading import model_config_offload_kwargs, model_loading_mode
from src.residuals import compute_normal_velocity, compute_styled_velocity
from src.score_student_runtime import ScoreResidualStudentRuntime


DIFFSYNTH_ROOT = "/mnt/local/yzh/prog/DiffSynth-Studio"
KLEIN9B_ROOT = "/mnt/local/klein9b"
STYLE_IMAGES_ROOT = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty"

PROMPTS = {
    "kp0": "A panda eat bamboo",
    "kp1": "A castle on a cliff by the sea",
    "kp2": "A cat sitting on a windowsill",
    "kp3": "A field of sunflowers at sunset",
    "kp4": "A portrait of an old fisherman",
}


def pooled_reference_code(style_latent_4d: torch.Tensor) -> torch.Tensor:
    """Match the frozen VAE reference code written by the native teacher probe."""

    tokens = style_latent_4d.float().flatten(2).transpose(1, 2)
    return torch.cat(
        [tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)],
        dim=-1,
    ).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Klein9B images with student residual guidance.")
    parser.add_argument("--style-id", default="27")
    parser.add_argument("--prompt-id", default="kp0")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--out-prefix", default="student_sample")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--adapter-checkpoint", default="results/residual_adapter_hp45_plain/adapter_best.pt")
    parser.add_argument("--adapter-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--student-scale", type=float, default=0.75)
    parser.add_argument("--score-student-checkpoint", default="results/score_residual_student_latent45_lam1_hard_delta/student_best.pt")
    parser.add_argument("--score-student-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--score-student-scale", type=float, default=1.0)
    parser.add_argument(
        "--score-student-native-only",
        action="store_true",
        help="Apply the score student to v_normal without the legacy K/V hard carrier.",
    )
    parser.add_argument(
        "--score-student-style-scales",
        default="",
        help=(
            "Optional per-style score student scale map, e.g. '08:0.75,22:1.25'. "
            "Values override --score-student-scale for matching style ids."
        ),
    )
    parser.add_argument(
        "--score-student-timestep-min",
        type=float,
        default=None,
        help="Optional lower timestep bound for applying score student correction.",
    )
    parser.add_argument(
        "--score-student-timestep-max",
        type=float,
        default=None,
        help="Optional upper timestep bound for applying score student correction.",
    )
    parser.add_argument(
        "--reference-guidance-timestep-min",
        type=float,
        default=None,
        help="Optional lower timestep bound for hard/reference guidance.",
    )
    parser.add_argument(
        "--reference-guidance-timestep-max",
        type=float,
        default=None,
        help=(
            "Optional upper timestep bound for hard/reference guidance. "
            "Set below the first scheduler timestep to skip reference guidance "
            "while the noised reference contains no image signal."
        ),
    )
    parser.add_argument("--modes", default="normal,hard,student")
    parser.add_argument("--inject-beta", type=float, default=1.0)
    parser.add_argument("--self-layer-start", type=int, default=12)
    parser.add_argument("--self-layer-end", type=int, default=28)
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        default=None,
        help=(
            "Enable lossless BF16 CPU offload and stop preloading CUDA modules "
            "after total GPU usage reaches this threshold."
        ),
    )
    return parser.parse_args()


def parse_modes(raw: str) -> List[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"normal", "hard", "student", "score_student"}
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}; valid={sorted(valid)}")
    if not modes:
        raise ValueError("--modes must not be empty")
    return modes


def parse_style_scales(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid style scale item {item!r}; expected style_id:scale")
        style_id, value = item.split(":", 1)
        style_id = style_id.strip()
        if not style_id:
            raise ValueError(f"Invalid empty style id in {item!r}")
        out[style_id] = float(value.strip())
    return out


def timestep_in_window(timestep: float, min_t: float | None, max_t: float | None) -> bool:
    if min_t is not None and timestep < float(min_t):
        return False
    if max_t is not None and timestep > float(max_t):
        return False
    return True


def timestep_window_scale(timestep: float, base_scale: float, min_t: float | None, max_t: float | None) -> float:
    if not timestep_in_window(timestep, min_t, max_t):
        return 0.0
    return float(base_scale)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _install_modelscope_stub() -> None:
    if "modelscope" in sys.modules:
        return
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("modelscope not installed; use local checkpoints")
    )
    sys.modules["modelscope"] = module


def _sorted_files(pattern: str) -> list[str]:
    import glob

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def main() -> int:
    args = parse_args()
    modes = parse_modes(args.modes)
    style_id = str(args.style_id)
    score_student_style_scales = parse_style_scales(args.score_student_style_scales)
    score_student_effective_scale = float(score_student_style_scales.get(style_id, args.score_student_scale))
    prompt_text = args.prompt or PROMPTS.get(args.prompt_id, args.prompt_id)
    style_path = Path(STYLE_IMAGES_ROOT) / f"{style_id}.png"
    if not style_path.exists():
        raise FileNotFoundError(f"Style image not found: {style_path}")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    os.environ.setdefault("DIFFSYNTH_ROOT", DIFFSYNTH_ROOT)
    os.environ.setdefault("KLEIN9B_ROOT", KLEIN9B_ROOT)
    diffsynth_root = os.environ["DIFFSYNTH_ROOT"]
    model_root = Path(os.environ["KLEIN9B_ROOT"])
    if str(diffsynth_root) not in sys.path:
        sys.path.insert(0, str(diffsynth_root))
    _install_modelscope_stub()

    from diffsynth.pipelines.flux2_image import (
        Flux2ImagePipeline,
        Flux2Unit_ImageIDs,
        Flux2Unit_PromptEmbedder,
        Flux2Unit_Qwen3PromptEmbedder,
        ModelConfig,
    )
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from einops import rearrange
    from PIL import Image

    result_dir = Path(args.results_dir)
    image_dir = result_dir / "student_samples"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_path = result_dir / f"{args.out_prefix}_meta.json"

    _set_seed(args.seed)
    print(f"Loading Klein9B from: {model_root}")
    offload_kwargs = model_config_offload_kwargs(args.vram_limit_gb)
    loading_mode = model_loading_mode(args.vram_limit_gb)
    print(f"Model loading mode: {loading_mode}")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        vram_limit=args.vram_limit_gb,
        model_configs=[
            ModelConfig(
                path=_sorted_files(str(model_root / "text_encoder" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(
                path=_sorted_files(str(model_root / "transformer" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(path=str(model_root / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(model_root / "tokenizer")),
    )
    if args.vram_limit_gb is None:
        pipe.vram_management_enabled = False

    vae = pipe.vae
    scheduler = FlowMatchScheduler("FLUX.2")

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        style_img = pipe.preprocess_image(
            Image.open(style_path).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
        )
        style_latent_4d = vae.encode(style_img)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")
        reference_code = pooled_reference_code(style_latent_4d)

    if pipe.text_encoder_qwen3 is not None:
        pipe.load_models_to_device(["text_encoder_qwen3"])
        embedder = Flux2Unit_Qwen3PromptEmbedder()
        prompt_embeds, text_ids = embedder.encode_prompt(
            pipe.text_encoder_qwen3,
            pipe.tokenizer,
            prompt_text,
            dtype=pipe.torch_dtype,
            device="cuda",
        )
        null_embeds, null_text_ids = embedder.encode_prompt(
            pipe.text_encoder_qwen3,
            pipe.tokenizer,
            "",
            dtype=pipe.torch_dtype,
            device="cuda",
        )
    else:
        pipe.load_models_to_device(["text_encoder"])
        embedder = Flux2Unit_PromptEmbedder()
        prompt_embeds, text_ids = embedder.encode_prompt(
            pipe.text_encoder,
            pipe.tokenizer,
            prompt_text,
            dtype=pipe.torch_dtype,
            device="cuda",
        )
        null_embeds, null_text_ids = embedder.encode_prompt(
            pipe.text_encoder,
            pipe.tokenizer,
            "",
            dtype=pipe.torch_dtype,
            device="cuda",
        )

    lat_h = args.height // 16
    lat_w = args.width // 16
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    init_noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16, generator=generator)
    latents_by_mode: Dict[str, torch.Tensor] = {
        mode: rearrange(init_noise.clone(), "B C H W -> B (H W) C") for mode in modes
    }

    image_id_unit = Flux2Unit_ImageIDs()
    image_ids = image_id_unit.prepare_latent_ids(lat_h, lat_w).to("cuda")
    style_lat_h = int(style_latent_4d.shape[-2])
    style_lat_w = int(style_latent_4d.shape[-1])
    style_image_ids = image_id_unit.prepare_latent_ids(style_lat_h, style_lat_w).to("cuda")

    scheduler.set_timesteps(args.num_steps, dynamic_shift_len=lat_h * lat_w)

    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    dit.to("cuda")
    dit.requires_grad_(False)

    cache = Klein9BCache()
    controller = Klein9BController(self_layers=(args.self_layer_start, args.self_layer_end))
    register_klein9b_attn_hooks(dit, controller, cache)

    adapter_runtime = None
    if "student" in modes:
        adapter_runtime = ResidualAdapterRuntime.from_checkpoint(
            args.adapter_checkpoint,
            device=args.adapter_device,
            student_scale=args.student_scale,
        )
        print(f"Loaded adapter type={adapter_runtime.adapter_type} scale={args.student_scale}")

    score_student_runtime = None
    if "score_student" in modes:
        score_student_runtime = ScoreResidualStudentRuntime.from_checkpoint(
            args.score_student_checkpoint,
            device=args.score_student_device,
            student_scale=score_student_effective_scale,
        )
        print(
            f"Loaded score student scale={score_student_effective_scale} "
            f"mode={score_student_runtime.prediction_mode}"
        )

    score_student_carrier_only = bool(
        score_student_runtime is not None
        and (args.score_student_native_only or score_student_runtime.prediction_mode == "carrier_anchored")
    )

    step_summaries = []
    for step_idx, t_val in enumerate(scheduler.timesteps):
        timestep_value = float(t_val.item())
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        style_timestep = scheduler.timesteps[step_idx: step_idx + 1]
        sty_noise = torch.randn_like(style_latent)
        style_latent_t = scheduler.add_noise(style_latent, sty_noise, style_timestep)
        reference_guidance_active = timestep_in_window(
            timestep_value,
            args.reference_guidance_timestep_min,
            args.reference_guidance_timestep_max,
        )

        ref_k_list = []
        ref_v_list = []
        if reference_guidance_active and (
            any(mode in modes for mode in ("hard", "student"))
            or ("score_student" in modes and not score_student_carrier_only)
        ):
            controller.reset_mode()
            cache.clear()
            controller.step()
            with torch.no_grad():
                _ = dit(
                    hidden_states=style_latent_t.to(device="cuda", dtype=torch.bfloat16),
                    timestep=style_timestep.to(device="cuda", dtype=torch.bfloat16) / 1000,
                    guidance=None,
                    encoder_hidden_states=null_embeds,
                    txt_ids=null_text_ids,
                    img_ids=style_image_ids,
                )
            _, ref_k_list, ref_v_list, _ = cache.get()
            if len(ref_k_list) == 0 or len(ref_v_list) == 0:
                raise RuntimeError("No reference K/V tensors captured")

        noise_preds: Dict[str, torch.Tensor] = {}
        if "normal" in modes:
            controller.reset_mode()
            cache.clear()
            controller.step()
            noise_preds["normal"] = compute_normal_velocity(
                dit,
                latents=latents_by_mode["normal"],
                timestep=timestep / 1000,
                guidance=None,
                prompt_embeds=prompt_embeds,
                txt_ids=text_ids,
                image_ids=image_ids,
            )

        if "hard" in modes:
            if reference_guidance_active:
                noise_preds["hard"] = compute_styled_velocity(
                    dit,
                    latents=latents_by_mode["hard"],
                    timestep=timestep / 1000,
                    guidance=None,
                    prompt_embeds=prompt_embeds,
                    txt_ids=text_ids,
                    image_ids=image_ids,
                    controller=controller,
                    cache=cache,
                    ref_k_list=ref_k_list,
                    ref_v_list=ref_v_list,
                    mode="inject",
                    inject_beta=args.inject_beta,
                )
            else:
                noise_preds["hard"] = compute_normal_velocity(
                    dit,
                    latents=latents_by_mode["hard"],
                    timestep=timestep / 1000,
                    guidance=None,
                    prompt_embeds=prompt_embeds,
                    txt_ids=text_ids,
                    image_ids=image_ids,
                )

        if "student" in modes:
            v_normal = compute_normal_velocity(
                dit,
                latents=latents_by_mode["student"],
                timestep=timestep / 1000,
                guidance=None,
                prompt_embeds=prompt_embeds,
                txt_ids=text_ids,
                image_ids=image_ids,
            )
            if reference_guidance_active:
                v_hard = compute_styled_velocity(
                    dit,
                    latents=latents_by_mode["student"],
                    timestep=timestep / 1000,
                    guidance=None,
                    prompt_embeds=prompt_embeds,
                    txt_ids=text_ids,
                    image_ids=image_ids,
                    controller=controller,
                    cache=cache,
                    ref_k_list=ref_k_list,
                    ref_v_list=ref_v_list,
                    mode="inject",
                    inject_beta=args.inject_beta,
                )
                hard_residual = (v_hard - v_normal).detach().float().cpu()
                adapter_delta = adapter_runtime.predict_delta(
                    hard_residual,
                    timestep=timestep_value,
                    style_id=style_id,
                    prompt_id=args.prompt_id,
                ).to(device=v_hard.device, dtype=v_hard.dtype)
                noise_preds["student"] = v_hard + float(args.student_scale) * adapter_delta
                hard_residual_norm = float(torch.linalg.norm(hard_residual.reshape(1, -1), dim=1)[0].item())
                adapter_delta_norm = float(torch.linalg.norm(adapter_delta.float().reshape(1, -1), dim=1)[0].item())
            else:
                noise_preds["student"] = v_normal
                hard_residual_norm = 0.0
                adapter_delta_norm = 0.0
            step_summaries.append(
                {
                    "branch": "student",
                    "step": step_idx,
                    "timestep": timestep_value,
                    "reference_guidance_active": bool(reference_guidance_active),
                    "hard_residual_norm": hard_residual_norm,
                    "adapter_delta_norm": adapter_delta_norm,
                }
            )

        if "score_student" in modes:
            step_score_scale = timestep_window_scale(
                timestep_value,
                score_student_effective_scale,
                args.score_student_timestep_min,
                args.score_student_timestep_max,
            )
            z_t = latents_by_mode["score_student"]
            v_normal = compute_normal_velocity(
                dit,
                latents=z_t,
                timestep=timestep / 1000,
                guidance=None,
                prompt_embeds=prompt_embeds,
                txt_ids=text_ids,
                image_ids=image_ids,
            )
            if score_student_carrier_only:
                score_delta = score_student_runtime.predict_velocity_correction(
                    z_t.detach().float(),
                    timestep=timestep_value,
                    style_id=style_id,
                    prompt_id=args.prompt_id,
                    sigma=scheduler.sigmas[step_idx],
                    reference_code=reference_code,
                ).to(device=v_normal.device, dtype=v_normal.dtype)
                noise_preds["score_student"] = v_normal + step_score_scale * score_delta
                hard_residual_norm = 0.0
                score_delta_norm = float(torch.linalg.norm(score_delta.float().reshape(1, -1), dim=1)[0].item())
            elif reference_guidance_active:
                v_hard = compute_styled_velocity(
                    dit,
                    latents=z_t,
                    timestep=timestep / 1000,
                    guidance=None,
                    prompt_embeds=prompt_embeds,
                    txt_ids=text_ids,
                    image_ids=image_ids,
                    controller=controller,
                    cache=cache,
                    ref_k_list=ref_k_list,
                    ref_v_list=ref_v_list,
                    mode="inject",
                    inject_beta=args.inject_beta,
                )
                hard_residual = (v_hard - v_normal).detach().float()
                score_delta = score_student_runtime.predict_velocity_correction(
                    z_t.detach().float(),
                    timestep=timestep_value,
                    style_id=style_id,
                    prompt_id=args.prompt_id,
                    sigma=scheduler.sigmas[step_idx],
                    reference_code=reference_code,
                ).to(device=v_hard.device, dtype=v_hard.dtype)
                noise_preds["score_student"] = v_hard + step_score_scale * score_delta
                hard_residual_norm = float(torch.linalg.norm(hard_residual.reshape(1, -1), dim=1)[0].item())
                score_delta_norm = float(torch.linalg.norm(score_delta.float().reshape(1, -1), dim=1)[0].item())
            else:
                noise_preds["score_student"] = v_normal
                hard_residual_norm = 0.0
                score_delta_norm = 0.0
            step_summaries.append(
                {
                    "branch": "score_student",
                    "step": step_idx,
                    "timestep": timestep_value,
                    "reference_guidance_active": bool(reference_guidance_active),
                    "score_student_base_scale": float(score_student_effective_scale),
                    "score_student_scale": float(
                        step_score_scale if (reference_guidance_active or score_student_carrier_only) else 0.0
                    ),
                    "score_student_prediction_mode": score_student_runtime.prediction_mode,
                    "hard_residual_norm": hard_residual_norm,
                    "score_delta_norm": score_delta_norm,
                }
            )

        for mode, noise_pred in noise_preds.items():
            latents_by_mode[mode] = scheduler.step(noise_pred, t_val, latents_by_mode[mode])
        print(f"step {step_idx + 1}/{len(scheduler.timesteps)} t={float(t_val.item()):.3f}", flush=True)

    pipe.load_models_to_device(["vae"])
    image_paths = {}
    for mode, latents in latents_by_mode.items():
        latents_4d = rearrange(latents, "B (H W) C -> B C H W", H=lat_h, W=lat_w)
        with torch.no_grad():
            decoded = vae.decode(latents_4d)
        image = pipe.vae_output_to_image(decoded)
        image_path = image_dir / f"{args.out_prefix}_{mode}.png"
        image.save(image_path)
        image_paths[mode] = str(image_path)
        print(f"OK: saved {mode} -> {image_path}")

    meta = {
        "mode": "sample_student_images",
        "style_id": style_id,
        "prompt_id": args.prompt_id,
        "prompt_text": prompt_text,
        "style_path": str(style_path),
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "vram_limit_gb": args.vram_limit_gb,
        "model_loading_mode": loading_mode,
        "modes": modes,
        "adapter_checkpoint": args.adapter_checkpoint if "student" in modes else None,
        "adapter_device": args.adapter_device if "student" in modes else None,
        "student_scale": args.student_scale if "student" in modes else None,
        "score_student_checkpoint": args.score_student_checkpoint if "score_student" in modes else None,
        "score_student_device": args.score_student_device if "score_student" in modes else None,
        "score_student_scale": args.score_student_scale if "score_student" in modes else None,
        "score_student_native_only": bool(args.score_student_native_only),
        "score_student_style_scales": score_student_style_scales if "score_student" in modes else None,
        "score_student_effective_scale": score_student_effective_scale if "score_student" in modes else None,
        "score_student_timestep_min": args.score_student_timestep_min if "score_student" in modes else None,
        "score_student_timestep_max": args.score_student_timestep_max if "score_student" in modes else None,
        "reference_guidance_timestep_min": args.reference_guidance_timestep_min,
        "reference_guidance_timestep_max": args.reference_guidance_timestep_max,
        "inject_beta": args.inject_beta,
        "self_layers": [args.self_layer_start, args.self_layer_end],
        "latent_shape": [int(v) for v in next(iter(latents_by_mode.values())).shape],
        "image_paths": image_paths,
        "step_summaries": step_summaries,
        "invariant": (
            "Each branch calls scheduler.step exactly once per timestep. Inside the configured "
            "reference window, learned branches use v_hard + scale * learned_delta evaluated at "
            "the same branch z_t,t; outside it, reference branches use v_normal."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"OK: wrote meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[sample failed] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
