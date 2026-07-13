"""Audit FLUX.2 Base native edit conditioning as a with-style teacher field.

The probe follows a no-reference content trajectory. At every step, all
reference-conditioned predictions are evaluated on the exact same target
latent, timestep, prompt, and scheduler state. Only the clean edit-image tokens
change. This script never injects or replaces attention K/V tensors.
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
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flux2_score import velocity_delta_to_score_delta

DIFFSYNTH_ROOT = "/mnt/local/yzh/prog/DiffSynth-Studio"
MODEL_ROOT = "/mnt/local/klein9b-base"
STYLE_ROOT = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty"

PROMPTS = {
    "kp0": "A panda eating bamboo",
    "kp1": "A castle on a cliff by the sea",
    "kp2": "A cat sitting on a windowsill",
    "kp3": "A field of sunflowers at sunset",
    "kp4": "A portrait of an old fisherman",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default=MODEL_ROOT)
    parser.add_argument("--diffsynth-root", default=DIFFSYNTH_ROOT)
    parser.add_argument("--style-root", default=STYLE_ROOT)
    parser.add_argument("--style-ids", default="00,13,27")
    parser.add_argument("--prompt-ids", default="kp0,kp1,kp3")
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--embedded-guidance", type=float, default=0.0)
    parser.add_argument(
        "--prompt-template",
        default=(
            "Create {content}. Use the visual style of the reference image, "
            "including its medium, palette, texture, and brushwork, but do not "
            "copy the reference objects or composition."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="results/native_edit_teacher_field_audit_base9b_s65",
    )
    parser.add_argument(
        "--dump-fields",
        action="store_true",
        help="Save native with-minus-without fields for score residual distillation.",
    )
    parser.add_argument(
        "--dump-latents",
        action="store_true",
        help="With --dump-fields, save the exact z_t used for each native field.",
    )
    return parser.parse_args()


def parse_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def install_modelscope_stub() -> None:
    if "modelscope" in sys.modules:
        return
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("modelscope is disabled; this probe requires local weights")
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


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).item())


def pair_metrics(fields: list[torch.Tensor]) -> dict[str, float]:
    if len(fields) < 2:
        return {
            "mean_pairwise_cosine": 1.0,
            "mean_pairwise_relative_l2": 0.0,
            "between_energy_fraction": 0.0,
            "mean_field_norm": tensor_norm(fields[0]) if fields else 0.0,
        }

    cosines: list[float] = []
    relative_l2: list[float] = []
    for left, right in combinations(fields, 2):
        left_flat = left.float().reshape(-1)
        right_flat = right.float().reshape(-1)
        cosines.append(float(F.cosine_similarity(left_flat, right_flat, dim=0).item()))
        denom = 0.5 * (
            torch.linalg.vector_norm(left_flat) + torch.linalg.vector_norm(right_flat)
        )
        relative_l2.append(
            float((torch.linalg.vector_norm(left_flat - right_flat) / denom.clamp_min(1e-12)).item())
        )

    stacked = torch.stack([field.float() for field in fields], dim=0)
    mean_field = stacked.mean(dim=0, keepdim=True)
    between = (stacked - mean_field).square().mean()
    total = stacked.square().mean().clamp_min(1e-12)
    return {
        "mean_pairwise_cosine": sum(cosines) / len(cosines),
        "mean_pairwise_relative_l2": sum(relative_l2) / len(relative_l2),
        "between_energy_fraction": float((between / total).item()),
        "mean_field_norm": sum(tensor_norm(field) for field in fields) / len(fields),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latent_dump(z_t: torch.Tensor) -> dict[str, object]:
    value = z_t.detach().to(device="cpu", dtype=torch.float16).contiguous()
    return {
        "z_t": value,
        "z_t_shape": [int(v) for v in value.shape],
        "z_t_dtype": "float16",
    }


def pooled_reference_code(style_latent_4d: torch.Tensor) -> torch.Tensor:
    """Build a deterministic global image code from the frozen FLUX VAE latent."""

    if style_latent_4d.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] style latent, got {tuple(style_latent_4d.shape)}")
    tokens = style_latent_4d.float().flatten(2).transpose(1, 2)
    code = torch.cat([tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)], dim=-1)
    return code[0].detach().to(device="cpu", dtype=torch.float32).contiguous()


def main() -> int:
    args = parse_args()
    if args.dump_latents and not args.dump_fields:
        raise ValueError("--dump-latents requires --dump-fields")
    style_ids = parse_ids(args.style_ids)
    prompt_ids = parse_ids(args.prompt_ids)
    if len(style_ids) < 2 or len(prompt_ids) < 2:
        raise ValueError("At least two styles and two prompts are required")

    model_root = Path(args.model_root)
    style_root = Path(args.style_root)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    csv_path = output_prefix.with_suffix(".csv")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    if args.diffsynth_root not in sys.path:
        sys.path.insert(0, args.diffsynth_root)
    install_modelscope_stub()

    from PIL import Image
    from einops import rearrange
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from diffsynth.pipelines.flux2_image import (
        Flux2ImagePipeline,
        Flux2Unit_EditImageEmbedder,
        Flux2Unit_ImageIDs,
        Flux2Unit_PromptEmbedder,
        Flux2Unit_Qwen3PromptEmbedder,
        ModelConfig,
        model_fn_flux2,
    )

    set_seed(args.seed)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=sorted_files(str(model_root / "text_encoder" / "*.safetensors"))),
            ModelConfig(path=sorted_files(str(model_root / "transformer" / "*.safetensors"))),
            ModelConfig(path=str(model_root / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(model_root / "tokenizer")),
    )
    pipe.vram_management_enabled = False

    reference_conditions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    reference_codes: dict[str, torch.Tensor] = {}
    edit_unit = Flux2Unit_EditImageEmbedder()
    pipe.load_models_to_device(["vae"])
    for style_id in style_ids:
        style_path = style_root / f"{style_id}.png"
        if not style_path.exists():
            raise FileNotFoundError(style_path)
        image = Image.open(style_path).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.BICUBIC
        )
        with torch.no_grad():
            style_pixels = pipe.preprocess_image(image)
            style_latent_4d = pipe.vae.encode(style_pixels)
            encoded = edit_unit.process(pipe, image, edit_image_auto_resize=False)
        reference_conditions[style_id] = (
            encoded["edit_latents"].to(device="cuda", dtype=torch.bfloat16),
            encoded["edit_image_ids"].to(device="cuda"),
        )
        reference_codes[style_id] = pooled_reference_code(style_latent_4d)

    prompt_conditions: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
    if pipe.text_encoder_qwen3 is not None:
        pipe.load_models_to_device(["text_encoder_qwen3"])
        prompt_unit = Flux2Unit_Qwen3PromptEmbedder()
        text_encoder = pipe.text_encoder_qwen3
    else:
        pipe.load_models_to_device(["text_encoder"])
        prompt_unit = Flux2Unit_PromptEmbedder()
        text_encoder = pipe.text_encoder

    for prompt_id in prompt_ids:
        content = PROMPTS.get(prompt_id, prompt_id)
        prompt_text = args.prompt_template.format(content=content)
        embeds, text_ids = prompt_unit.encode_prompt(
            text_encoder,
            pipe.tokenizer,
            prompt_text,
            dtype=pipe.torch_dtype,
            device="cuda",
        )
        prompt_conditions[prompt_id] = (embeds, text_ids, prompt_text)

    lat_h, lat_w = args.height // 16, args.width // 16
    image_ids = Flux2Unit_ImageIDs().prepare_latent_ids(lat_h, lat_w).to("cuda")
    set_seed(args.seed)
    initial_noise = torch.randn(
        1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16
    )
    initial_latents = rearrange(initial_noise, "B C H W -> B (H W) C")

    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(args.num_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps.clone()

    pipe.load_models_to_device(["dit"])
    dit = pipe.dit.to("cuda")
    dit.requires_grad_(False)

    fields: dict[tuple[str, int, str], torch.Tensor] = {}
    normal_norms: dict[tuple[str, int], float] = {}
    rows: list[dict[str, object]] = []
    field_dump_dir = output_prefix.parent / f"{output_prefix.name}_fields"
    if args.dump_fields:
        field_dump_dir.mkdir(parents=True, exist_ok=True)

    for prompt_id in prompt_ids:
        prompt_embeds, text_ids, _ = prompt_conditions[prompt_id]
        latents = initial_latents.clone()
        for step_index, timestep_cpu in enumerate(timesteps):
            timestep = timestep_cpu.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                velocity_without = model_fn_flux2(
                    dit=dit,
                    latents=latents,
                    timestep=timestep,
                    embedded_guidance=args.embedded_guidance,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_ids=image_ids,
                )
            normal_norms[(prompt_id, step_index)] = tensor_norm(velocity_without)

            for style_id in style_ids:
                edit_latents, edit_ids = reference_conditions[style_id]
                with torch.no_grad():
                    velocity_with = model_fn_flux2(
                        dit=dit,
                        latents=latents,
                        timestep=timestep,
                        embedded_guidance=args.embedded_guidance,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        image_ids=image_ids,
                        edit_latents=edit_latents,
                        edit_image_ids=edit_ids,
                    )
                delta = (velocity_with - velocity_without).detach().float().cpu()
                sigma = scheduler.sigmas[step_index].detach().float().cpu()
                delta_score, score_coeff = velocity_delta_to_score_delta(delta, sigma)
                fields[(prompt_id, step_index, style_id)] = delta
                if args.dump_fields:
                    dump_payload = {
                        "style_id": style_id,
                        "style_path": str(style_root / f"{style_id}.png"),
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_conditions[prompt_id][2],
                        "timestep": float(timestep_cpu.item()),
                        "step_index": step_index,
                        "height": args.height,
                        "width": args.width,
                        "fields": {
                            "native_delta_velocity": delta,
                            "native_delta_score": delta_score,
                        },
                        "reference_code": reference_codes[style_id],
                        "reference_code_type": "flux2_vae_mean_std",
                        "sigma": float(sigma.item()),
                        "score_delta_coefficient": float(score_coeff.item()),
                        "teacher_type": "native_with_minus_without_score",
                        "model_root": str(model_root),
                        "seed": args.seed,
                        "num_steps": args.num_steps,
                    }
                    if args.dump_latents:
                        dump_payload["latents"] = latent_dump(latents)
                    dump_path = field_dump_dir / (
                        f"{output_prefix.name}_{prompt_id}_{style_id}_step{step_index}_t{int(round(float(timestep_cpu.item())))}.pt"
                    )
                    torch.save(dump_payload, dump_path)
                rows.append(
                    {
                        "axis": "field",
                        "prompt_id": prompt_id,
                        "style_id": style_id,
                        "step_index": step_index,
                        "timestep": float(timestep_cpu.item()),
                        "delta_norm": tensor_norm(delta),
                        "score_delta_norm": tensor_norm(delta_score),
                        "normal_velocity_norm": normal_norms[(prompt_id, step_index)],
                        "delta_to_normal_norm": tensor_norm(delta)
                        / max(normal_norms[(prompt_id, step_index)], 1e-12),
                    }
                )

            latents = scheduler.step(velocity_without, timestep_cpu, latents)

    reference_metrics: list[dict[str, object]] = []
    prompt_metrics: list[dict[str, object]] = []
    for step_index, timestep in enumerate(timesteps):
        for prompt_id in prompt_ids:
            metric = pair_metrics(
                [fields[(prompt_id, step_index, style_id)] for style_id in style_ids]
            )
            reference_metrics.append(
                {
                    "prompt_id": prompt_id,
                    "step_index": step_index,
                    "timestep": float(timestep.item()),
                    **metric,
                }
            )
        for style_id in style_ids:
            metric = pair_metrics(
                [fields[(prompt_id, step_index, style_id)] for prompt_id in prompt_ids]
            )
            prompt_metrics.append(
                {
                    "style_id": style_id,
                    "step_index": step_index,
                    "timestep": float(timestep.item()),
                    **metric,
                }
            )

    top_reference = [item for item in reference_metrics if item["step_index"] == 0]
    top_prompt = [item for item in prompt_metrics if item["step_index"] == 0]
    summary = {
        "method": "native_edit_with_minus_without_velocity",
        "teacher": "FLUX.2 Klein Base 9B native edit conditioning",
        "attention_kv_replacement_used": False,
        "same_state_contract": (
            "Within each prompt and step, with/without fields use identical target "
            "latent, timestep, text condition, model, and scheduler; only edit-image "
            "tokens differ. The no-reference branch advances the trajectory once."
        ),
        "model_root": str(model_root),
        "transformer_config_sha256": file_sha256(model_root / "transformer" / "config.json"),
        "scheduler_config_sha256": file_sha256(model_root / "scheduler" / "scheduler_config.json"),
        "style_ids": style_ids,
        "prompt_ids": prompt_ids,
        "prompt_texts": {
            key: value[2] for key, value in prompt_conditions.items()
        },
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "timesteps": [float(value.item()) for value in timesteps],
        "top_timestep_reference_metrics": top_reference,
        "top_timestep_prompt_metrics": top_prompt,
        "reference_metrics": reference_metrics,
        "prompt_metrics": prompt_metrics,
        "dump_fields": bool(args.dump_fields),
        "dump_latents": bool(args.dump_latents),
        "field_dump_dir": str(field_dump_dir) if args.dump_fields else None,
        "historical_kv_teacher_comparison": {
            "old_dynamic_reference_energy_fraction_at_t1000": 0.0,
            "source": "docs/klein9b_probe_notes.md reference-information onset audit",
        },
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps({"top_reference": top_reference, "top_prompt": top_prompt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
