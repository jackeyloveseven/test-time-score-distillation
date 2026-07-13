#!/usr/bin/env python3
"""Persistent full-fake critic trainer for bounded P0 replay plumbing."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ca_dmd_backend import (
    FakeConditioning,
    FakeSchedulerLookup,
    KleinFullFakeBackendConfig,
    build_fake_score_backend,
    clean_flow_target_from_noise,
    flow_match_add_noise,
    resolve_schedule_point,
)
from src.csv_rows import csv_fieldnames_for_rows
from src.flux2_score import velocity_delta_to_score_delta
from src.klein9b_loading import model_config_offload_kwargs
from src.native_edit_teacher import compute_native_velocity, freeze_condition_tensor, prepare_edit_condition
from src.persistent_replay import PersistentReplayBuffer, PersistentReplayEntry, validate_reference_hash_split


PROMPT_TEMPLATE = (
    "Create {content}. Use the visual style of the reference image, "
    "including its medium, palette, texture, and brushwork, but do not "
    "copy the reference objects or composition."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsynth-root", default="/mnt/local/yzh/prog/DiffSynth-Studio")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--fake-model-root", default="/mnt/local/klein9b-base")
    parser.add_argument("--teacher-model-root", default="/mnt/local/klein9b-base")
    parser.add_argument("--fake-vram-limit-gb", type=float, default=None)
    parser.add_argument("--teacher-vram-limit-gb", type=float, default=None)
    parser.add_argument("--fake-optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--fake-lr", type=float, default=1e-3)
    parser.add_argument("--fake-grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--fake-train-timestep-mode", default="uniform", choices=["current", "uniform"])
    parser.add_argument("--fake-train-seed", type=int, default=1064)
    parser.add_argument("--replay-mode", default="replay", choices=["replay", "no_replay"])
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--updates-per-sample", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--replay-capacity", type=int, default=256)
    parser.add_argument("--replay-seed", type=int, default=20260713)
    parser.add_argument("--heldout-eval-seed", type=int, default=20260714)
    parser.add_argument("--scoring-seed", type=int, default=20260715)
    parser.add_argument("--scoring-timestep-mode", default="uniform", choices=["uniform", "fixed"])
    parser.add_argument("--scoring-fixed-timestep", type=float, default=500.0)
    parser.add_argument("--default-reference-timestep", type=float, default=500.0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=64)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--skip-checkpoint-save", action="store_true")
    parser.add_argument("--resume-trainer-state-path", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-prefix", default="persistent_fake_critic")
    return parser.parse_args()


def install_modelscope_stub() -> None:
    if "modelscope" in sys.modules:
        return
    import types

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


def manifest_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}:{stat.st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def jsonl_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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
            ModelConfig(path=sorted_files(str(model_root / "text_encoder" / "*.safetensors")), **offload_kwargs),
            ModelConfig(path=sorted_files(str(model_root / "transformer" / "*.safetensors")), **offload_kwargs),
            ModelConfig(path=str(model_root / "vae" / "diffusion_pytorch_model.safetensors"), **offload_kwargs),
        ],
        tokenizer_config=ModelConfig(path=str(model_root / "tokenizer")),
        vram_limit=vram_limit_gb,
    )
    pipe.vram_management_enabled = vram_limit_gb is not None
    return pipe


def encode_prompt_for_pipe(pipe, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
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
    return (
        freeze_condition_tensor(prompt_embeds, device="cuda", dtype=pipe.torch_dtype),
        freeze_condition_tensor(text_ids, device="cuda"),
    )


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


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    required = {
        "sample_id",
        "split",
        "source_dataset",
        "clean_latent_path",
        "reference_path",
        "reference_sha256",
        "content_prompt",
        "prompt_id",
        "seed",
        "main_model_manifest_sha256",
        "teacher_model_manifest_sha256",
        "fake_init_model_manifest_sha256",
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{path}:{line_no} missing required fields: {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any rows")
    return rows


def load_clean_latent_export(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "clean_latent" not in payload or "metadata" not in payload:
        raise ValueError(f"{path} is not a valid clean-latent export payload")
    return payload


def resolve_prompt_text(row: dict[str, Any], export_meta: dict[str, Any]) -> str:
    prompt_text = row.get("prompt_text") or export_meta.get("prompt_text")
    if prompt_text:
        return str(prompt_text)
    content_prompt = row.get("content_prompt") or export_meta.get("content_prompt_text")
    if not content_prompt:
        raise ValueError(
            f"sample_id={row.get('sample_id')} must provide prompt_text or content_prompt/content_prompt_text"
        )
    return PROMPT_TEMPLATE.format(content=str(content_prompt))


def resolve_latent_hw(row: dict[str, Any], export_meta: dict[str, Any], clean_latent: torch.Tensor) -> tuple[int, int]:
    latent_h = row.get("latent_h", export_meta.get("latent_h"))
    latent_w = row.get("latent_w", export_meta.get("latent_w"))
    if latent_h is not None and latent_w is not None:
        return int(latent_h), int(latent_w)
    token_count = int(clean_latent.shape[1])
    side = int(round(token_count ** 0.5))
    if side * side != token_count:
        raise ValueError(
            f"Unable to infer latent_h/latent_w from token_count={token_count}; "
            "store latent_h/latent_w in the export metadata."
        )
    return side, side


def conditioning_key(
    *,
    prompt_text: str,
    reference_path: str,
    reference_sha256: str,
    latent_h: int,
    latent_w: int,
) -> tuple[str, str, str, int, int]:
    return (prompt_text, reference_path, reference_sha256, latent_h, latent_w)


def build_conditioning_cache(pipe, rows: list[dict[str, Any]], *, role: str) -> dict[tuple[str, str, str, int, int], FakeConditioning]:
    from diffsynth.pipelines.flux2_image import Flux2Unit_ImageIDs

    cache: dict[tuple[str, str, str, int, int], FakeConditioning] = {}
    image_id_unit = Flux2Unit_ImageIDs()
    for row in rows:
        export_payload = load_clean_latent_export(Path(row["clean_latent_path"]).expanduser().resolve())
        clean_latent = export_payload["clean_latent"]
        export_meta = export_payload["metadata"]
        prompt_text = resolve_prompt_text(row, export_meta)
        reference_path = str(row["reference_path"])
        reference_sha256 = str(row["reference_sha256"])
        latent_h, latent_w = resolve_latent_hw(row, export_meta, clean_latent)
        width = int(row.get("width", export_meta.get("width")))
        height = int(row.get("height", export_meta.get("height")))
        key = conditioning_key(
            prompt_text=prompt_text,
            reference_path=reference_path,
            reference_sha256=reference_sha256,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        if key in cache:
            continue
        prompt_embeds, text_ids = encode_prompt_for_pipe(pipe, prompt_text)
        image_ids = freeze_condition_tensor(image_id_unit.prepare_latent_ids(latent_h, latent_w).to("cuda"), device="cuda")
        reference_image = Image.open(reference_path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        edit_condition = prepare_edit_condition(pipe, reference_image, auto_resize=False)
        cache[key] = FakeConditioning(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
            edit_condition=edit_condition,
        )
    print(f"[startup] built {len(cache)} {role} conditioning entries", flush=True)
    return cache


def build_replay_entries(
    rows: list[dict[str, Any]],
    *,
    fake_conditioning_cache: dict[tuple[str, str, str, int, int], FakeConditioning],
) -> list[PersistentReplayEntry]:
    entries: list[PersistentReplayEntry] = []
    for row in rows:
        export_payload = load_clean_latent_export(Path(row["clean_latent_path"]).expanduser().resolve())
        clean_latent = export_payload["clean_latent"]
        export_meta = export_payload["metadata"]
        prompt_text = resolve_prompt_text(row, export_meta)
        reference_path = str(row["reference_path"])
        reference_sha256 = str(row["reference_sha256"])
        latent_h, latent_w = resolve_latent_hw(row, export_meta, clean_latent)
        key = conditioning_key(
            prompt_text=prompt_text,
            reference_path=reference_path,
            reference_sha256=reference_sha256,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        if key not in fake_conditioning_cache:
            raise KeyError(f"Missing fake conditioning cache for sample_id={row['sample_id']}")
        entries.append(
            PersistentReplayEntry.from_tensors(
                sample_id=str(row["sample_id"]),
                split=str(row["split"]),
                source_dataset=str(row["source_dataset"]),
                reference_path=reference_path,
                reference_sha256=reference_sha256,
                content_prompt=str(row["content_prompt"]),
                prompt_id=str(row["prompt_id"]),
                seed=int(row["seed"]),
                main_model_manifest_sha256=str(row["main_model_manifest_sha256"]),
                teacher_model_manifest_sha256=str(row["teacher_model_manifest_sha256"]),
                fake_init_model_manifest_sha256=str(row["fake_init_model_manifest_sha256"]),
                clean_latent=clean_latent,
                conditioning=fake_conditioning_cache[key],
                prompt_text=prompt_text,
                style_id=row.get("style_id"),
                clean_latent_sha256=row.get("clean_latent_sha256") or export_meta.get("tensor_sha256"),
                originating_outer_step=row.get("originating_outer_step"),
                reference_timestep=row.get("reference_timestep"),
            )
        )
    return entries


def validate_manifest_latent_contract(rows: list[dict[str, Any]]) -> dict[str, int]:
    latent_shapes: set[tuple[int, int]] = set()
    image_shapes: set[tuple[int, int]] = set()
    for row in rows:
        export_payload = load_clean_latent_export(Path(row["clean_latent_path"]).expanduser().resolve())
        clean_latent = export_payload["clean_latent"]
        export_meta = export_payload["metadata"]
        latent_h, latent_w = resolve_latent_hw(row, export_meta, clean_latent)
        height = int(row.get("height", export_meta.get("height", latent_h * 8)))
        width = int(row.get("width", export_meta.get("width", latent_w * 8)))
        latent_shapes.add((latent_h, latent_w))
        image_shapes.add((height, width))
    if len(latent_shapes) != 1:
        raise ValueError(f"Manifest must keep one latent_h/latent_w contract, got {sorted(latent_shapes)}")
    if len(image_shapes) != 1:
        raise ValueError(f"Manifest must keep one image height/width contract, got {sorted(image_shapes)}")
    latent_h, latent_w = next(iter(latent_shapes))
    height, width = next(iter(image_shapes))
    return {
        "latent_h": int(latent_h),
        "latent_w": int(latent_w),
        "height": int(height),
        "width": int(width),
    }


def _sample_schedule_point(
    *,
    scheduler_lookup: FakeSchedulerLookup,
    ref_tensor: torch.Tensor,
    generator: torch.Generator,
    mode: str,
    fixed_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if mode == "fixed":
        point = resolve_schedule_point(
            scheduler_lookup=scheduler_lookup,
            timestep=torch.tensor([float(fixed_timestep)]),
            ref_tensor=ref_tensor,
        )
        return point.timestep, point.sigma, point.timestep_index
    sampled_index = int(
        torch.randint(
            low=0,
            high=len(scheduler_lookup.timesteps),
            size=(1,),
            generator=generator,
            dtype=torch.int64,
        ).item()
    )
    timestep = scheduler_lookup.timesteps[sampled_index].to(device=ref_tensor.device, dtype=ref_tensor.dtype).reshape(1)
    sigma = scheduler_lookup.sigmas[sampled_index].to(device=ref_tensor.device, dtype=ref_tensor.dtype).reshape(1)
    return timestep, sigma, sampled_index


def _sample_noise_like(latents: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32)
    return noise.to(device=latents.device, dtype=latents.dtype)


def evaluate_single_entry(
    *,
    entry: PersistentReplayEntry,
    fake_backend,
    teacher_dit,
    teacher_conditioning: FakeConditioning,
    scheduler_lookup: FakeSchedulerLookup,
    scoring_generator: torch.Generator,
    scoring_timestep_mode: str,
    scoring_fixed_timestep: float,
    device: str,
    dtype: torch.dtype,
    scoring_point: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> dict[str, float]:
    x0 = entry.clean_latent_cpu.to(device=device, dtype=dtype).detach()
    if scoring_point is None:
        timestep, sigma, timestep_index = _sample_schedule_point(
            scheduler_lookup=scheduler_lookup,
            ref_tensor=x0,
            generator=scoring_generator,
            mode=scoring_timestep_mode,
            fixed_timestep=scoring_fixed_timestep,
        )
    else:
        timestep, sigma, timestep_index = scoring_point
    noise = _sample_noise_like(x0, generator=scoring_generator)
    z_sigma = flow_match_add_noise(x0, noise, sigma)
    target_v = clean_flow_target_from_noise(x0, noise)
    fake_conditioning = entry.conditioning.to_fake_conditioning(device=device, dtype=dtype)
    fake_v = fake_backend.predict_velocity(
        z_sigma=z_sigma,
        timestep=timestep,
        scoring_sigma=sigma,
        cond={"conditioning": fake_conditioning},
    )
    with torch.no_grad():
        teacher_v = compute_native_velocity(
            dit=teacher_dit,
            latents=z_sigma,
            timestep=timestep,
            embedded_guidance=0.0,
            prompt_embeds=teacher_conditioning.prompt_embeds,
            text_ids=teacher_conditioning.text_ids,
            image_ids=teacher_conditioning.image_ids,
            edit_condition=teacher_conditioning.edit_condition,
            no_grad=True,
        ).detach()
    delta_dm_score, _ = velocity_delta_to_score_delta(
        teacher_v - fake_v,
        sigma,
    )
    return {
        "eval_loss": float(F.mse_loss(fake_v.float(), target_v.float(), reduction="mean").item()),
        "eval_dm_score_rms": float(delta_dm_score.detach().float().square().mean().sqrt().item()),
        "eval_timestep": float(timestep.detach().item()),
        "eval_timestep_index": float(timestep_index),
        "eval_sigma": float(sigma.detach().item()),
    }


def evaluate_entries(
    *,
    entries: list[PersistentReplayEntry],
    fake_backend,
    teacher_dit,
    teacher_conditioning_by_sample_id: dict[str, FakeConditioning],
    scheduler_lookup: FakeSchedulerLookup,
    eval_seed: int,
    scoring_timestep_mode: str,
    scoring_fixed_timestep: float,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(eval_seed))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.sample_id not in teacher_conditioning_by_sample_id:
            raise KeyError(f"Missing teacher conditioning for sample_id={entry.sample_id}")
        metrics = evaluate_single_entry(
            entry=entry,
            fake_backend=fake_backend,
            teacher_dit=teacher_dit,
            teacher_conditioning=teacher_conditioning_by_sample_id[entry.sample_id],
            scheduler_lookup=scheduler_lookup,
            scoring_generator=generator,
            scoring_timestep_mode=scoring_timestep_mode,
            scoring_fixed_timestep=scoring_fixed_timestep,
            device=device,
            dtype=dtype,
        )
        rows.append(
            {
                "sample_id": entry.sample_id,
                "split": entry.split,
                "reference_sha256": entry.reference_sha256,
                "prompt_id": entry.prompt_id,
                **metrics,
            }
        )
    if not rows:
        return {"count": 0, "loss_mean": None, "dm_score_rms_mean": None, "rows": []}
    return {
        "count": len(rows),
        "loss_mean": float(sum(row["eval_loss"] for row in rows) / len(rows)),
        "dm_score_rms_mean": float(sum(row["eval_dm_score_rms"] for row in rows) / len(rows)),
        "rows": rows,
    }


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    out_prefix: str,
    update_count: int,
    fake_backend,
    replay_buffer: PersistentReplayBuffer,
    next_train_index: int,
    next_sample_update_index: int,
    manifest_path: Path,
    manifest_sha: str,
    replay_mode: str,
    scoring_rng: torch.Generator,
    update_rows: list[dict[str, Any]],
) -> dict[str, str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fake_state_path = checkpoint_dir / f"{out_prefix}_fake_backend_u{update_count}.pt"
    trainer_state_path = checkpoint_dir / f"{out_prefix}_trainer_state_u{update_count}.pt"
    fake_backend.save_state(fake_state_path)
    torch.save(
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "replay_mode": replay_mode,
            "next_train_index": int(next_train_index),
            "next_sample_update_index": int(next_sample_update_index),
            "update_count": int(update_count),
            "fake_backend_state_path": str(fake_state_path),
            "replay_buffer": replay_buffer.state_dict(),
            "scoring_rng_state": scoring_rng.get_state(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "update_rows": update_rows,
        },
        trainer_state_path,
    )
    return {
        "fake_backend_state_path": str(fake_state_path),
        "trainer_state_path": str(trainer_state_path),
    }


def skipped_checkpoint_record(
    *,
    checkpoint_dir: Path,
    update_count: int,
    reason: str,
) -> dict[str, str]:
    return {
        "fake_backend_state_path": "",
        "trainer_state_path": "",
        "checkpoint_dir": str(checkpoint_dir),
        "update_count": str(update_count),
        "skipped": "true",
        "reason": reason,
    }


def main() -> int:
    args = parse_args()
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if args.updates_per_sample <= 0:
        raise ValueError("--updates-per-sample must be positive")
    if args.checkpoint_every_updates <= 0:
        raise ValueError("--checkpoint-every-updates must be positive")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    if args.diffsynth_root not in sys.path:
        sys.path.insert(0, args.diffsynth_root)
    install_modelscope_stub()

    set_seed(args.fake_train_seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    manifest_path = Path(args.manifest_jsonl).expanduser().resolve()
    manifest_rows = load_manifest_rows(manifest_path)
    manifest_sha = jsonl_sha256(manifest_path)
    latent_contract = validate_manifest_latent_contract(manifest_rows)
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = f"{args.out_prefix}_{args.replay_mode}"
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else (results_dir / f"{out_prefix}_checkpoints")
    )

    trainer_state_payload = None
    if args.resume_trainer_state_path:
        trainer_state_payload = torch.load(
            Path(args.resume_trainer_state_path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if trainer_state_payload["manifest_sha256"] != manifest_sha:
            raise ValueError(
                "Resume manifest sha mismatch: "
                f"{trainer_state_payload['manifest_sha256']} != {manifest_sha}"
            )

    fake_model_root = Path(args.fake_model_root).resolve()
    teacher_model_root = Path(args.teacher_model_root).resolve()
    fake_pipe = load_flux2_pipe(fake_model_root, vram_limit_gb=args.fake_vram_limit_gb)
    teacher_pipe = load_flux2_pipe(teacher_model_root, vram_limit_gb=args.teacher_vram_limit_gb)

    fake_conditioning_cache = build_conditioning_cache(fake_pipe, manifest_rows, role="fake")
    teacher_conditioning_cache = build_conditioning_cache(teacher_pipe, manifest_rows, role="teacher")

    fake_aux_offloaded = unload_pipe_auxiliary_modules_to_cpu(fake_pipe)
    teacher_aux_offloaded = unload_pipe_auxiliary_modules_to_cpu(teacher_pipe)
    fake_pipe.load_models_to_device(["dit"])
    teacher_pipe.load_models_to_device(["dit"])
    fake_dit = fake_pipe.dit.to("cuda")
    teacher_dit = teacher_pipe.dit.to("cuda")
    fake_dit.requires_grad_(True)
    fake_dit.train()
    teacher_dit.requires_grad_(False)
    teacher_dit.eval()

    from diffsynth.diffusion.flow_match import FlowMatchScheduler

    fake_scheduler = FlowMatchScheduler("FLUX.2")
    fake_scheduler.set_timesteps(
        args.num_steps,
        dynamic_shift_len=int(latent_contract["latent_h"] * latent_contract["latent_w"]),
    )
    scheduler_lookup = FakeSchedulerLookup(
        timesteps=fake_scheduler.timesteps.detach().clone(),
        sigmas=fake_scheduler.sigmas.detach().clone(),
    )

    fake_backend = build_fake_score_backend(
        "klein_full_fake",
        fake_dit=fake_dit,
        conditioning=None,
        scheduler_lookup=scheduler_lookup,
        config=KleinFullFakeBackendConfig(
            fake_model_root=str(fake_model_root),
            optimizer=args.fake_optimizer,
            lr=float(args.fake_lr),
            grad_clip_norm=args.fake_grad_clip_norm,
            updates_per_inner=1,
            train_timestep_mode=args.fake_train_timestep_mode,
            train_seed=int(args.fake_train_seed),
            init_checkpoint_root=str(fake_model_root),
            resume_state_path=(
                trainer_state_payload["fake_backend_state_path"]
                if trainer_state_payload is not None
                else None
            ),
            experimental_scope="plumbing_only",
        ),
    )
    if fake_backend is None:
        raise RuntimeError("Failed to construct persistent fake backend")

    entries = build_replay_entries(
        manifest_rows,
        fake_conditioning_cache=fake_conditioning_cache,
    )
    split_contract = validate_reference_hash_split(entries)
    teacher_conditioning_by_sample_id: dict[str, FakeConditioning] = {}
    for row, entry in zip(manifest_rows, entries):
        export_payload = load_clean_latent_export(Path(row["clean_latent_path"]).expanduser().resolve())
        clean_latent = export_payload["clean_latent"]
        export_meta = export_payload["metadata"]
        prompt_text = resolve_prompt_text(row, export_meta)
        reference_path = str(row["reference_path"])
        reference_sha256 = str(row["reference_sha256"])
        latent_h, latent_w = resolve_latent_hw(row, export_meta, clean_latent)
        key = conditioning_key(
            prompt_text=prompt_text,
            reference_path=reference_path,
            reference_sha256=reference_sha256,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        teacher_conditioning_by_sample_id[entry.sample_id] = teacher_conditioning_cache[key]

    train_entries = [entry for entry in entries if entry.split == "train"]
    heldout_entries = [entry for entry in entries if entry.split == "heldout"]
    replay_buffer = (
        PersistentReplayBuffer.from_state_dict(trainer_state_payload["replay_buffer"])
        if trainer_state_payload is not None
        else PersistentReplayBuffer(capacity=args.replay_capacity, seed=args.replay_seed)
    )
    scoring_rng = torch.Generator(device="cpu")
    if trainer_state_payload is not None:
        scoring_rng.set_state(trainer_state_payload["scoring_rng_state"])
        torch.set_rng_state(trainer_state_payload["cpu_rng_state"])
        if torch.cuda.is_available() and trainer_state_payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(trainer_state_payload["cuda_rng_state_all"])
    else:
        scoring_rng.manual_seed(int(args.scoring_seed))
    next_train_index = int(trainer_state_payload["next_train_index"]) if trainer_state_payload is not None else 0
    next_sample_update_index = int(trainer_state_payload.get("next_sample_update_index", 0)) if trainer_state_payload is not None else 0
    update_rows: list[dict[str, Any]] = list(trainer_state_payload.get("update_rows", [])) if trainer_state_payload is not None else []

    print(
        "[startup] "
        f"manifest={manifest_path} rows={len(entries)} train={len(train_entries)} heldout={len(heldout_entries)} "
        f"fake_model_root={fake_model_root} teacher_model_root={teacher_model_root} "
        f"replay_mode={args.replay_mode} updates_per_sample={args.updates_per_sample} "
        f"grad_accum_steps={args.grad_accum_steps} capacity={args.replay_capacity}",
        flush=True,
    )

    before_train_eval = evaluate_entries(
        entries=train_entries,
        fake_backend=fake_backend,
        teacher_dit=teacher_dit,
        teacher_conditioning_by_sample_id=teacher_conditioning_by_sample_id,
        scheduler_lookup=scheduler_lookup,
        eval_seed=args.heldout_eval_seed + 101,
        scoring_timestep_mode=args.scoring_timestep_mode,
        scoring_fixed_timestep=args.scoring_fixed_timestep,
        device="cuda",
        dtype=torch.bfloat16,
    )
    before_heldout_eval = evaluate_entries(
        entries=heldout_entries,
        fake_backend=fake_backend,
        teacher_dit=teacher_dit,
        teacher_conditioning_by_sample_id=teacher_conditioning_by_sample_id,
        scheduler_lookup=scheduler_lookup,
        eval_seed=args.heldout_eval_seed,
        scoring_timestep_mode=args.scoring_timestep_mode,
        scoring_fixed_timestep=args.scoring_fixed_timestep,
        device="cuda",
        dtype=torch.bfloat16,
    )

    checkpoint_records: list[dict[str, str]] = []
    total_update_count = int(fake_backend.summary_dict()["update_count"])
    default_timestep = torch.tensor([float(args.default_reference_timestep)], device="cuda", dtype=torch.bfloat16)
    for train_idx in range(next_train_index, len(train_entries)):
        latest_entry = train_entries[train_idx]
        sample_start_update_idx = next_sample_update_index if train_idx == next_train_index else 0
        if sample_start_update_idx == 0:
            replay_buffer.add(latest_entry)
        for sample_update_idx in range(sample_start_update_idx, args.updates_per_sample):
            batch_entries = replay_buffer.build_accumulation_entries(
                latest_entry=latest_entry,
                grad_accum_steps=args.grad_accum_steps,
                replay_mode=args.replay_mode,
                split="train",
            )
            train_entries_gpu = [
                entry.to_train_entry(
                    device="cuda",
                    dtype=torch.bfloat16,
                    default_timestep=default_timestep,
                )
                for entry in batch_entries
            ]
            scoring_timestep, scoring_sigma, scoring_index = _sample_schedule_point(
                scheduler_lookup=scheduler_lookup,
                ref_tensor=train_entries_gpu[0].fake_latents_clean,
                generator=scoring_rng,
                mode=args.scoring_timestep_mode,
                fixed_timestep=args.scoring_fixed_timestep,
            )
            backend_log = fake_backend.train_step(
                fake_latents_clean=train_entries_gpu[0].fake_latents_clean,
                z_sigma=torch.zeros_like(train_entries_gpu[0].fake_latents_clean),
                timestep=train_entries_gpu[0].timestep,
                scoring_sigma=scoring_sigma,
                cond={"train_entries": train_entries_gpu},
            )
            latest_eval = evaluate_single_entry(
                entry=latest_entry,
                fake_backend=fake_backend,
                teacher_dit=teacher_dit,
                teacher_conditioning=teacher_conditioning_by_sample_id[latest_entry.sample_id],
                scheduler_lookup=scheduler_lookup,
                scoring_generator=scoring_rng,
                scoring_timestep_mode=args.scoring_timestep_mode,
                scoring_fixed_timestep=args.scoring_fixed_timestep,
                device="cuda",
                dtype=torch.bfloat16,
                scoring_point=(scoring_timestep, scoring_sigma, scoring_index),
            )
            total_update_count += 1
            update_rows.append(
                {
                    "global_update_index": total_update_count,
                    "train_sample_index": train_idx,
                    "sample_update_index": sample_update_idx,
                    "sample_id": latest_entry.sample_id,
                    "reference_sha256": latest_entry.reference_sha256,
                    "prompt_id": latest_entry.prompt_id,
                    "buffer_size": len(replay_buffer),
                    "batch_sample_ids": [entry.sample_id for entry in batch_entries],
                    "scoring_timestep_for_log": float(scoring_timestep.item()),
                    "scoring_timestep_index_for_log": float(scoring_index),
                    "scoring_sigma_for_log": float(scoring_sigma.item()),
                    **backend_log,
                    "latest_eval_loss": latest_eval["eval_loss"],
                    "latest_eval_dm_score_rms": latest_eval["eval_dm_score_rms"],
                    "latest_eval_timestep": latest_eval["eval_timestep"],
                    "latest_eval_timestep_index": latest_eval["eval_timestep_index"],
                    "latest_eval_sigma": latest_eval["eval_sigma"],
                }
            )
            if (not args.skip_checkpoint_save) and total_update_count % int(args.checkpoint_every_updates) == 0:
                resume_train_index = train_idx
                resume_sample_update_index = sample_update_idx + 1
                if resume_sample_update_index >= args.updates_per_sample:
                    resume_train_index = train_idx + 1
                    resume_sample_update_index = 0
                checkpoint_records.append(
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        out_prefix=out_prefix,
                        update_count=total_update_count,
                        fake_backend=fake_backend,
                        replay_buffer=replay_buffer,
                        next_train_index=resume_train_index,
                        next_sample_update_index=resume_sample_update_index,
                        manifest_path=manifest_path,
                        manifest_sha=manifest_sha,
                        replay_mode=args.replay_mode,
                        scoring_rng=scoring_rng,
                        update_rows=update_rows,
                    )
                )
        next_sample_update_index = 0

    if args.skip_checkpoint_save:
        final_checkpoint = skipped_checkpoint_record(
            checkpoint_dir=checkpoint_dir,
            update_count=total_update_count,
            reason="skip_checkpoint_save",
        )
    else:
        final_checkpoint = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            out_prefix=out_prefix,
            update_count=total_update_count,
            fake_backend=fake_backend,
            replay_buffer=replay_buffer,
            next_train_index=len(train_entries),
            next_sample_update_index=0,
            manifest_path=manifest_path,
            manifest_sha=manifest_sha,
            replay_mode=args.replay_mode,
            scoring_rng=scoring_rng,
            update_rows=update_rows,
        )
    checkpoint_records.append(final_checkpoint)

    after_train_eval = evaluate_entries(
        entries=train_entries,
        fake_backend=fake_backend,
        teacher_dit=teacher_dit,
        teacher_conditioning_by_sample_id=teacher_conditioning_by_sample_id,
        scheduler_lookup=scheduler_lookup,
        eval_seed=args.heldout_eval_seed + 101,
        scoring_timestep_mode=args.scoring_timestep_mode,
        scoring_fixed_timestep=args.scoring_fixed_timestep,
        device="cuda",
        dtype=torch.bfloat16,
    )
    after_heldout_eval = evaluate_entries(
        entries=heldout_entries,
        fake_backend=fake_backend,
        teacher_dit=teacher_dit,
        teacher_conditioning_by_sample_id=teacher_conditioning_by_sample_id,
        scheduler_lookup=scheduler_lookup,
        eval_seed=args.heldout_eval_seed,
        scoring_timestep_mode=args.scoring_timestep_mode,
        scoring_fixed_timestep=args.scoring_fixed_timestep,
        device="cuda",
        dtype=torch.bfloat16,
    )

    csv_path = results_dir / f"{out_prefix}.csv"
    summary_path = results_dir / f"{out_prefix}_summary.json"
    write_csv(update_rows, csv_path)
    summary = {
        "manifest_jsonl": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_row_count": len(entries),
        "latent_contract": latent_contract,
        "scheduler_contract": {
            "scheduler_name": "FLUX.2",
            "num_steps": int(args.num_steps),
            "dynamic_shift_len": int(latent_contract["latent_h"] * latent_contract["latent_w"]),
            "scheduler_config_sha256": file_sha256(fake_model_root / "scheduler" / "scheduler_config.json"),
            "timesteps": [float(value) for value in fake_scheduler.timesteps.detach().cpu().tolist()],
            "sigmas": [float(value) for value in fake_scheduler.sigmas.detach().cpu().tolist()],
        },
        "split_contract": split_contract,
        "replay_mode": args.replay_mode,
        "updates_per_sample": args.updates_per_sample,
        "grad_accum_steps": args.grad_accum_steps,
        "replay_capacity": args.replay_capacity,
        "fake_model_root": str(fake_model_root),
        "teacher_model_root": str(teacher_model_root),
        "fake_model_manifest_sha256": manifest_sha256(fake_model_root),
        "teacher_model_manifest_sha256": manifest_sha256(teacher_model_root),
        "fake_model_load_count": 1,
        "teacher_model_load_count": 1,
        "fake_aux_modules_offloaded_to_cpu": fake_aux_offloaded,
        "teacher_aux_modules_offloaded_to_cpu": teacher_aux_offloaded,
        "fake_backend_contract": fake_backend.summary_dict(),
        "buffer_composition_final": replay_buffer.composition_summary(),
        "before_train_eval": before_train_eval,
        "before_heldout_eval": before_heldout_eval,
        "after_train_eval": after_train_eval,
        "after_heldout_eval": after_heldout_eval,
        "checkpoint_records": checkpoint_records,
        "final_checkpoint": final_checkpoint,
        "checkpoint_save_enabled": not args.skip_checkpoint_save,
        "update_count_total": total_update_count,
        "update_rows_count": len(update_rows),
        "csv_path": str(csv_path),
        "cuda_peak_memory_gb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else None
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved update log -> {csv_path}")
    print(f"Saved summary    -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
