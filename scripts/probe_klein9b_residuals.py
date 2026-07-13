"""Single-sample residual probe skeleton for Klein9B.

Default behavior is dry-run with synthetic tensors so it works without loading
DiffSynth/Klein9B. Real model path is used by --run-model --runtime-smoke-only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
import types
from pathlib import Path
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention_hooks import Klein9BCache, Klein9BController, register_klein9b_attn_hooks
from src.profiling import (
    compare_field_pairs,
    compare_fields,
    write_csv,
    write_json_summary,
    write_pairwise_csv,
    write_pairwise_json_summary,
)
from src.latent_ops import highpass_corrected_teacher
from src.klein9b_loading import model_config_offload_kwargs, model_loading_mode
from src.residuals import compute_normal_velocity, compute_styled_velocity
from src.adapter_runtime import ResidualAdapterRuntime

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Klein9B residual fields (dry-run synthetic default). "
            "Use --run-model --runtime-smoke-only for minimal real runtime check."
        )
    )
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Run real Klein9B runtime smoke path (placeholder if not paired with --runtime-smoke-only).",
    )
    parser.add_argument(
        "--runtime-smoke-only",
        action="store_true",
        help="With --run-model, only execute runtime smoke checks (no DiT forward, no decode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run synthetic dry-run path (default when --run-model is not provided).",
    )
    parser.add_argument(
        "--style-id",
        default="27",
        help="Style id.",
    )
    parser.add_argument(
        "--prompt-id",
        default="kp0",
        help="Prompt id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for synthetic fields and runtime noise.",
    )
    parser.add_argument(
        "--selected-timesteps",
        default="700,500,300",
        help="Comma-separated timesteps to sample, e.g. 700,500,300.",
    )
    parser.add_argument(
        "--out-prefix",
        default="residual_profile_klein9b_probe_dryrun",
        help="Output file prefix under results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for csv/json/meta outputs.",
    )
    parser.add_argument(
        "--no-hyperion-grad",
        action="store_true",
        help="Skip hyperion-style proxy teachers (compute only hard_scfg and otcfg).",
    )
    parser.add_argument(
        "--dump-fields",
        action="store_true",
        help="Save each timestep field dict to results/field_dumps/*.pt.",
    )
    parser.add_argument(
        "--dump-latents",
        action="store_true",
        help=(
            "With --dump-fields, also save the probed z_t latent for direct "
            "score/velocity residual student training."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Synthetic spatial height for dry-run field metrics.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="Synthetic spatial width for dry-run field metrics.",
    )
    parser.add_argument(
        "--runtime-height",
        type=int,
        default=512,
        help="Image height for real Klein9B runtime latent shape.",
    )
    parser.add_argument(
        "--runtime-width",
        type=int,
        default=512,
        help="Image width for real Klein9B runtime latent shape.",
    )
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        default=None,
        help="Optional DiffSynth VRAM limit in GB; enables low-VRAM model management.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=25,
        help="Number of scheduler timesteps for runtime smoke.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=128,
        help="Synthetic channel dimension.",
    )
    parser.add_argument(
        "--inject-beta",
        type=float,
        default=1.0,
        help="Reserved for next stage.",
    )
    parser.add_argument(
        "--sinkhorn-tau",
        type=float,
        default=0.02,
        help="Reserved for next stage.",
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=10,
        help="Reserved for next stage.",
    )
    parser.add_argument(
        "--self-layer-start",
        type=int,
        default=12,
        help="First hooked self-attention layer index for residual profiling.",
    )
    parser.add_argument(
        "--self-layer-end",
        type=int,
        default=28,
        help="Exclusive end hooked self-attention layer index for residual profiling.",
    )
    parser.add_argument(
        "--teacher-lambdas",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated lambda values for hard + lambda * highpass(ot-hard).",
    )
    parser.add_argument(
        "--teacher-highpass-kernel",
        type=int,
        default=5,
        help="Odd spatial avg-pool kernel used to build highpass correction teachers.",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        default=None,
        help="Optional residual adapter checkpoint. Adds online student_scfg variants.",
    )
    parser.add_argument(
        "--adapter-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for the residual adapter runtime.",
    )
    parser.add_argument(
        "--student-scales",
        default="0.5,0.75,1.0",
        help="Comma-separated adapter scales for online student_scfg variants.",
    )
    parser.add_argument(
        "--student-main-scale",
        type=float,
        default=None,
        help=(
            "If set, use v_normal + hard_scfg + scale * adapter_delta as the selected-step "
            "scheduler prediction. Default only profiles student variants."
        ),
    )
    return parser.parse_args()


def parse_selected_timesteps(raw: str) -> List[float]:
    if not raw:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return []
    return [float(v) for v in values]


def parse_float_list(raw: str) -> List[float]:
    if not raw:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return []
    return [float(v) for v in values]


def _format_lambda(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def build_latent_dump(z_t: torch.Tensor) -> dict:
    """Pack the probed latent for direct score-residual distillation."""
    z_t_cpu = z_t.detach().to(device="cpu", dtype=torch.float16).contiguous()
    return {
        "z_t": z_t_cpu,
        "z_t_shape": [int(v) for v in z_t_cpu.shape],
        "z_t_dtype": "float16",
    }


def build_highpass_teacher_fields(
    hard_scfg: torch.Tensor,
    otcfg: torch.Tensor,
    height: int,
    width: int,
    lambdas: List[float],
    kernel_size: int,
) -> dict[str, torch.Tensor]:
    teachers: dict[str, torch.Tensor] = {}
    for lamb in lambdas:
        name = f"teacher_hp_lam{_format_lambda(lamb)}"
        teachers[name] = highpass_corrected_teacher(
            hard_scfg,
            otcfg,
            height=height,
            width=width,
            lamb=lamb,
            kernel_size=kernel_size,
        )
    return teachers


def build_student_adapter_fields(
    hard_scfg: torch.Tensor,
    adapter_delta: torch.Tensor,
    scales: List[float],
) -> dict[str, torch.Tensor]:
    fields: dict[str, torch.Tensor] = {}
    for scale in scales:
        name = f"student_scfg_s{_format_lambda(scale)}"
        fields[name] = hard_scfg + float(scale) * adapter_delta
    return fields


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _synthesize_field(
    shape: tuple[int, int, int],
    reference: Optional[torch.Tensor] = None,
    mix_ratio: float = 0.6,
) -> torch.Tensor:
    base = torch.randn(shape)
    if reference is None:
        return base

    ref = reference / (reference.norm(dim=-1, keepdim=True) + 1e-8)
    proj = (base * ref).sum(dim=-1, keepdim=True)
    tangent = base - proj * ref
    return mix_ratio * ref + (1.0 - mix_ratio) * tangent


def run_dry_run(args: argparse.Namespace) -> int:
    _set_seed(args.seed)
    if args.dump_latents and not args.dump_fields:
        raise ValueError("--dump-latents requires --dump-fields")

    selected_timesteps = parse_selected_timesteps(args.selected_timesteps)
    if not selected_timesteps:
        raise ValueError("--selected-timesteps must contain at least one value")
    teacher_lambdas = parse_float_list(args.teacher_lambdas)

    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    dump_dir = result_dir / "field_dumps"
    if args.dump_fields:
        dump_dir.mkdir(parents=True, exist_ok=True)

    l = args.height * args.width
    shape = (1, l, args.d)
    records = []
    pairwise_records = []

    for t in selected_timesteps:
        hyperion_grad: Optional[torch.Tensor]
        hyperion_delta: Optional[torch.Tensor]
        z_probe = torch.randn(*shape)

        if args.no_hyperion_grad:
            hyperion_grad = None
            hyperion_delta = None
            teacher_type = "disabled"
        else:
            hyperion_grad = torch.randn(*shape)
            hyperion_delta = 0.7 * hyperion_grad + 0.2 * torch.randn(*shape)
            teacher_type = "hyperion_grad_proxy"

        hard_scfg = _synthesize_field(shape, reference=hyperion_grad, mix_ratio=0.82)
        otcfg = _synthesize_field(shape, reference=hyperion_grad, mix_ratio=0.66)
        lrs_otcfg = 0.8 * otcfg

        fields = {
            "hard_scfg": hard_scfg,
            "otcfg": otcfg,
        }
        teacher_fields = build_highpass_teacher_fields(
            hard_scfg=hard_scfg,
            otcfg=otcfg,
            height=args.height,
            width=args.width,
            lambdas=teacher_lambdas,
            kernel_size=args.teacher_highpass_kernel,
        )
        fields.update(teacher_fields)
        if not args.no_hyperion_grad:
            fields["lrs_otcfg"] = lrs_otcfg

        step_records = compare_fields(
            fields=fields,
            hyperion_grad=hyperion_grad,
            hyperion_delta=hyperion_delta,
            height=args.height,
            width=args.width,
            style_id=args.style_id,
            prompt_id=args.prompt_id,
            timestep=t,
        )
        records.extend(step_records)
        pairwise_pairs = [("hard_scfg", "otcfg")]
        for teacher_name in teacher_fields:
            pairwise_pairs.append(("hard_scfg", teacher_name))
            pairwise_pairs.append((teacher_name, "otcfg"))

        pairwise_records.extend(
            compare_field_pairs(
                fields=fields,
                pairs=pairwise_pairs,
                height=args.height,
                width=args.width,
                style_id=args.style_id,
                prompt_id=args.prompt_id,
                timestep=t,
            )
        )

        if args.dump_fields:
            dump_path = dump_dir / f"{args.out_prefix}_{args.prompt_id}_{args.style_id}_t{int(t)}.pt"
            dump_payload = {
                "style_id": args.style_id,
                "prompt_id": args.prompt_id,
                "timestep": t,
                "height": args.height,
                "width": args.width,
                "fields": fields,
                "teacher_type": teacher_type,
                "teacher_lambdas": teacher_lambdas,
                "teacher_highpass_kernel": args.teacher_highpass_kernel,
                "hyperion_grad_proxy": hyperion_grad,
                "hyperion_delta_proxy": hyperion_delta,
                "has_latents": bool(args.dump_latents),
            }
            if args.dump_latents:
                dump_payload["latents"] = build_latent_dump(z_probe)
            torch.save(dump_payload, dump_path)

    csv_path = result_dir / f"{args.out_prefix}.csv"
    summary_path = result_dir / f"{args.out_prefix}_summary.json"
    pairwise_csv_path = result_dir / f"{args.out_prefix}_pairwise.csv"
    pairwise_summary_path = result_dir / f"{args.out_prefix}_pairwise_summary.json"
    meta_path = result_dir / f"{args.out_prefix}_meta.json"

    write_csv(records, csv_path)
    summary = write_json_summary(records, summary_path)
    write_pairwise_csv(pairwise_records, pairwise_csv_path)
    pairwise_summary = write_pairwise_json_summary(pairwise_records, pairwise_summary_path)

    teacher_type = "hyperion_grad_proxy" if not args.no_hyperion_grad else "disabled"
    variants = (
        ["hard_scfg", "otcfg"]
        + [f"teacher_hp_lam{_format_lambda(lamb)}" for lamb in teacher_lambdas]
        + ([] if args.no_hyperion_grad else ["lrs_otcfg"])
    )

    meta = {
        "style_id": args.style_id,
        "prompt_id": args.prompt_id,
        "seed": args.seed,
        "selected_timesteps": selected_timesteps,
        "dry_run": True,
        "variants": variants,
        "teacher_type": teacher_type,
        "teacher_lambdas": teacher_lambdas,
        "teacher_highpass_kernel": args.teacher_highpass_kernel,
        "out_prefix": args.out_prefix,
        "results_dir": str(result_dir),
        "height": args.height,
        "width": args.width,
        "d": args.d,
        "num_records": len(records),
        "num_pairwise_records": len(pairwise_records),
        "dump_fields": bool(args.dump_fields),
        "dump_latents": bool(args.dump_latents),
        "summary": summary,
        "pairwise_summary": pairwise_summary,
        "pairwise_csv": str(pairwise_csv_path),
        "pairwise_summary_path": str(pairwise_summary_path),
        "inject_beta": args.inject_beta,
        "sinkhorn_tau": args.sinkhorn_tau,
        "sinkhorn_iters": args.sinkhorn_iters,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"OK (dry-run): wrote {len(records)} rows -> {csv_path}")
    print(f"OK: wrote summary -> {summary_path}")
    print(f"OK: wrote pairwise rows -> {pairwise_csv_path}")
    print(f"OK: wrote pairwise summary -> {pairwise_summary_path}")
    print(f"OK: wrote meta -> {meta_path}")
    if args.dump_fields:
        print(f"OK: dumped fields in {dump_dir}")
    return 0


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


def _to_list(value, max_items: int = 12):
    if value is None:
        return []
    if torch.is_tensor(value):
        t = value.view(-1)
        cnt = min(max_items, int(t.numel()))
        return [float(x) for x in t[:cnt].tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in list(value)[:max_items]]
    return [float(value)]


def run_model_smoke(args: argparse.Namespace) -> int:
    if not args.runtime_smoke_only:
        raise RuntimeError("--runtime-smoke-only is required for runtime smoke mode")

    style_id = str(args.style_id)
    prompt_text = PROMPTS.get(args.prompt_id, args.prompt_id)
    style_path = Path(STYLE_IMAGES_ROOT) / f"{style_id}.png"
    if not style_path.exists():
        raise FileNotFoundError(f"Style image not found: {style_path}")

    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    meta_path = result_dir / f"{args.out_prefix}_meta.json"

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    os.environ.setdefault("DIFFSYNTH_ROOT", DIFFSYNTH_ROOT)
    os.environ.setdefault("KLEIN9B_ROOT", KLEIN9B_ROOT)
    diffsynth_root = os.environ["DIFFSYNTH_ROOT"]
    model_root = os.environ["KLEIN9B_ROOT"]

    model_root_path = Path(model_root)
    if not model_root_path.exists():
        raise FileNotFoundError(f"Klein9B root not found: {model_root_path}")

    if str(diffsynth_root) not in sys.path:
        sys.path.insert(0, str(diffsynth_root))
    _install_modelscope_stub()

    from diffsynth.pipelines.flux2_image import (
        Flux2ImagePipeline,
        Flux2Unit_PromptEmbedder,
        Flux2Unit_Qwen3PromptEmbedder,
        ModelConfig,
    )
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from einops import rearrange
    from PIL import Image

    _set_seed(args.seed)
    print(f"Loading Klein9B from: {model_root_path}")
    print(f"Loading DiffSynth from: {diffsynth_root}")
    offload_kwargs = model_config_offload_kwargs(args.vram_limit_gb)
    loading_mode = model_loading_mode(args.vram_limit_gb)
    print(f"Model loading mode: {loading_mode}")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        vram_limit=args.vram_limit_gb,
        model_configs=[
            ModelConfig(
                path=_sorted_files(str(model_root_path / "text_encoder" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(
                path=_sorted_files(str(model_root_path / "transformer" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(path=str(model_root_path / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(model_root_path / "tokenizer")),
    )
    if args.vram_limit_gb is None:
        pipe.vram_management_enabled = False

    vae = pipe.vae
    scheduler = FlowMatchScheduler("FLUX.2")

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        style_img = pipe.preprocess_image(
            Image.open(style_path)
            .convert("RGB")
            .resize((args.runtime_width, args.runtime_height), Image.BICUBIC)
        )
        style_latent_4d = vae.encode(style_img)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

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

    lat_h = args.runtime_height // 16
    lat_w = args.runtime_width // 16
    noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    latents = rearrange(noise, "B C H W -> B (H W) C")

    scheduler.set_timesteps(args.num_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps
    requested_timesteps = parse_selected_timesteps(args.selected_timesteps)
    timesteps_all = [float(t.item()) for t in timesteps]
    actual_timesteps = [
        min(timesteps_all, key=lambda actual: abs(actual - requested))
        for requested in requested_timesteps
    ]

    meta = {
        "mode": "runtime_smoke",
        "style_id": style_id,
        "prompt_id": args.prompt_id,
        "prompt_text": prompt_text,
        "style_path": str(style_path),
        "seed": args.seed,
        "width": args.runtime_width,
        "height": args.runtime_height,
        "dry_run_width": args.width,
        "dry_run_height": args.height,
        "num_steps": args.num_steps,
        "vram_limit_gb": args.vram_limit_gb,
        "requested_timesteps": requested_timesteps,
        "actual_timesteps": actual_timesteps,
        "latent_shape": [int(v) for v in latents.shape],
        "style_latent_shape": [int(v) for v in style_latent.shape],
        "style_latent_4d_shape": [int(v) for v in style_latent_4d.shape],
        "timesteps_head": _to_list(timesteps),
        "timesteps_count": int(len(timesteps)),
        "prompt_embeds_shape": [int(v) for v in prompt_embeds.shape],
        "prompt_embed_shape": [int(v) for v in prompt_embeds.shape],
        "null_embeds_shape": [int(v) for v in null_embeds.shape],
        "null_prompt_embed_shape": [int(v) for v in null_embeds.shape],
        "text_ids_shape": [int(v) for v in text_ids.shape],
        "null_text_ids_shape": [int(v) for v in null_text_ids.shape],
        "model_root": str(model_root_path),
        "diffsynth_root": diffsynth_root,
        "invariant": (
            "Future residual probe must compute v_normal/v_styled on the same z_t,t "
            "and call scheduler.step at most once per main-path timestep."
        ),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"OK (runtime smoke): wrote meta -> {meta_path}")
    print(f"Timesteps: count={meta['timesteps_count']} head={meta['timesteps_head']}")
    print(f"Requested timesteps: {requested_timesteps}")
    print(f"Actual timesteps: {actual_timesteps}")
    print(f"Latent shape: {meta['latent_shape']}")
    print(f"Style latent shape: {meta['style_latent_shape']}")
    return 0


def run_model_profile(args: argparse.Namespace) -> int:
    if args.dump_latents and not args.dump_fields:
        raise ValueError("--dump-latents requires --dump-fields")

    style_id = str(args.style_id)
    prompt_text = PROMPTS.get(args.prompt_id, args.prompt_id)
    style_path = Path(STYLE_IMAGES_ROOT) / f"{style_id}.png"
    if not style_path.exists():
        raise FileNotFoundError(f"Style image not found: {style_path}")

    requested_timesteps = parse_selected_timesteps(args.selected_timesteps)
    if not requested_timesteps:
        raise ValueError("--selected-timesteps must contain at least one value")
    teacher_lambdas = parse_float_list(args.teacher_lambdas)
    student_scales = parse_float_list(args.student_scales) if args.adapter_checkpoint else []

    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = result_dir / "field_dumps"
    if args.dump_fields:
        dump_dir.mkdir(parents=True, exist_ok=True)

    csv_path = result_dir / f"{args.out_prefix}.csv"
    summary_path = result_dir / f"{args.out_prefix}_summary.json"
    pairwise_csv_path = result_dir / f"{args.out_prefix}_pairwise.csv"
    pairwise_summary_path = result_dir / f"{args.out_prefix}_pairwise_summary.json"
    meta_path = result_dir / f"{args.out_prefix}_meta.json"

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    os.environ.setdefault("DIFFSYNTH_ROOT", DIFFSYNTH_ROOT)
    os.environ.setdefault("KLEIN9B_ROOT", KLEIN9B_ROOT)
    diffsynth_root = os.environ["DIFFSYNTH_ROOT"]
    model_root = os.environ["KLEIN9B_ROOT"]

    model_root_path = Path(model_root)
    if not model_root_path.exists():
        raise FileNotFoundError(f"Klein9B root not found: {model_root_path}")

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

    _set_seed(args.seed)
    print(f"Loading Klein9B from: {model_root_path}")
    print(f"Loading DiffSynth from: {diffsynth_root}")
    offload_kwargs = model_config_offload_kwargs(args.vram_limit_gb)
    loading_mode = model_loading_mode(args.vram_limit_gb)
    print(f"Model loading mode: {loading_mode}")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        vram_limit=args.vram_limit_gb,
        model_configs=[
            ModelConfig(
                path=_sorted_files(str(model_root_path / "text_encoder" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(
                path=_sorted_files(str(model_root_path / "transformer" / "*.safetensors")),
                **offload_kwargs,
            ),
            ModelConfig(path=str(model_root_path / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(model_root_path / "tokenizer")),
    )
    if args.vram_limit_gb is None:
        pipe.vram_management_enabled = False

    vae = pipe.vae
    scheduler = FlowMatchScheduler("FLUX.2")

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        style_img = pipe.preprocess_image(
            Image.open(style_path)
            .convert("RGB")
            .resize((args.runtime_width, args.runtime_height), Image.BICUBIC)
        )
        style_latent_4d = vae.encode(style_img)
        style_latent = rearrange(style_latent_4d, "B C H W -> B (H W) C")

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

    lat_h = args.runtime_height // 16
    lat_w = args.runtime_width // 16
    noise = torch.randn(1, 128, lat_h, lat_w, device="cuda", dtype=torch.bfloat16)
    latents = rearrange(noise, "B C H W -> B (H W) C")

    image_id_unit = Flux2Unit_ImageIDs()
    image_ids = image_id_unit.prepare_latent_ids(lat_h, lat_w).to("cuda")
    style_lat_h = int(style_latent_4d.shape[-2])
    style_lat_w = int(style_latent_4d.shape[-1])
    style_image_ids = image_id_unit.prepare_latent_ids(style_lat_h, style_lat_w).to("cuda")

    scheduler.set_timesteps(args.num_steps, dynamic_shift_len=lat_h * lat_w)
    timesteps = scheduler.timesteps
    timesteps_all = [float(t.item()) for t in timesteps]
    selected_indices = sorted(
        {
            min(range(len(timesteps_all)), key=lambda idx: abs(timesteps_all[idx] - requested))
            for requested in requested_timesteps
        }
    )
    actual_timesteps = [timesteps_all[idx] for idx in selected_indices]

    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    dit.to("cuda")
    dit.requires_grad_(False)

    cache = Klein9BCache()
    controller = Klein9BController(self_layers=(args.self_layer_start, args.self_layer_end))
    register_klein9b_attn_hooks(dit, controller, cache)
    adapter_runtime = None
    if args.adapter_checkpoint:
        adapter_runtime = ResidualAdapterRuntime.from_checkpoint(
            args.adapter_checkpoint,
            device=args.adapter_device,
            student_scale=1.0,
        )
        print(
            f"Loaded residual adapter: type={adapter_runtime.adapter_type} "
            f"device={args.adapter_device} scales={student_scales}",
            flush=True,
        )

    records = []
    pairwise_records = []
    selected_done = 0
    print(f"Selected timestep indices: {selected_indices}")
    print(f"Actual timesteps: {actual_timesteps}")

    for i, t_val in enumerate(timesteps):
        timestep = t_val.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        is_selected = i in selected_indices

        if is_selected:
            z_probe = latents.detach().clone()
            style_timestep = timesteps[i:i + 1]
            sty_noise = torch.randn_like(style_latent)
            style_latent_t = scheduler.add_noise(style_latent, sty_noise, style_timestep)

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
                raise RuntimeError("No reference K/V tensors captured; check self layer range and hooks")

            controller.reset_mode()
            cache.clear()
            controller.step()
            v_normal = compute_normal_velocity(
                dit,
                latents=z_probe,
                timestep=timestep / 1000,
                guidance=None,
                prompt_embeds=prompt_embeds,
                txt_ids=text_ids,
                image_ids=image_ids,
            )
            v_hard = compute_styled_velocity(
                dit,
                latents=z_probe,
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
            v_ot = compute_styled_velocity(
                dit,
                latents=z_probe,
                timestep=timestep / 1000,
                guidance=None,
                prompt_embeds=prompt_embeds,
                txt_ids=text_ids,
                image_ids=image_ids,
                controller=controller,
                cache=cache,
                ref_k_list=ref_k_list,
                ref_v_list=ref_v_list,
                mode="sinkhorn",
                inject_beta=1.0,
                sinkhorn_tau=args.sinkhorn_tau,
                sinkhorn_iters=args.sinkhorn_iters,
            )

            hard_scfg = (v_hard - v_normal).detach().float().cpu()
            otcfg = (v_ot - v_normal).detach().float().cpu()
            fields = {"hard_scfg": hard_scfg, "otcfg": otcfg}
            teacher_fields = build_highpass_teacher_fields(
                hard_scfg=hard_scfg,
                otcfg=otcfg,
                height=lat_h,
                width=lat_w,
                lambdas=teacher_lambdas,
                kernel_size=args.teacher_highpass_kernel,
            )
            fields.update(teacher_fields)
            adapter_delta = None
            student_fields = {}
            if adapter_runtime is not None:
                adapter_delta = adapter_runtime.predict_delta(
                    hard_scfg,
                    timestep=float(t_val.item()),
                    style_id=style_id,
                    prompt_id=args.prompt_id,
                ).detach().float().cpu()
                student_fields = build_student_adapter_fields(
                    hard_scfg=hard_scfg,
                    adapter_delta=adapter_delta,
                    scales=student_scales,
                )
                fields.update(student_fields)
            step_records = compare_fields(
                fields=fields,
                hyperion_grad=None,
                hyperion_delta=None,
                height=lat_h,
                width=lat_w,
                style_id=style_id,
                prompt_id=args.prompt_id,
                timestep=float(t_val.item()),
            )
            records.extend(step_records)
            pairwise_pairs = [("hard_scfg", "otcfg")]
            for teacher_name in teacher_fields:
                pairwise_pairs.append(("hard_scfg", teacher_name))
                pairwise_pairs.append((teacher_name, "otcfg"))
            for student_name in student_fields:
                pairwise_pairs.append(("hard_scfg", student_name))
                pairwise_pairs.append((student_name, "otcfg"))
                if "teacher_hp_lam1" in teacher_fields:
                    pairwise_pairs.append((student_name, "teacher_hp_lam1"))
            pairwise_records.extend(
                compare_field_pairs(
                    fields=fields,
                    pairs=pairwise_pairs,
                    height=lat_h,
                    width=lat_w,
                    style_id=style_id,
                    prompt_id=args.prompt_id,
                    timestep=float(t_val.item()),
                )
            )

            if args.dump_fields:
                dump_path = dump_dir / f"{args.out_prefix}_{args.prompt_id}_{style_id}_t{int(float(t_val.item()))}.pt"
                dump_payload = {
                    "style_id": style_id,
                    "prompt_id": args.prompt_id,
                    "prompt_text": prompt_text,
                    "style_path": str(style_path),
                    "timestep": float(t_val.item()),
                    "height": lat_h,
                    "width": lat_w,
                    "fields": fields,
                    "teacher_lambdas": teacher_lambdas,
                    "teacher_highpass_kernel": args.teacher_highpass_kernel,
                    "ref_k_shapes": [[int(v) for v in x.shape] for x in ref_k_list],
                    "ref_v_shapes": [[int(v) for v in x.shape] for x in ref_v_list],
                    "has_latents": bool(args.dump_latents),
                }
                if args.dump_latents:
                    dump_payload["latents"] = build_latent_dump(z_probe)
                torch.save(dump_payload, dump_path)

            noise_pred = v_normal
            if adapter_delta is not None and args.student_main_scale is not None:
                student_residual = hard_scfg + float(args.student_main_scale) * adapter_delta
                noise_pred = v_normal + student_residual.to(device=v_normal.device, dtype=v_normal.dtype)
            selected_done += 1
            del v_hard, v_ot, hard_scfg, otcfg, ref_k_list, ref_v_list
            if adapter_delta is not None:
                del adapter_delta
        else:
            controller.reset_mode()
            cache.clear()
            controller.step()
            with torch.no_grad():
                noise_pred = dit(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                )

        latents = scheduler.step(noise_pred, t_val, latents)
        del noise_pred
        if selected_done >= len(selected_indices):
            break

    summary = write_json_summary(records, summary_path)
    write_csv(records, csv_path)
    write_pairwise_csv(pairwise_records, pairwise_csv_path)
    pairwise_summary = write_pairwise_json_summary(pairwise_records, pairwise_summary_path)

    meta = {
        "mode": "run_model",
        "style_id": style_id,
        "prompt_id": args.prompt_id,
        "prompt_text": prompt_text,
        "style_path": str(style_path),
        "seed": args.seed,
        "width": args.runtime_width,
        "height": args.runtime_height,
        "num_steps": args.num_steps,
        "vram_limit_gb": args.vram_limit_gb,
        "requested_timesteps": requested_timesteps,
        "selected_indices": selected_indices,
        "actual_timesteps": actual_timesteps,
        "variants": (
            ["hard_scfg", "otcfg"]
            + [f"teacher_hp_lam{_format_lambda(lamb)}" for lamb in teacher_lambdas]
            + [f"student_scfg_s{_format_lambda(scale)}" for scale in student_scales]
        ),
        "teacher_lambdas": teacher_lambdas,
        "teacher_highpass_kernel": args.teacher_highpass_kernel,
        "adapter_checkpoint": args.adapter_checkpoint,
        "adapter_device": args.adapter_device if args.adapter_checkpoint else None,
        "student_scales": student_scales,
        "student_main_scale": args.student_main_scale,
        "dump_fields": bool(args.dump_fields),
        "dump_latents": bool(args.dump_latents),
        "no_hyperion_grad": True,
        "inject_beta": args.inject_beta,
        "sinkhorn_tau": args.sinkhorn_tau,
        "sinkhorn_iters": args.sinkhorn_iters,
        "self_layers": [args.self_layer_start, args.self_layer_end],
        "latent_shape": [int(v) for v in latents.shape],
        "style_latent_shape": [int(v) for v in style_latent.shape],
        "style_latent_4d_shape": [int(v) for v in style_latent_4d.shape],
        "image_ids_shape": [int(v) for v in image_ids.shape],
        "style_image_ids_shape": [int(v) for v in style_image_ids.shape],
        "prompt_embeds_shape": [int(v) for v in prompt_embeds.shape],
        "null_embeds_shape": [int(v) for v in null_embeds.shape],
        "num_records": len(records),
        "num_pairwise_records": len(pairwise_records),
        "summary": summary,
        "pairwise_summary": pairwise_summary,
        "pairwise_csv": str(pairwise_csv_path),
        "pairwise_summary_path": str(pairwise_summary_path),
        "model_root": str(model_root_path),
        "diffsynth_root": diffsynth_root,
        "invariant": (
            "This profile computes v_normal/v_styled on the same z_t,t and calls "
            "scheduler.step at most once per main-path timestep."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"OK (run-model): wrote {len(records)} rows -> {csv_path}")
    print(f"OK: wrote summary -> {summary_path}")
    print(f"OK: wrote pairwise rows -> {pairwise_csv_path}")
    print(f"OK: wrote pairwise summary -> {pairwise_summary_path}")
    print(f"OK: wrote meta -> {meta_path}")
    return 0


def main() -> int:
    args = parse_args()

    if args.run_model:
        if args.runtime_smoke_only:
            try:
                return run_model_smoke(args)
            except Exception as exc:
                print(f"[run-model smoke failed] {exc}", file=sys.stderr)
                traceback.print_exc()
                return 1

        try:
            return run_model_profile(args)
        except Exception as exc:
            print(f"[run-model profile failed] {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    return run_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
