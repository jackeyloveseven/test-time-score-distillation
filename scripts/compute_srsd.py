"""SRSD (Score-Residual Style Distance): a single-shot, generator-native STYLE metric,
computed and VALIDATED against CSD on an already-scored subset of generated outputs.

Standalone script. Does NOT modify any src/*.py method code -- it only *reuses* the frozen
CA-contrast machinery the method itself is built on:

  - src/native_edit_teacher.py::prepare_edit_condition   (native FLUX.2 edit-image encode of R)
  - src/native_edit_teacher.py::evaluate_native_edit_triplet
        with_velocity    = v(z | edit_condition=R, prompt=full_style_prompt)
        without_velocity = v(z | edit_condition=None, prompt=full_style_prompt)   ["without_prompt_mode=full_style_prompt"]
        g = with_velocity - without_velocity        <- exactly the CA term the ca_only style_mode
        uses as its style-loss driver in scripts/run_tpami_with_style_teacher.py (see that file's
        "elif args.style_mode == 'ca_only':" branch, which computes the *same* triplet the *same*
        way at the *same* single-model config: main==teacher==/mnt/local/klein9b-base).

Idea (SRSD): if a generated image x already carries reference R's style, the frozen model has
little style left to add on top of x, so the norm of that CA residual g, evaluated ON x (not on
an intermediate denoising trajectory latent), should be SMALL. So:

    SRSD(x, R) = mean_sigma  RMS( v(z_sigma | edit=R, full_style_prompt)
                                  - v(z_sigma | edit=None, full_style_prompt) )

where z_sigma = (1 - sigma) * x0 + sigma * eps  (rectified-flow forward noise; eps fixed, seed=65)
is built directly from x's own clean VAE latent x0, at a small fixed set of noise levels sigma
(default {0.3, 0.5, 0.7}).

LOWER SRSD => model has less new style content left to inject => x is judged MORE style-faithful.
This script also runs the validation gate: correlate SRSD against pre-computed CSD (CLIP style
similarity) scores for a representative LOW->HIGH style-fidelity subset, and reports Pearson r /
Spearman rho, plus a scatter plot, plus a verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.native_edit_teacher import (  # noqa: E402
    evaluate_native_edit_triplet,
    freeze_condition_tensor,
    prepare_edit_condition,
)

DIFFSYNTH_ROOT_DEFAULT = "/mnt/local/yzh/prog/DiffSynth-Studio"
MODEL_ROOT_DEFAULT = "/mnt/local/klein9b-base"  # single model, main==teacher (per task spec)
STYLE_ROOT_DEFAULT = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty"
MANIFEST_CSV_DEFAULT = (
    "/tmp/claude-1000/-home-ubuntu-chengyanli/0da1ecad-87ce-4e83-abe2-cbcdbb020c1e/"
    "scratchpad/pg_manifest_metrics.csv"
)
SCATTER_PNG_DEFAULT = (
    "/tmp/claude-1000/-home-ubuntu-chengyanli/0da1ecad-87ce-4e83-abe2-cbcdbb020c1e/"
    "scratchpad/srsd_vs_csd.png"
)
RESULTS_CSV_DEFAULT = (
    "/tmp/claude-1000/-home-ubuntu-chengyanli/0da1ecad-87ce-4e83-abe2-cbcdbb020c1e/"
    "scratchpad/srsd_results.csv"
)

# Same PROMPTS/template as scripts/run_tpami_with_style_teacher.py / probe_native_edit_teacher.py
# so the encoded "full style prompt" text is byte-identical to what the method uses.
PROMPTS = {
    "kp0": "A panda eating bamboo",
    "kp1": "A castle on a cliff by the sea",
    "kp2": "A cat sitting on a windowsill",
    "kp3": "A field of sunflowers at sunset",
    "kp4": "A portrait of an old fisherman",
}
PROMPT_TEMPLATE = (
    "Create {content}. Use the visual style of the reference image, "
    "including its medium, palette, texture, and brushwork, but do not "
    "copy the reference objects or composition."
)

# Representative LOW->HIGH style-fidelity subset (per task): 5 styles x 4 tiers of CA-only configs
# (low, low-mid, mid, high) spanning CSD ~0.21 -> ~0.90 in the manifest CSV.
DEFAULT_METHODS = ["s06", "pg_n6_s05_rt", "s08", "se_i8_s08_rt"]
DEFAULT_STYLES = ["00", "09", "10", "23", "34"]


def install_modelscope_stub() -> None:
    """Same stub scripts/run_tpami_with_style_teacher.py::install_modelscope_stub() installs --
    diffsynth's ModelConfig unconditionally imports modelscope at module load time even though we
    only ever load local checkpoints; this avoids requiring the (unused) modelscope package."""
    if "modelscope" in sys.modules:
        return
    module = types.ModuleType("modelscope")
    module.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("modelscope is disabled; compute_srsd.py only loads local checkpoints")
    )
    sys.modules["modelscope"] = module


def sorted_files(pattern: str) -> list[str]:
    import glob

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def wait_for_gpu(min_free_mib: int, device_index: int, poll_s: int) -> int:
    """Gate on nvidia-smi memory.free before a model forward. NEVER kills anything; just polls."""
    while True:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(device_index),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        free_mib = int(out.splitlines()[0].strip())
        if free_mib >= min_free_mib:
            return free_mib
        print(
            f"[gpu-gate] free={free_mib}MiB < required={min_free_mib}MiB; sleeping {poll_s}s",
            flush=True,
        )
        time.sleep(poll_s)


def load_pipe(model_root: Path, diffsynth_root: str):
    if diffsynth_root not in sys.path:
        sys.path.insert(0, diffsynth_root)
    install_modelscope_stub()
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

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
    return pipe


def encode_prompt_for_pipe(pipe, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirrors scripts/run_tpami_with_style_teacher.py::encode_prompt_for_pipe (teacher branch
    only -- SRSD never needs a null/negative-prompt embed)."""
    from diffsynth.pipelines.flux2_image import Flux2Unit_PromptEmbedder, Flux2Unit_Qwen3PromptEmbedder

    if pipe.text_encoder_qwen3 is not None:
        pipe.load_models_to_device(["text_encoder_qwen3"])
        prompt_unit = Flux2Unit_Qwen3PromptEmbedder()
        text_encoder = pipe.text_encoder_qwen3
    else:
        pipe.load_models_to_device(["text_encoder"])
        prompt_unit = Flux2Unit_PromptEmbedder()
        text_encoder = pipe.text_encoder
    with torch.no_grad():
        prompt_embeds, text_ids = prompt_unit.encode_prompt(
            text_encoder, pipe.tokenizer, prompt_text, dtype=pipe.torch_dtype, device="cuda",
        )
    prompt_embeds = freeze_condition_tensor(prompt_embeds, device="cuda", dtype=pipe.torch_dtype)
    text_ids = freeze_condition_tensor(text_ids, device="cuda")
    return prompt_embeds, text_ids


def encode_clean_latent(pipe, image: Image.Image) -> torch.Tensor:
    """VAE-encode `image` to its clean latent x0 in the (B, N, C) token layout the DiT consumes.
    Same reuse path as scripts/run_tpami_with_style_teacher.py::encode_style_latent, just applied
    to the GENERATED image x instead of the style reference."""
    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        pixels = pipe.preprocess_image(image)
        latent_4d = pipe.vae.encode(pixels)
        latent = rearrange(latent_4d, "B C H W -> B (H W) C").to(device="cuda", dtype=torch.bfloat16)
    return latent


def compute_srsd_for_sample(
    pipe,
    dit,
    image_x: Image.Image,
    reference_r: Image.Image,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    sigmas: list[float],
    eps_seed: int,
) -> dict[str, object]:
    """Implements SRSD(x, R) = mean_sigma RMS(v_with(z_sigma) - v_without(z_sigma))."""

    x0 = encode_clean_latent(pipe, image_x)  # (1, N, C) bf16, N = (H/16)*(W/16)
    edit_condition = prepare_edit_condition(pipe, reference_r, auto_resize=False)

    from diffsynth.pipelines.flux2_image import Flux2Unit_ImageIDs

    lat_h, lat_w = image_x.height // 16, image_x.width // 16
    image_ids = Flux2Unit_ImageIDs().prepare_latent_ids(lat_h, lat_w).to("cuda")

    # Fixed epsilon: a local (never-global) CPU generator seeded `eps_seed`, matching the codebase's
    # pattern of local seeded generators for any deterministic-noise construction (see
    # src/native_edit_teacher.py::build_layout_free_edit_condition's local torch.Generator use).
    generator = torch.Generator(device="cpu").manual_seed(int(eps_seed))
    eps = torch.randn(tuple(x0.shape), generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=torch.float32
    )

    x0_f32 = x0.float()
    per_sigma_g_rms: dict[str, float] = {}
    per_sigma_without_rms: dict[str, float] = {}
    per_sigma_with_rms: dict[str, float] = {}
    for sigma in sigmas:
        z = ((1.0 - sigma) * x0_f32 + sigma * eps).to(dtype=torch.bfloat16)
        # model_fn_flux2 divides `timestep` by 1000 internally (see diffsynth flux2_image.py), so
        # passing timestep=sigma*1000 makes the DiT condition on exactly this sigma -- the same
        # raw-timestep convention scripts/run_tpami_with_style_teacher.py uses via
        # `timestep_style_cuda = timesteps[outer_idx+1]` (a raw 0..1000-scale scheduler timestep).
        timestep = torch.tensor([sigma * 1000.0], device="cuda", dtype=torch.bfloat16)
        triplet = evaluate_native_edit_triplet(
            dit=dit,
            latents=z,
            timestep=timestep,
            embedded_guidance=0.0,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=prompt_embeds,  # without_prompt_mode=full_style_prompt: SAME prompt
            negative_text_ids=text_ids,            # for with/without branches; only edit_condition differs
            image_ids=image_ids,
            edit_condition=edit_condition,
            require_pre_grad=False,
            share_without_with_pre=True,
        )
        g = (triplet.with_velocity - triplet.without_velocity).float()
        g_rms = torch.sqrt(torch.mean(g * g)).item()
        per_sigma_g_rms[f"sigma_{sigma}"] = g_rms
        # Discriminating diagnostic (does SRSD track a genuine style/content residual, or is it
        # just riding a general latent-scale/OOD blowup in the reference-FREE branch?): RMS of
        # v_without alone has nothing to do with R, so if it scales up in lockstep with g_rms for
        # heavily-stylized/composition-collapsed samples, SRSD is not isolating a style signal.
        without_rms = torch.sqrt(torch.mean(triplet.without_velocity.float() ** 2)).item()
        with_rms = torch.sqrt(torch.mean(triplet.with_velocity.float() ** 2)).item()
        per_sigma_without_rms[f"sigma_{sigma}"] = without_rms
        per_sigma_with_rms[f"sigma_{sigma}"] = with_rms

    srsd = float(np.mean(list(per_sigma_g_rms.values())))
    mean_without_rms = float(np.mean(list(per_sigma_without_rms.values())))
    mean_with_rms = float(np.mean(list(per_sigma_with_rms.values())))
    srsd_norm = srsd / mean_without_rms if mean_without_rms > 0 else float("nan")
    return {
        "srsd": srsd,
        "mean_without_rms": mean_without_rms,
        "mean_with_rms": mean_with_rms,
        "srsd_norm": srsd_norm,
        **per_sigma_g_rms,
        **{f"without_{k}": v for k, v in per_sigma_without_rms.items()},
    }


def load_manifest_rows(manifest_csv: Path) -> list[dict[str, str]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_samples(
    rows: list[dict[str, str]],
    methods: list[str],
    styles: list[str],
    pairs: list[tuple[str, str]] | None,
) -> list[dict[str, str]]:
    by_key = {(row["method"], row["style_id"]): row for row in rows}
    selected: list[dict[str, str]] = []
    if pairs:
        wanted = pairs
    else:
        wanted = [(m, s) for m in methods for s in styles]
    missing: list[tuple[str, str]] = []
    for method, style_id in wanted:
        row = by_key.get((method, style_id))
        if row is None:
            missing.append((method, style_id))
            continue
        selected.append(row)
    if missing:
        print(f"[select] WARNING: {len(missing)} requested (method,style) pairs not found in manifest: {missing}", flush=True)
    return selected


def pearson_spearman(xs: list[float], ys: list[float]) -> dict[str, object]:
    from scipy import stats

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    pearson_r, pearson_p = stats.pearsonr(xs_arr, ys_arr)
    spearman_rho, spearman_p = stats.spearmanr(xs_arr, ys_arr)
    return {
        "n": int(len(xs_arr)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def make_scatter(records: list[dict[str, object]], out_path: Path, corr: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["srsd"] for r in records]
    ys = [r["csd"] for r in records]
    labels = [f"{r['method']}/s{r['style_id']}" for r in records]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    ax.scatter(xs, ys, s=60, c="#2563eb", edgecolors="white", linewidths=0.6, zorder=3)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=6.5, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    # least-squares trend line for visual reference
    if len(xs) >= 2:
        coeffs = np.polyfit(xs, ys, 1)
        xs_line = np.linspace(min(xs), max(xs), 50)
        ax.plot(xs_line, np.polyval(coeffs, xs_line), color="#dc2626", linewidth=1.5, alpha=0.8, zorder=2)

    ax.set_xlabel("SRSD (Score-Residual Style Distance) — lower = more style-faithful")
    ax.set_ylabel("CSD style similarity — higher = more style-faithful")
    ax.set_title(
        f"SRSD vs CSD (n={corr['n']})\n"
        f"Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.3g})  "
        f"Spearman ρ={corr['spearman_rho']:.3f} (p={corr['spearman_p']:.3g})"
    )
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsynth-root", default=DIFFSYNTH_ROOT_DEFAULT)
    parser.add_argument("--model-root", default=MODEL_ROOT_DEFAULT)
    parser.add_argument("--style-root", default=STYLE_ROOT_DEFAULT)
    parser.add_argument("--manifest-csv", default=MANIFEST_CSV_DEFAULT)
    parser.add_argument("--prompt-id", default="kp0")
    parser.add_argument("--sigmas", default="0.3,0.5,0.7")
    parser.add_argument("--eps-seed", type=int, default=65)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--styles", default=",".join(DEFAULT_STYLES))
    parser.add_argument(
        "--pairs",
        default=None,
        help="Optional explicit 'method:style,method:style,...' list overriding --methods x --styles.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-min-free-mib", type=int, default=40000)
    parser.add_argument("--gpu-poll-s", type=int, default=20)
    parser.add_argument("--results-csv", default=RESULTS_CSV_DEFAULT)
    parser.add_argument("--scatter-png", default=SCATTER_PNG_DEFAULT)
    return parser.parse_args()


def main() -> int:
    import os

    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_index))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    pairs = None
    if args.pairs:
        pairs = []
        for item in args.pairs.split(","):
            method, style_id = item.split(":")
            pairs.append((method.strip(), style_id.strip()))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    manifest_rows = load_manifest_rows(Path(args.manifest_csv))
    samples = select_samples(manifest_rows, methods, styles, pairs)
    if not samples:
        raise RuntimeError("No samples selected -- check --manifest-csv/--methods/--styles/--pairs")
    print(f"[select] {len(samples)} samples selected from manifest {args.manifest_csv}", flush=True)

    wait_for_gpu(args.gpu_min_free_mib, args.gpu_index, args.gpu_poll_s)
    print("[load] loading single model (main==teacher) ...", flush=True)
    pipe = load_pipe(Path(args.model_root), args.diffsynth_root)
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit.to("cuda")
    dit.requires_grad_(False)

    prompt_text = PROMPT_TEMPLATE.format(content=PROMPTS.get(args.prompt_id, args.prompt_id))
    prompt_embeds, text_ids = encode_prompt_for_pipe(pipe, prompt_text)
    print(f"[prompt] '{prompt_text}'", flush=True)

    records: list[dict[str, object]] = []
    for i, row in enumerate(samples):
        method = row["method"]
        style_id = row["style_id"]
        output_path = Path(row["output_path"])
        style_path = Path(row["style_path"])
        csd = float(row["csd_style_sim"])
        if not output_path.exists():
            print(f"[skip] {method}/{style_id}: missing generated PNG {output_path}", flush=True)
            continue
        if not style_path.exists():
            print(f"[skip] {method}/{style_id}: missing reference PNG {style_path}", flush=True)
            continue

        wait_for_gpu(args.gpu_min_free_mib, args.gpu_index, args.gpu_poll_s)

        image_x = Image.open(output_path).convert("RGB")
        reference_r = Image.open(style_path).convert("RGB").resize(
            image_x.size, Image.Resampling.BICUBIC
        )

        result = compute_srsd_for_sample(
            pipe, dit, image_x, reference_r, prompt_embeds, text_ids, sigmas, args.eps_seed,
        )
        srsd = result["srsd"]
        finite = math.isfinite(srsd) and all(math.isfinite(result[f"sigma_{s}"]) for s in sigmas)
        record = {
            "method": method,
            "style_id": style_id,
            "output_path": str(output_path),
            "style_path": str(style_path),
            "csd": csd,
            "srsd": srsd,
            "finite": finite,
            "dino_layout_leak": float(row["dino_layout_leak"]) if row.get("dino_layout_leak") else None,
            "clip_t": float(row["clip_t"]) if row.get("clip_t") else None,
            "vgg_gram_dist": float(row["vgg_gram_dist"]) if row.get("vgg_gram_dist") else None,
            "dino_style_sim": float(row["dino_style_sim"]) if row.get("dino_style_sim") else None,
            **{f"srsd_{k}": v for k, v in result.items() if k != "srsd"},
        }
        records.append(record)
        print(
            f"[{i + 1}/{len(samples)}] {method}/s{style_id}  CSD={csd:.4f}  SRSD={srsd:.6f}  "
            f"SRSD_norm={result['srsd_norm']:.4f}  ||v_without||_rms={result['mean_without_rms']:.6f}  "
            f"||v_with||_rms={result['mean_with_rms']:.6f}  (finite={finite})",
            flush=True,
        )

    if not records:
        raise RuntimeError("No valid samples produced SRSD values")

    non_finite = [r for r in records if not r["finite"]]
    if non_finite:
        print(f"[WARN] {len(non_finite)} samples produced non-finite SRSD; excluding from correlation.", flush=True)
    clean_records = [r for r in records if r["finite"]]

    results_csv_path = Path(args.results_csv)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with results_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[write] {results_csv_path}", flush=True)

    xs = [r["srsd"] for r in clean_records]
    ys = [r["csd"] for r in clean_records]
    corr = pearson_spearman(xs, ys)
    print(json.dumps({"srsd_vs_csd_correlation": corr}, indent=2), flush=True)

    # Scale-artifact check: does raw g_rms just ride a general blowup of the reference-FREE branch
    # ||v_without|| for OOD/heavily-stylized latents? If so, SRSD isn't isolating a style/content
    # residual. Report both the raw diagnostic and the scale-corrected SRSD_norm = g_rms/||v_without||.
    without_rms_vals = [r["srsd_mean_without_rms"] for r in clean_records]
    srsd_norm_vals = [r["srsd_srsd_norm"] for r in clean_records]
    without_rms_vs_csd = pearson_spearman(without_rms_vals, ys)
    without_rms_vs_srsd = pearson_spearman(without_rms_vals, xs)
    srsd_norm_vs_csd = pearson_spearman(srsd_norm_vals, ys)
    print(
        json.dumps(
            {
                "scale_diagnostic": {
                    "without_velocity_rms_vs_csd": without_rms_vs_csd,
                    "without_velocity_rms_vs_srsd": without_rms_vs_srsd,
                    "srsd_norm_vs_csd_correlation": srsd_norm_vs_csd,
                }
            },
            indent=2,
        ),
        flush=True,
    )

    # Supplementary: does SRSD track composition/layout leakage instead of / in addition to style?
    supplementary: dict[str, object] = {}
    for aux_key in ("dino_layout_leak", "clip_t", "vgg_gram_dist", "dino_style_sim"):
        aux_vals = [r[aux_key] for r in clean_records if r.get(aux_key) is not None]
        if len(aux_vals) == len(clean_records) and len(aux_vals) >= 3:
            aux_corr = pearson_spearman(xs, aux_vals)
            supplementary[f"srsd_vs_{aux_key}"] = aux_corr
    if supplementary:
        print(json.dumps({"supplementary_correlations": supplementary}, indent=2), flush=True)

    # Per-sigma sensitivity: correlation of CSD with each individual sigma's g_rms.
    sigma_sensitivity: dict[str, object] = {}
    for sigma in sigmas:
        key = f"srsd_sigma_{sigma}"
        vals = [r[key] for r in clean_records if key in r]
        if len(vals) == len(clean_records) and len(vals) >= 3:
            sigma_sensitivity[key] = pearson_spearman(vals, ys)
    if sigma_sensitivity:
        print(json.dumps({"per_sigma_vs_csd_correlation": sigma_sensitivity}, indent=2), flush=True)

    scatter_path = Path(args.scatter_png)
    make_scatter(clean_records, scatter_path, corr)
    print(f"[write] {scatter_path}", flush=True)

    verdict_threshold = -0.5
    validated = corr["pearson_r"] <= verdict_threshold and corr["spearman_rho"] <= verdict_threshold
    verdict = "VALIDATES" if validated else "DOES NOT CLEANLY VALIDATE"
    print(
        f"\n[VERDICT] SRSD {verdict} as a style metric "
        f"(Pearson r={corr['pearson_r']:.3f}, Spearman rho={corr['spearman_rho']:.3f}, n={corr['n']}; "
        f"threshold for a clean validation: both <= {verdict_threshold}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
