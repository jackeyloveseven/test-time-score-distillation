"""Baseline: klein's NATIVE image-editing capability, NO CA optimization.

This is a comparison baseline for the ca_only (Test-Time Score Distillation) method in
``scripts/run_tpami_with_style_teacher.py``. It does NOT touch method code. It feeds the
frozen klein (FLUX.2 [klein] 9B, /mnt/local/klein9b-base) DiffSynth ``Flux2ImagePipeline``
directly with (style image as edit_image + an edit-prompt text) and runs the pipeline's own
standard CFG-guided denoise loop -- no latent optimization, no CA pseudo-gradient, no
attention-injection, nothing method-specific.

Why a *standalone* script instead of ``--lambda-ca 0`` on the method script: investigation
(see report) showed the method's own frozen reverse trajectory (the trajectory that is
actually integrated via ``scheduler.step`` in run_tpami_with_style_teacher.py:2200-2212) is
edit-image-UNCONDITIONED -- it only ever sees ``main_prompt_embeds`` (prompt text), never
``edit_latents``/``edit_image_ids``. The edit image only ever enters through the CA
pseudo-gradient latent push in ``inner_optimize`` (run_tpami_with_style_teacher.py:2291,
using src/native_edit_teacher.py's with/without velocity triplet). With
``--lambda-ca 0``, ``pseudo_grad = lambda_ca * (cfg_scale - 1) * delta_ca_score`` is exactly
the zero tensor (src/style_objectives.py:222), so the rms-normalized pseudo-direction is
also exactly zero and ``build_pseudo_target_loss`` (src/flux2_score.py:233-234) sets
``x0_target == x0_pred.detach()`` -> zero loss -> zero gradient -> AdamW no-op. Combined
with the canonical config's ``--disable-slt --adain-alpha 0`` (which kills the only other
style-injection channel, ``warm_start_latents``), ``--lambda-ca 0`` on the method script
would just reproduce prompt-only text-to-image, NOT klein's native edit capability. Hence
this standalone script (Option B), which calls the real edit-conditioned denoise loop.

Model loading reuses ``load_flux2_pipe`` from scripts/run_tpami_with_style_teacher.py
(imported via importlib, without executing that module's ``main()``), so dtype (bf16) and
VRAM offload behavior (layer-wise cpu_offload_bf16 via src/klein9b_loading.py) exactly match
the method's model loading.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

# Match run_tpami_with_style_teacher.py:1833 EXACTLY, and set it before diffsynth is ever
# imported (diffsynth.core.attention.attention reads this env var once at module-import time
# to pick its attention backend). Without this, diffsynth's priority list falls through to
# xformers (flash_attn/sage_attention are not installed in this env), and the installed
# xformers build has no CUDA kernels for this H200 (sm90) -> NotImplementedError at the first
# denoise step. The method forces "torch" (SDPA) for exactly this reason.
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match run_tpami_with_style_teacher.py's default --diffsynth-root exactly (that is the
# diffsynth source tree the method itself imports; a *different* editable-installed diffsynth
# happens to be importable from the base conda env, but it is a different git revision and is
# NOT what the method used -- see report). Must be on sys.path before any `from diffsynth...`
# import (those are all deferred inside function bodies in run_tpami_with_style_teacher.py, so
# this is safe to do before importing its helpers).
DIFFSYNTH_ROOT = "/mnt/local/yzh/prog/DiffSynth-Studio"
if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

STYLE_DIR = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty")
MODEL_ROOT = Path("/mnt/local/klein9b-base")
VRAM_LIMIT_GB = 38.0
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 25
CFG_SCALE = 4.0            # klein's own documented default (DiffSynth-Studio example for this
                             # exact checkpoint uses cfg_scale=4); the method's --cfg-scale 7 is
                             # NOT real CFG for the frozen trajectory (only scales the CA
                             # pseudo-gradient), so it is not the relevant "matched" knob here.
EMBEDDED_GUIDANCE = 4.0     # Flux2ImagePipeline.__call__'s own default.
SEED = 65
GATE_FREE_MIB = 40000
GATE_SLEEP_S = 20

PROMPTS = {
    "sem": "A panda eating bamboo",
    "gst": (
        "A panda eating bamboo, rendered in the artistic medium and style of the reference "
        "image, in a completely new composition"
    ),
}
STYLE_IDS = ["23", "09", "10", "00", "34"]
PROMPT_ID = "kp0"


def _import_run_tpami_helpers():
    """Import load_flux2_pipe / sorted_files from run_tpami_with_style_teacher.py WITHOUT
    executing its main() (that only runs under __main__)."""
    spec = importlib.util.spec_from_file_location(
        "run_tpami_with_style_teacher_ro",
        REPO_ROOT / "scripts" / "run_tpami_with_style_teacher.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # only module-level defs/dicts run; main() is not invoked
    return module.load_flux2_pipe, module.sorted_files, module.install_modelscope_stub


def gpu_free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return int(out.strip())


def wait_for_gpu(tag: str) -> None:
    while True:
        free = gpu_free_mib()
        if free >= GATE_FREE_MIB:
            print(f"[gate] {tag}: free={free} MiB >= {GATE_FREE_MIB} -> proceed", flush=True)
            return
        print(f"[gate] {tag}: free={free} MiB < {GATE_FREE_MIB} -> sleep {GATE_SLEEP_S}s", flush=True)
        time.sleep(GATE_SLEEP_S)


def check_non_black(png_path: Path) -> tuple[float, float]:
    arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="comma list style_id:prompt_key, e.g. 23:sem for a single smoke test")
    args = parser.parse_args()

    load_flux2_pipe, _sorted_files, install_modelscope_stub = _import_run_tpami_helpers()
    install_modelscope_stub()  # run_tpami_with_style_teacher.py:1836 equivalent; diffsynth's
    # config loader imports modelscope at import time purely for a remote-download fallback we
    # never use (local checkpoints only), so the method stubs it out rather than requiring the
    # dependency. Must run before the first `from diffsynth...` import (inside load_flux2_pipe).

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.only:
        for token in args.only.split(","):
            sid, pkey = token.split(":")
            jobs.append((sid.strip(), pkey.strip()))
    else:
        for sid in STYLE_IDS:
            for pkey in ("sem", "gst"):
                jobs.append((sid, pkey))

    wait_for_gpu("model_load")
    print(f"[load] Flux2ImagePipeline from {MODEL_ROOT} vram_limit_gb={VRAM_LIMIT_GB}", flush=True)
    pipe = load_flux2_pipe(MODEL_ROOT, vram_limit_gb=VRAM_LIMIT_GB)
    print("[load] pipe ready", flush=True)

    for sid, pkey in jobs:
        out_path = out_dir / f"diagnostic_kleinedit_{pkey}_style{sid}_{PROMPT_ID}.png"
        if out_path.exists():
            print(f"[skip] {out_path} already exists", flush=True)
            continue
        style_path = STYLE_DIR / f"{sid}.png"
        if not style_path.exists():
            raise FileNotFoundError(style_path)
        edit_prompt = PROMPTS[pkey]

        wait_for_gpu(f"gen sid={sid} pkey={pkey}")

        style_image = Image.open(style_path).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)

        t0 = time.time()
        image = pipe(
            prompt=edit_prompt,
            negative_prompt="",
            cfg_scale=CFG_SCALE,
            embedded_guidance=EMBEDDED_GUIDANCE,
            edit_image=[style_image],
            edit_image_auto_resize=False,
            height=HEIGHT,
            width=WIDTH,
            seed=SEED,
            rand_device="cuda",
            num_inference_steps=NUM_INFERENCE_STEPS,
        )
        image.save(out_path)
        dt = time.time() - t0
        mean, std = check_non_black(out_path)
        print(
            f"[done] sid={sid} pkey={pkey} -> {out_path} dt={dt:.1f}s pixel_mean={mean:.2f} pixel_std={std:.2f}",
            flush=True,
        )
        if std < 1.0:
            print(f"[WARN] {out_path} looks flat/black (std={std:.3f})", flush=True)

    print("[all done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
