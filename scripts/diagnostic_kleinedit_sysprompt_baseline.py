"""Baseline: klein's NATIVE image-editing capability, given its STRONGEST FAIR style-transfer
prompt, NO CA optimization. Sibling of ``scripts/diagnostic_kleinedit_baseline.py`` -- same
pipeline, same knobs, same non-method-code-touching philosophy -- but with a proper
style-transfer instruction instead of the earlier WEAK prompts ("A panda eating bamboo" / a
generic "...rendered in the medium/style of the reference image..." one-liner), so the
klein-native-edit comparison baseline is not accidentally squeezed by weak prompting.

--- System-prompt investigation (see report) ---
``Flux2ImagePipeline.__call__`` (diffsynth/pipelines/flux2_image.py:74-97) exposes only
``prompt`` and ``negative_prompt`` as text-conditioning inputs -- there is no
``system_prompt``/``template``/instruction-mode kwarg anywhere in its signature.

Whether a *hardcoded* system message is even applied depends on which text encoder the
checkpoint ships. There are two mutually-exclusive prompt-embedder units wired into the
pipeline (flux2_image.py:35-43): ``Flux2Unit_PromptEmbedder`` (Mistral-3-small text encoder,
"flux2_text_encoder") DOES have a hardcoded ``self.system_message`` (flux2_image.py:161) that
is always injected as a chat "system" role turn (format_text_input, flux2_image.py:163-178) --
but it is NOT user-configurable through ``__call__``, only by mutating the unit instance
in-process. ``Flux2Unit_Qwen3PromptEmbedder`` (Qwen3 text encoder, "z_image_text_encoder") is
the OTHER one, and per this checkpoint's own ``model_index.json``
(``"_class_name": "Flux2KleinPipeline"``, ``"text_encoder": ["transformers",
"Qwen3ForCausalLM"]``) this is the one actually used for ``/mnt/local/klein9b-base``. Its
``get_qwen3_prompt_embeds`` (flux2_image.py:311-364) hardcodes
``messages = [{"role": "user", "content": single_prompt}]`` (flux2_image.py:329) -- there is
NO system_message attribute or parameter anywhere on this unit. This is true even though the
checkpoint's own Qwen chat template
(``/mnt/local/klein9b-base/tokenizer/chat_template.jinja``) has first-class support for a
leading "system" role turn (``{%- if messages[0].role == 'system' %} ... {%- endif %}``) --
DiffSynth's Qwen3 embedder simply never uses it.

Conclusion: for the checkpoint this repo actually loads, there is NO system-prompt channel
reachable through the public pipeline API. We fold the full expert style-transfer instruction
into the single ``prompt`` string instead of reaching for a runtime patch (e.g. overwriting
``pipe.units[i]``'s message-building method in-process to splice in a system turn) for two
reasons: (1) it is not exposed by the public API klein ships, so using it would no longer be
testing "klein's native edit capability" through its documented interface, and (2)
hand-rolling a replacement text-encoding call risks a silently-broken embedding (wrong
padding/truncation/message format vs. what the checkpoint was trained on) that would make
klein look unfairly WORSE, not better -- exactly the failure mode this baseline is trying to
avoid. (An in-memory unit patch only affects this process's own `pipe` object -- it does not
touch the on-disk DiffSynth-Studio install, so it would NOT be able to affect concurrently
running jobs like the E6 param sweep; that is not the reason for avoiding it.) We additionally
use the ``negative_prompt`` channel (which IS exposed and IS real CFG here, since cfg_scale=4
!= 1 -- see ``BasePipeline.cfg_guided_model_fn``, base_pipeline.py:315-334) to push away from
generic photorealistic renders that ignore the reference's medium -- kept reference-agnostic
(see ``_NEGATIVE_BASE`` below) since two of the five style refs are themselves flat-color
mediums (halftone pop-art, stained glass), so a negative prompt that penalizes flatness would
unfairly weaken the baseline on exactly those references. If fold-in results come back weak,
the in-memory system-role patch remains available as a next lever.

Model loading reuses ``load_flux2_pipe`` from scripts/run_tpami_with_style_teacher.py exactly
as ``diagnostic_kleinedit_baseline.py`` does (bf16, cpu_offload, DIFFSYNTH_ATTENTION_IMPLEMENTATION
forced to "torch", DiffSynth root pinned to /mnt/local/yzh/prog/DiffSynth-Studio). Does NOT
touch method code (src/*.py); only this standalone script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Match diagnostic_kleinedit_baseline.py / run_tpami_with_style_teacher.py:1833 EXACTLY, and set
# it before diffsynth is ever imported (diffsynth.core.attention.attention reads this env var
# once at module-import time to pick its attention backend). Without this, diffsynth's priority
# list falls through to xformers (flash_attn/sage_attention are not installed in this env), and
# the installed xformers build has no CUDA kernels for this H200 (sm90) -> NotImplementedError
# at the first denoise step.
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match run_tpami_with_style_teacher.py's default --diffsynth-root exactly.
DIFFSYNTH_ROOT = "/mnt/local/yzh/prog/DiffSynth-Studio"
if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

STYLE_DIR = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty")
MODEL_ROOT = Path("/mnt/local/klein9b-base")
VRAM_LIMIT_GB = 38.0
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 25
CFG_SCALE = 4.0            # same as diagnostic_kleinedit_baseline.py -- klein's own documented
                             # default (DiffSynth-Studio example for this exact checkpoint).
EMBEDDED_GUIDANCE = 4.0     # Flux2ImagePipeline.__call__'s own default.
SEED = 65
GATE_FREE_MIB = 40000
GATE_SLEEP_S = 20

# --- Strongest fair style-transfer prompt ------------------------------------------------
# No system-prompt channel is reachable through the public pipeline API for this checkpoint
# (see module docstring), so the full expert instruction is folded into the single content
# `prompt` string. negative_prompt IS a real, exposed CFG channel here (cfg_scale=4 != 1), so
# it is also used to push away from generic/flat renders that ignore the reference style.
#
# Multi-prompt robustness extension (2026-07-21): the subject clause ("a panda eating
# bamboo") is now parametrized per --prompt-id so this baseline can be benchmarked across
# run_tpami_with_style_teacher.py's kp1-kp4 prompts too, not just kp0. The SAME strong
# style-transfer instruction wrapper and negative_prompt are kept -- only the subject noun
# phrase changes. Default subject (kp0) is reproduced byte-for-byte, so --prompt-id kp0
# (the default) is unchanged from the original hardcoded behaviour.
DEFAULT_SUBJECT = "a panda eating bamboo"


def _style_instruction(subject: str) -> str:
    return (
        "Expert style transfer: reproduce the reference image's exact artistic medium, material, "
        "and technique -- for example mosaic tesserae, bronze relief, oil impasto, halftone "
        "pop-art, or stained-glass leading, whichever it actually is -- matching its color "
        f"palette, texture, and line quality precisely. Fully render {subject} in that "
        "exact medium and style"
    )


_NEGATIVE_BASE = (
    "generic photorealistic photo, ignoring the reference's artistic medium, blurry, "
    "watermark, text, low quality"
)
# NOTE: deliberately does NOT include attribute-prescribing terms like "flat colors" / "low
# detail" / "plain rendering" -- two of the five style refs actually used (style00: halftone
# pop-art flat color fills; style34: stained-glass flat panes + black leading) are themselves
# flat-color mediums. A negative prompt that penalizes flatness would unfairly steer klein AWAY
# from faithfully reproducing exactly those references, which would confound the "did the
# medium transfer" read and unfairly weaken (not strengthen) this baseline. Kept
# reference-agnostic: only rules out generic-quality failure modes and explicit
# non-reproduction of the reference's medium, valid regardless of what that medium is.


def build_prompts(subject: str) -> dict[str, str]:
    style_instruction = _style_instruction(subject)
    return {
        "sem": style_instruction + ", preserving the reference's composition and layout.",
        "gst": (
            style_instruction
            + ", in a COMPLETELY NEW composition -- do NOT reuse the reference's layout or "
            "arrangement."
        ),
    }


NEGATIVE_PROMPTS = {
    "sem": _NEGATIVE_BASE,
    "gst": _NEGATIVE_BASE + ", identical composition to the reference, copying the reference's layout",
}
# Kept for reference/back-compat (e.g. anything introspecting module-level PROMPTS at kp0).
PROMPTS = build_prompts(DEFAULT_SUBJECT)
STYLE_IDS = [f"{i:02d}" for i in range(40)]
PROMPT_ID = "kp0"

# subject clause per prompt-id: the SAME literal subject strings as
# run_tpami_with_style_teacher.py's PROMPTS dict (kp1-kp4), lowercased-first-letter so they
# read naturally after "Fully render ...". kp0 reproduces the original hardcoded subject
# exactly (DEFAULT_SUBJECT == "a panda eating bamboo" == PROMPTS_KP["kp0"] lowercased).
SUBJECT_BY_PROMPT_ID = {
    "kp0": "a panda eating bamboo",
    "kp1": "a castle on a cliff by the sea",
    "kp2": "a cat sitting on a windowsill",
    "kp3": "a field of sunflowers at sunset",
    "kp4": "a portrait of an old fisherman",
    "kp5": "a lighthouse on a rocky coast",
}


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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--prompt-id",
        default="kp0",
        choices=sorted(SUBJECT_BY_PROMPT_ID),
        help="Which subject to render (default kp0 = original hardcoded 'a panda eating bamboo', "
        "unchanged from this script's original behaviour). kp1-kp4 reuse the SAME style-transfer "
        "instruction wrapper and negative_prompt, only the subject clause changes.",
    )
    args = parser.parse_args()

    subject = SUBJECT_BY_PROMPT_ID[args.prompt_id]
    prompts = build_prompts(subject)

    load_flux2_pipe, _sorted_files, install_modelscope_stub = _import_run_tpami_helpers()
    install_modelscope_stub()

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the exact prompts used, for the report, next to the outputs. kp0 keeps the
    # original filename (byte-for-byte back-compat); other prompt-ids get their own file so
    # they don't clobber the kp0 record.
    suffix = "" if args.prompt_id == "kp0" and args.seed == SEED else f"_{args.prompt_id}_seed{args.seed}"
    prompt_log_path = out_dir / f"diagnostic_kleinsys_prompts{suffix}.json"
    prompt_log_path.write_text(json.dumps({"prompts": prompts, "negative_prompts": NEGATIVE_PROMPTS}, indent=2))
    print(f"[log] wrote {prompt_log_path}", flush=True)

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
    print(
        f"[load] pipe ready text_encoder(mistral)={pipe.text_encoder is not None} "
        f"text_encoder_qwen3={pipe.text_encoder_qwen3 is not None}",
        flush=True,
    )
    dit = getattr(pipe, "dit", None)
    if dit is None or not hasattr(dit, "register_forward_hook"):
        raise RuntimeError("native edit pipeline does not expose a hookable DiT for forward-call accounting")

    for sid, pkey in jobs:
        if args.seed == SEED:
            out_path = out_dir / f"diagnostic_kleinsys_{pkey}_style{sid}_{args.prompt_id}.png"
        else:
            out_path = out_dir / (
                f"diagnostic_kleinsys_{pkey}_style{sid}_{args.prompt_id}_seed{args.seed}.png"
            )
        if out_path.exists():
            print(f"[skip] {out_path} already exists", flush=True)
            continue
        style_path = STYLE_DIR / f"{sid}.png"
        if not style_path.exists():
            raise FileNotFoundError(style_path)
        edit_prompt = prompts[pkey]
        negative_prompt = NEGATIVE_PROMPTS[pkey]

        wait_for_gpu(f"gen sid={sid} pkey={pkey}")

        style_image = Image.open(style_path).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)

        forward_calls = 0

        def _count_forward(_module, _inputs, _output):
            nonlocal forward_calls
            forward_calls += 1

        hook = dit.register_forward_hook(_count_forward)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = pipe(
            prompt=edit_prompt,
            negative_prompt=negative_prompt,
            cfg_scale=CFG_SCALE,
            embedded_guidance=EMBEDDED_GUIDANCE,
            edit_image=[style_image],
            edit_image_auto_resize=False,
            height=HEIGHT,
            width=WIDTH,
            seed=args.seed,
            rand_device="cuda",
            num_inference_steps=NUM_INFERENCE_STEPS,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        hook.remove()
        image.save(out_path)
        mean, std = check_non_black(out_path)
        summary_path = out_path.with_name(out_path.stem + "_summary.json")
        summary_path.write_text(
            json.dumps(
                {
                    "method": "native_mmdit_edit",
                    "variant": pkey,
                    "style_id": sid,
                    "prompt_id": args.prompt_id,
                    "seed": int(args.seed),
                    "model_root": str(MODEL_ROOT),
                    "height": HEIGHT,
                    "width": WIDTH,
                    "num_steps": NUM_INFERENCE_STEPS,
                    "cfg_scale": CFG_SCALE,
                    "embedded_guidance": EMBEDDED_GUIDANCE,
                    "wall_time_inference_s": float(dt),
                    "dit_forward_calls_total": int(forward_calls),
                    "latent_backward_calls": 0,
                    "cuda_peak_memory_gb": (
                        float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                        if torch.cuda.is_available()
                        else None
                    ),
                    "image_path": str(out_path),
                    "style_path": str(style_path),
                    "prompt": edit_prompt,
                    "negative_prompt": negative_prompt,
                    "pixel_mean": mean,
                    "pixel_std": std,
                },
                indent=2,
            )
        )
        print(
            f"[done] sid={sid} pkey={pkey} seed={args.seed} -> {out_path} "
            f"dt={dt:.1f}s forwards={forward_calls} pixel_mean={mean:.2f} pixel_std={std:.2f}",
            flush=True,
        )
        if std < 1.0:
            print(f"[WARN] {out_path} looks flat/black (std={std:.3f})", flush=True)

    print("[all done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
