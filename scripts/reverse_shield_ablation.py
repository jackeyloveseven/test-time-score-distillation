#!/usr/bin/env python3
"""Inventory, diagnostics, and aggregation for the reverse-process shield ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_open_judge import MODEL as OPEN_MODEL
from vlm_open_judge import BASE_URL as OPEN_BASE_URL
from vlm_open_judge import data_uri, extract_json, normalize_style_match


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "reverse_shield"
STYLE_DIR = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty")
STYLE_IDS = ("00", "09", "10", "23", "34")
SETTINGS = ("standard", "stress")
RHOS = (1.0, 0.5, 0.0)
SEED = 65
PROMPT_ID = "kp0"
PROMPT = "A panda eating bamboo"

VLM_PROMPT = (
    "Image 1 is a STYLE REFERENCE. Image 2 is a generated image of a panda. Judge the output "
    "on four independent fields. style_match is 1-5 for artistic medium, texture, brushwork, "
    "palette, and rendering technique. content_clarity is 1-5 for a clear recognizable panda. "
    "artifacts is exactly none, minor, or major. overall_quality is 1-5 for coherence and visual "
    "quality. Return ONLY JSON with keys style_match, content_clarity, artifacts, overall_quality, note."
)


def rho_tag(rho: float) -> str:
    return f"{int(round(100 * rho)):03d}"


def output_path(setting: str, rho: float, style_id: str) -> Path:
    prefix = f"shield_{setting}_rho{rho_tag(rho)}"
    return RESULTS / f"diagnostic_{prefix}_ca_only_style{style_id}_{PROMPT_ID}_seed{SEED}.png"


def records() -> list[dict[str, object]]:
    rows = []
    for setting in SETTINGS:
        for rho in RHOS:
            for style_id in STYLE_IDS:
                path = output_path(setting, rho, style_id)
                rows.append(
                    {
                        "method": f"{setting}_rho{rho_tag(rho)}",
                        "setting": setting,
                        "reverse_step_strength": rho,
                        "style_id": style_id,
                        "prompt_id": PROMPT_ID,
                        "seed": SEED,
                        "output_path": str(path),
                        "exists": path.exists(),
                    }
                )
    return rows


def inventory(rows: list[dict[str, object]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "expected": len(rows),
        "present": sum(bool(row["exists"]) for row in rows),
        "missing": sum(not bool(row["exists"]) for row in rows),
        "styles": list(STYLE_IDS),
        "settings": list(SETTINGS),
        "reverse_step_strengths": list(RHOS),
    }
    (OUTDIR / "inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def manifest(rows: list[dict[str, object]], require_complete: bool) -> None:
    missing = [row for row in rows if not row["exists"]]
    if require_complete and missing:
        raise SystemExit(f"reverse-shield ablation incomplete: {len(missing)} cells missing")
    fields = ("method", "output_path", "style_path", "prompt", "style_id", "prompt_id", "seed")
    with (OUTDIR / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row["exists"]:
                continue
            writer.writerow(
                {
                    "method": row["method"],
                    "output_path": row["output_path"],
                    "style_path": str(STYLE_DIR / f"{row['style_id']}.png"),
                    "prompt": PROMPT,
                    "style_id": row["style_id"],
                    "prompt_id": PROMPT_ID,
                    "seed": SEED,
                }
            )


def image_diagnostics(path: Path) -> dict[str, float]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    dx = np.diff(luma, axis=1)
    dy = np.diff(luma, axis=0)
    lap = -4.0 * luma.copy()
    lap[1:, :] += luma[:-1, :]
    lap[:-1, :] += luma[1:, :]
    lap[:, 1:] += luma[:, :-1]
    lap[:, :-1] += luma[:, 1:]
    spectrum = np.fft.fftshift(np.fft.fft2(luma - luma.mean()))
    power = np.abs(spectrum) ** 2
    yy, xx = np.indices(luma.shape)
    cy, cx = (np.asarray(luma.shape) - 1.0) / 2.0
    radius = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2)
    high = power[radius >= 0.45].sum()
    total = power.sum()
    return {
        "luma_clip_fraction": float(np.mean((luma <= 0.01) | (luma >= 0.99))),
        "channel_clip_fraction": float(np.mean((rgb <= 0.01) | (rgb >= 0.99))),
        "total_variation": float(np.mean(np.abs(dx)) + np.mean(np.abs(dy))),
        "laplacian_rms": float(np.sqrt(np.mean(lap ** 2))),
        "high_frequency_power_ratio": float(high / total) if total > 0 else 0.0,
        "luma_std": float(luma.std()),
    }


def diagnostics(rows: list[dict[str, object]]) -> None:
    fields = [
        "method",
        "setting",
        "reverse_step_strength",
        "style_id",
        "seed",
        "output_path",
        "luma_clip_fraction",
        "channel_clip_fraction",
        "total_variation",
        "laplacian_rms",
        "high_frequency_power_ratio",
        "luma_std",
    ]
    with (OUTDIR / "image_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row["exists"]:
                continue
            writer.writerow({**{key: row[key] for key in fields[:6]}, **image_diagnostics(Path(row["output_path"]))})


def vlm_judge(reference: Path, output: Path) -> dict[str, object]:
    import requests

    body = {
        "model": OPEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri(reference)}},
                    {"type": "image_url", "image_url": {"url": data_uri(output)}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }
    response = requests.post(OPEN_BASE_URL + "/chat/completions", json=body, timeout=240)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    payload = extract_json(content)
    artifacts = str(payload.get("artifacts", "")).lower()
    if artifacts not in {"none", "minor", "major"}:
        artifacts = "unknown"
    return {
        "style_match": normalize_style_match(payload.get("style_match")),
        "content_clarity": normalize_style_match(payload.get("content_clarity")),
        "artifacts": artifacts,
        "overall_quality": normalize_style_match(payload.get("overall_quality")),
        "note": str(payload.get("note", ""))[:240],
        "raw": content,
    }


def run_vlm(rows: list[dict[str, object]]) -> None:
    output_file = OUTDIR / "vlm_diagnostics.jsonl"
    done = {}
    if output_file.exists():
        for line in output_file.open(encoding="utf-8"):
            try:
                item = json.loads(line)
                done[item["output_path"]] = item
            except (json.JSONDecodeError, KeyError):
                continue
    with output_file.open("a", encoding="utf-8") as handle:
        for row in rows:
            if not row["exists"] or row["output_path"] in done:
                continue
            verdict = vlm_judge(
                STYLE_DIR / f"{row['style_id']}.png",
                Path(row["output_path"]),
            )
            item = {**row, "model": OPEN_MODEL, "verdict": verdict}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[shield-vlm] {row['method']} style{row['style_id']} "
                f"style={verdict['style_match']} artifacts={verdict['artifacts']} "
                f"quality={verdict['overall_quality']}"
            )


def mean(values: list[float]) -> float:
    return float(stats.mean(values)) if values else math.nan


def aggregate(metrics_path: Path) -> None:
    metrics = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    image_diag = list(csv.DictReader((OUTDIR / "image_diagnostics.csv").open(encoding="utf-8")))
    vlm_rows = [json.loads(line) for line in (OUTDIR / "vlm_diagnostics.jsonl").open(encoding="utf-8")]
    by_method_metrics = {row["method"]: [] for row in metrics}
    for row in metrics:
        by_method_metrics.setdefault(row["method"], []).append(row)
    by_method_diag = {}
    for row in image_diag:
        by_method_diag.setdefault(row["method"], []).append(row)
    by_method_vlm = {}
    for row in vlm_rows:
        by_method_vlm.setdefault(row["method"], []).append(row)

    fields = [
        "setting",
        "reverse_step_strength",
        "n",
        "csd_mean",
        "clip_t_mean",
        "layout_leak_mean",
        "hf_power_ratio_mean",
        "laplacian_rms_mean",
        "clip_fraction_mean",
        "vlm_style_mean",
        "vlm_content_mean",
        "vlm_quality_mean",
        "vlm_major_artifact_rate",
    ]
    with (OUTDIR / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in SETTINGS:
            for rho in RHOS:
                method = f"{setting}_rho{rho_tag(rho)}"
                mr = by_method_metrics.get(method, [])
                dr = by_method_diag.get(method, [])
                vr = by_method_vlm.get(method, [])
                writer.writerow(
                    {
                        "setting": setting,
                        "reverse_step_strength": rho,
                        "n": len(mr),
                        "csd_mean": mean([float(row["csd_style_sim"]) for row in mr]),
                        "clip_t_mean": mean([float(row["clip_t"]) for row in mr]),
                        "layout_leak_mean": mean([float(row["dino_layout_leak"]) for row in mr]),
                        "hf_power_ratio_mean": mean([float(row["high_frequency_power_ratio"]) for row in dr]),
                        "laplacian_rms_mean": mean([float(row["laplacian_rms"]) for row in dr]),
                        "clip_fraction_mean": mean([float(row["channel_clip_fraction"]) for row in dr]),
                        "vlm_style_mean": mean([
                            float(row["verdict"]["style_match"])
                            for row in vr if row["verdict"].get("style_match") is not None
                        ]),
                        "vlm_content_mean": mean([
                            float(row["verdict"]["content_clarity"])
                            for row in vr if row["verdict"].get("content_clarity") is not None
                        ]),
                        "vlm_quality_mean": mean([
                            float(row["verdict"]["overall_quality"])
                            for row in vr if row["verdict"].get("overall_quality") is not None
                        ]),
                        "vlm_major_artifact_rate": mean([
                            float(row["verdict"].get("artifacts") == "major") for row in vr
                        ]),
                    }
                )
    print((OUTDIR / "aggregate.csv").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--require-complete", action="store_true")
    sub.add_parser("diagnostics")
    sub.add_parser("vlm")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--metrics-csv", type=Path, default=OUTDIR / "metrics.csv")
    args = parser.parse_args()
    rows = records()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.command == "inventory":
        inventory(rows)
    elif args.command == "manifest":
        manifest(rows, args.require_complete)
    elif args.command == "diagnostics":
        diagnostics(rows)
    elif args.command == "vlm":
        run_vlm(rows)
    elif args.command == "aggregate":
        aggregate(args.metrics_csv)


if __name__ == "__main__":
    main()
