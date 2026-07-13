#!/usr/bin/env python3
"""Evaluate matched-teacher outputs from a manifest of existing images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = ("method", "output_path", "style_path", "prompt", "style_id", "prompt_id", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csd-root", default="/home/ubuntu/chengyanli/items/CSD_Score")
    parser.add_argument("--clip-root", default="/home/ubuntu/chengyanli/hfmodels/clip")
    parser.add_argument("--dino-model-name", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--dino-hf-cache-root", default=str(Path.home() / ".cache" / "huggingface" / "hub"))
    parser.add_argument("--torch-cache-root", default=str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}:{stat.st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def maybe_sha(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return sha256_tree(path)
    return None


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return (a * b).sum(dim=-1)


def rows_by_batch(rows: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    out: list[list[dict[str, object]]] = []
    for idx in range(0, len(rows), batch_size):
        out.append(rows[idx : idx + batch_size])
    return out


def manifest_rows(manifest_csv: Path) -> list[dict[str, object]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest_csv} has no header")
        missing = [field for field in REQUIRED_COLUMNS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{manifest_csv} missing required columns: {missing}")
        rows: list[dict[str, object]] = []
        for raw in reader:
            row = dict(raw)
            row["output_path"] = str(Path(str(row["output_path"])).expanduser())
            row["style_path"] = str(Path(str(row["style_path"])).expanduser())
            if not Path(str(row["output_path"])).exists():
                raise FileNotFoundError(row["output_path"])
            if not Path(str(row["style_path"])).exists():
                raise FileNotFoundError(row["style_path"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{manifest_csv} did not contain any rows")
    return rows


def aggregate(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        grouped[str(row["method"])].append(float(value))
    out: dict[str, dict[str, float | int | None]] = {}
    for method, values in sorted(grouped.items()):
        out[method] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return out


def resolve_hf_snapshot(cache_root: Path, repo_dir_name: str) -> str | None:
    repo_root = cache_root / repo_dir_name
    refs_main = repo_root / "refs" / "main"
    if not refs_main.exists():
        return None
    revision = refs_main.read_text(encoding="utf-8").strip()
    snapshot = repo_root / "snapshots" / revision
    return str(snapshot) if snapshot.exists() else None


def resolve_dino_cache_path(args: argparse.Namespace) -> str | None:
    candidates = [
        resolve_hf_snapshot(Path(args.dino_hf_cache_root), "models--facebook--dinov2-base"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def infer_vgg_weight_path(torch_cache_root: Path) -> str | None:
    candidate = torch_cache_root / "vgg19-dcbb9e9d.pth"
    return str(candidate) if candidate.exists() else None


def write_rows(rows: list[dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "method",
        "output_path",
        "style_path",
        "prompt",
        "style_id",
        "prompt_id",
        "seed",
        "csd_style_sim",
        "clip_t",
        "dino_style_sim",
        "vgg_gram_dist",
        "dino_layout_leak",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class MetricRunner:
    def __init__(self, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
        self.args = args
        self.rows = rows
        self.device = args.device
        self.model_info: dict[str, dict[str, object]] = {}
        self.metric_info = {
            "csd_style_sim": {
                "direction": "higher_better",
                "preprocess": "ToTensor -> bilinear resize 224x224 -> CLIP normalize, no crop",
            },
            "clip_t": {
                "direction": "higher_better",
                "preprocess": "Resize 224 bicubic -> center crop 224 -> ToTensor -> CLIP normalize",
            },
            "dino_style_sim": {
                "direction": "higher_better",
                "preprocess": "ToTensor -> bilinear resize 518x518 -> ImageNet normalize, no crop",
            },
            "vgg_gram_dist": {
                "direction": "lower_better",
                "preprocess": "ToTensor -> bilinear resize 224x224 -> ImageNet normalize, no crop",
            },
            "dino_layout_leak": {
                "direction": "higher_more_layout_leak",
                "preprocess": "Same DINO preprocessing as dino_style_sim; Pearson corr of off-diagonal patch self-similarity",
            },
        }

    def run(self) -> None:
        self._compute_csd()
        self._compute_clip_t()
        self._compute_dino_metrics()
        self._compute_vgg_gram()

    def _compute_csd(self) -> None:
        sys.path.insert(0, str(Path(self.args.csd_root)))
        from model import CSD_CLIP, convert_state_dict

        csd_root = Path(self.args.csd_root)
        clip_weight = csd_root / "models" / "ViT-L-14.pt"
        checkpoint = csd_root / "models" / "checkpoint.pth"
        model = CSD_CLIP("vit_large", "default", model_path=str(clip_weight))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(convert_state_dict(state["model_state_dict"]), strict=False)
        model = model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        clip_mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        clip_std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        to_tensor = transforms.ToTensor()

        @torch.no_grad()
        def embed_style(paths: list[str]) -> torch.Tensor:
            batch = torch.stack([to_tensor(open_rgb(path)) for path in paths]).to(self.device)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - clip_mean) / clip_std
            _, _, style = model(batch.float())
            return F.normalize(style.float(), dim=-1).cpu()

        style_cache = {path: embed_style([path])[0] for path in sorted({str(r["style_path"]) for r in self.rows})}
        for batch in rows_by_batch(self.rows, batch_size=8):
            output_embeds = embed_style([str(row["output_path"]) for row in batch])
            for idx, row in enumerate(batch):
                row["csd_style_sim"] = float(
                    cosine(output_embeds[idx : idx + 1], style_cache[str(row["style_path"])].unsqueeze(0)).item()
                )

        self.model_info["csd"] = {
            "model_name": "CSD_CLIP(vit_large, default)",
            "style_clip_weight_path": str(clip_weight),
            "style_clip_weight_sha256": sha256_file(clip_weight),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_clip_t(self) -> None:
        import open_clip

        model_name = "ViT-B-32"
        clip_root = Path(self.args.clip_root)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        model = model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )

        @torch.no_grad()
        def embed_images(paths: list[str]) -> torch.Tensor:
            batch = torch.stack([preprocess(open_rgb(path)) for path in paths]).to(self.device)
            return F.normalize(model.encode_image(batch).float(), dim=-1).cpu()

        @torch.no_grad()
        def embed_texts(texts: list[str]) -> torch.Tensor:
            tokens = tokenizer(texts).to(self.device)
            return F.normalize(model.encode_text(tokens).float(), dim=-1).cpu()

        prompt_cache = {prompt: embed_texts([prompt])[0] for prompt in sorted({str(r["prompt"]) for r in self.rows})}
        for batch in rows_by_batch(self.rows, batch_size=8):
            output_embeds = embed_images([str(row["output_path"]) for row in batch])
            for idx, row in enumerate(batch):
                row["clip_t"] = float(
                    cosine(output_embeds[idx : idx + 1], prompt_cache[str(row["prompt"])].unsqueeze(0)).item()
                )

        self.model_info["clip_t"] = {
            "model_name": f"open_clip:{model_name}",
            "pretrained_tag": "openai",
            "local_root_path": str(clip_root),
            "local_root_sha256": sha256_tree(clip_root),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_dino_metrics(self) -> None:
        import timm

        dino = timm.create_model(self.args.dino_model_name, pretrained=True, num_classes=0).to(self.device).eval()
        n_prefix = int(getattr(dino, "num_prefix_tokens", 1))
        to_tensor = transforms.ToTensor()
        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        tri_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        @torch.no_grad()
        def dino_tokens(paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            batch = torch.stack([to_tensor(open_rgb(path)) for path in paths]).to(self.device)
            batch = F.interpolate(batch, size=(518, 518), mode="bilinear", align_corners=False)
            batch = (batch - imagenet_mean) / imagenet_std
            features = dino.forward_features(batch)
            if isinstance(features, dict):
                if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
                    cls = features["x_norm_clstoken"]
                    patches = features["x_norm_patchtokens"]
                elif "x_prenorm" in features:
                    tokens = features["x_prenorm"]
                    cls = tokens[:, 0]
                    patches = tokens[:, n_prefix:]
                else:
                    raise RuntimeError(f"Unsupported timm DINO forward_features output keys: {sorted(features.keys())}")
            else:
                cls = features[:, 0]
                patches = features[:, n_prefix:]
            return cls.float().cpu(), patches.float().cpu()

        def layout_leakage(patches_out: torch.Tensor, patches_sty: torch.Tensor) -> float:
            p_out = F.normalize(patches_out.float(), dim=-1)
            p_sty = F.normalize(patches_sty.float(), dim=-1)
            sim_out = p_out @ p_out.t()
            sim_sty = p_sty @ p_sty.t()
            n = sim_out.shape[0]
            tri = tri_cache.get(n)
            if tri is None:
                tri = torch.triu_indices(n, n, offset=1, device=sim_out.device)
                tri_cache[n] = tri
            a = sim_out[tri[0], tri[1]]
            b = sim_sty[tri[0], tri[1]]
            a = a - a.mean()
            b = b - b.mean()
            denom = (a.norm() * b.norm()).clamp_min(1e-8)
            return float(((a * b).sum() / denom).item())

        style_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for path in sorted({str(r["style_path"]) for r in self.rows}):
            cls, patches = dino_tokens([path])
            style_cache[path] = (cls[0], patches[0])

        for batch in rows_by_batch(self.rows, batch_size=4):
            cls_batch, patches_batch = dino_tokens([str(row["output_path"]) for row in batch])
            for idx, row in enumerate(batch):
                sty_cls, sty_patches = style_cache[str(row["style_path"])]
                row["dino_style_sim"] = float(
                    cosine(cls_batch[idx : idx + 1], sty_cls.unsqueeze(0)).item()
                )
                row["dino_layout_leak"] = layout_leakage(patches_batch[idx], sty_patches)

        self.model_info["dino"] = {
            "model_name": self.args.dino_model_name,
            "resolved_cache_path": resolve_dino_cache_path(self.args),
            "resolved_cache_sha256": maybe_sha(resolve_dino_cache_path(self.args)),
        }
        del dino
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_vgg_gram(self) -> None:
        from torchvision.models import VGG19_Weights, vgg19

        weights = VGG19_Weights.IMAGENET1K_V1
        vgg = vgg19(weights=weights).features.to(self.device).eval()
        for parameter in vgg.parameters():
            parameter.requires_grad_(False)
        to_tensor = transforms.ToTensor()
        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        gram_layers = {1, 6, 11, 20, 29}

        @torch.no_grad()
        def vgg_grams(paths: list[str]) -> list[list[torch.Tensor]]:
            batch = torch.stack([to_tensor(open_rgb(path)) for path in paths]).to(self.device)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - imagenet_mean) / imagenet_std
            outputs: list[list[torch.Tensor]] = [[] for _ in paths]
            h = batch
            for idx, layer in enumerate(vgg):
                h = layer(h)
                if idx in gram_layers:
                    bsz, channels, hh, ww = h.shape
                    feats = h.view(bsz, channels, hh * ww)
                    grams = torch.matmul(feats, feats.transpose(1, 2)) / float(channels * hh * ww)
                    for batch_idx in range(bsz):
                        outputs[batch_idx].append(grams[batch_idx].float().cpu())
                if idx >= max(gram_layers):
                    break
            return outputs

        style_cache = {path: vgg_grams([path])[0] for path in sorted({str(r["style_path"]) for r in self.rows})}
        for batch in rows_by_batch(self.rows, batch_size=4):
            output_grams = vgg_grams([str(row["output_path"]) for row in batch])
            for idx, row in enumerate(batch):
                dists = [
                    torch.norm(go - gs).item()
                    for go, gs in zip(output_grams[idx], style_cache[str(row["style_path"])])
                ]
                row["vgg_gram_dist"] = float(mean(dists))

        vgg_weight_path = infer_vgg_weight_path(Path(self.args.torch_cache_root))
        self.model_info["vgg_gram"] = {
            "model_name": "torchvision:vgg19",
            "weights_name": "IMAGENET1K_V1",
            "resolved_weight_path": vgg_weight_path,
            "resolved_weight_sha256": maybe_sha(vgg_weight_path),
        }
        del vgg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summary_payload(
    *,
    args: argparse.Namespace,
    manifest_csv: Path,
    out_csv: Path,
    rows: list[dict[str, object]],
    runner: MetricRunner,
) -> dict[str, object]:
    return {
        "manifest_csv": str(manifest_csv),
        "manifest_sha256": sha256_file(manifest_csv),
        "out_csv": str(out_csv),
        "row_count": len(rows),
        "method_count": len({str(row["method"]) for row in rows}),
        "style_reference_count": len({str(row["style_path"]) for row in rows}),
        "device": args.device,
        "model_info": runner.model_info,
        "metric_info": runner.metric_info,
        "method_summary": {
            "csd_style_sim": aggregate(rows, "csd_style_sim"),
            "clip_t": aggregate(rows, "clip_t"),
            "dino_style_sim": aggregate(rows, "dino_style_sim"),
            "vgg_gram_dist": aggregate(rows, "vgg_gram_dist"),
            "dino_layout_leak": aggregate(rows, "dino_layout_leak"),
        },
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else manifest_csv.with_name(f"{manifest_csv.stem}_metrics.csv")
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else manifest_csv.with_name(f"{manifest_csv.stem}_metrics_summary.json")
    )

    rows = manifest_rows(manifest_csv)
    runner = MetricRunner(args, rows)
    runner.run()
    write_rows(rows, out_csv)
    summary = summary_payload(args=args, manifest_csv=manifest_csv, out_csv=out_csv, rows=rows, runner=runner)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] wrote metrics csv -> {out_csv}")
    print(f"[done] wrote summary   -> {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
