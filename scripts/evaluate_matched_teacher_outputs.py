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
    parser.add_argument(
        "--csls-k",
        type=int,
        default=5,
        help=(
            "Requested k for the CSD+ CSLS readout (Frochte 2026, arXiv:2605.09030, Sec. 5). "
            "The paper's own default is k=15 (below their median per-artist anchor count of ~20); "
            "our style-reference gallery is far smaller, so the default here is lower and is still "
            "clamped to min(k, gallery_size - 1) at run time (the effective k is logged)."
        ),
    )
    parser.add_argument(
        "--csls-gallery-dir",
        default=None,
        help=(
            "Optional directory of extra style-reference images to fold into the CSLS r_k "
            "neighbourhood pool (e.g. the full style-reference set, not just the styles present "
            "in this manifest). Does NOT change s(x,A)=csd_style_sim, which still uses only the "
            "manifest's own style_path per row -- it only gives r_k a less tiny gallery to estimate "
            "local density over, closer to the paper's full-corpus reference pool. Effective k is "
            "still min(--csls-k, gallery_size - 1) and is logged."
        ),
    )
    parser.add_argument("--csls-gallery-glob", default="*.png", help="Glob for --csls-gallery-dir images.")
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


# --- CSD+ CSLS readout ------------------------------------------------------
#
# Source: Joerg Frochte, "When Style Similarity Scores Fail: Diagnosing Raw CSD
# Cosine in Artist-Style Evaluation," arXiv:2605.09030 (2026), Sec. 5 ("CSD+
# Readouts for Practical Evaluation"). Verbatim (OCR'd from the arXiv PDF):
#
#   "With r_k(x) = (1/k) * sum_{y in N_k(x)} <x, y> denoting the mean cosine of
#   x to its k nearest neighbours, CSLS(x, A) = 2 s(x, A) - r_k(x)
#   - (1/|A|) * sum_{a in A} r_k(a) is the doubly-locally-centred analogue of s."
#   [...] "We use k = 15 [3], below the median per-artist anchor count of 20;
#   counts and AUC are stable across k in {5, 10, 15, 20, 50}."
#   "[3]" is Conneau, Lample, Ranzato, Denoyer & Jegou, "Word Translation
#   Without Parallel Data" (ICLR 2018) -- the canonical source of CSLS, which
#   Frochte imports unmodified from cross-lingual lexicon induction.
#   The reference pool for r_k is "the candidate evaluation corpus itself"
#   (their full 91-artist anchor pool), not a bipartite source/target split.
#
# Our corpus has exactly one style-reference image per style_id (the anchor
# pool A is a singleton {ref}), so s(x, A) = cos(x, ref) is exactly the
# existing raw csd_style_sim column, and the formula collapses to the
# pairwise form CSLS(x, ref) = 2*cos(x, ref) - r_k(x) - r_k(ref).
# Per task instruction, our analogue of "the candidate evaluation corpus" is
# the gallery of all distinct style-reference images present in the manifest
# being evaluated (there is no larger corpus of per-style anchor images here).


def topk_mean_similarity(sim: torch.Tensor, k: int, exclude_mask: torch.Tensor | None = None) -> torch.Tensor:
    """r_k(x) for every row of a [Q, N] cosine-similarity matrix.

    r_k(x) = (1/k) * sum over the k largest entries of row x (Frochte 2026 Eq.
    in Sec. 5; orig. Conneau et al. 2018). ``exclude_mask`` (shape [Q, N], True
    = exclude) implements the leave-one-out self-exclusion the paper uses when
    a query is itself a gallery member ("cos(x,x) = 1 does not artificially
    inflate ..."): pass an identity mask when scoring the gallery against
    itself, and ``None`` when scoring queries that are not gallery members.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if exclude_mask is not None:
        sim = sim.masked_fill(exclude_mask, float("-inf"))
    topk = sim.topk(k, dim=1).values
    return topk.mean(dim=1)


def csls_score(s_x_a: float, r_k_x: float, mean_r_k_a: float) -> float:
    """CSLS(x, A) = 2*s(x, A) - r_k(x) - mean_{a in A} r_k(a).

    ``s_x_a`` is the (possibly pool-aggregated) raw cosine s(x, A); for our
    singleton pools this is just cos(x, ref) = csd_style_sim. ``mean_r_k_a``
    is the mean of r_k(a) over the anchor pool A (a single value when |A|=1).
    """
    return 2.0 * s_x_a - r_k_x - mean_r_k_a


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
        "csd_plus_csls",
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
        output_cache: dict[str, torch.Tensor] = {}
        for batch in rows_by_batch(self.rows, batch_size=8):
            output_embeds = embed_style([str(row["output_path"]) for row in batch])
            for idx, row in enumerate(batch):
                row["csd_style_sim"] = float(
                    cosine(output_embeds[idx : idx + 1], style_cache[str(row["style_path"])].unsqueeze(0)).item()
                )
                output_cache[str(row["output_path"])] = output_embeds[idx]

        # Optional CSLS gallery extension: fold in extra style-reference images
        # (e.g. the full style set, not just the styles present in this
        # manifest) so r_k has a less tiny neighbourhood to estimate local
        # density over -- closer to Frochte's "candidate evaluation corpus"
        # pool. Still the same loaded CSD model / embed_style() call, just
        # applied to more images; s(x,A)=csd_style_sim above is untouched.
        gallery_cache: dict[str, torch.Tensor] = dict(style_cache)
        n_extra = 0
        if self.args.csls_gallery_dir:
            gallery_root = Path(self.args.csls_gallery_dir)
            extra_paths = sorted(
                str(p) for p in gallery_root.glob(self.args.csls_gallery_glob)
                if p.is_file() and str(p) not in gallery_cache
            )
            if extra_paths:
                extra_embeds = embed_style(extra_paths)
                for path, embed in zip(extra_paths, extra_embeds):
                    gallery_cache[path] = embed
                n_extra = len(extra_paths)
            print(
                f"[csd+] CSLS gallery extended from {self.args.csls_gallery_dir}: "
                f"{len(style_cache)} manifest ref(s) + {n_extra} extra -> {len(gallery_cache)} total"
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

        # CSD+ CSLS readout, reusing the CSD embeddings computed above (no new
        # model; the only "extra" forward passes are the optional gallery-dir
        # images above, still through the same CSD backbone already loaded for
        # csd_style_sim).
        self._compute_csd_plus_csls(gallery_cache=gallery_cache, output_cache=output_cache, n_gallery_extra=n_extra)

    def _compute_csd_plus_csls(
        self,
        *,
        gallery_cache: dict[str, torch.Tensor],
        output_cache: dict[str, torch.Tensor],
        n_gallery_extra: int = 0,
    ) -> None:
        """CSD+ CSLS readout (Frochte 2026, arXiv:2605.09030, Sec. 5).

        See the module-level comment above ``topk_mean_similarity`` for the
        exact formula and source. Reference-pool substitution: Frochte's pool
        is "the candidate evaluation corpus itself" (their full 91-artist
        anchor set); our analogue is the gallery of style-reference images
        (by default just the manifest's own distinct style_paths; optionally
        extended via --csls-gallery-dir to a broader style-reference set),
        since our corpus has only one anchor image per style_id rather than a
        multi-image per-artist pool.
        """
        gallery_paths = sorted(gallery_cache.keys())
        n_gallery = len(gallery_paths)
        k_requested = int(self.args.csls_k)
        k_effective = min(k_requested, n_gallery - 1) if n_gallery > 1 else 0

        self.metric_info["csd_plus_csls"] = {
            "direction": "higher_better",
            "definition": (
                "CSLS(x,ref) = 2*cos(x,ref) - r_k(x) - r_k(ref); r_k(z) = mean cosine of z to its "
                "k nearest neighbours in the style-reference gallery (leave-one-out for gallery members)."
            ),
            "source": (
                "Frochte (2026), 'When Style Similarity Scores Fail', arXiv:2605.09030, Sec. 5 "
                "(CSD+ / CSLS readout); CSLS originates in Conneau et al. 2018, "
                "'Word Translation Without Parallel Data' (ICLR), their ref [3]."
            ),
            "gallery": (
                "distinct style_path images in this manifest, optionally extended via "
                "--csls-gallery-dir (task-directed substitution for the paper's full-corpus "
                "anchor pool, since our corpus has no multi-image per-style anchor set)"
            ),
            "gallery_size": n_gallery,
            "gallery_size_from_manifest": n_gallery - n_gallery_extra,
            "gallery_size_extra": n_gallery_extra,
            "k_requested": k_requested,
            "k_effective": k_effective,
        }

        if k_effective < 1:
            for row in self.rows:
                row["csd_plus_csls"] = None
            print(
                f"[csd+] CSLS skipped: gallery has only {n_gallery} distinct style reference(s), "
                "no neighbours available for r_k. csd_plus_csls left blank for all rows."
            )
            return

        gallery_embeds = torch.stack([gallery_cache[p] for p in gallery_paths])  # [N, D], L2-normalized
        gallery_sim = gallery_embeds @ gallery_embeds.t()  # [N, N] cosine (embeddings pre-normalized)
        self_mask = torch.eye(n_gallery, dtype=torch.bool)
        r_k_ref = topk_mean_similarity(gallery_sim, k_effective, exclude_mask=self_mask)  # [N]
        r_k_ref_by_path = {p: float(r_k_ref[i].item()) for i, p in enumerate(gallery_paths)}

        for row in self.rows:
            output_embed = output_cache[str(row["output_path"])].unsqueeze(0)  # [1, D]
            sim_to_gallery = output_embed @ gallery_embeds.t()  # [1, N]
            r_k_x = float(topk_mean_similarity(sim_to_gallery, k_effective, exclude_mask=None)[0].item())
            r_k_ref_val = r_k_ref_by_path[str(row["style_path"])]
            row["csd_plus_csls"] = csls_score(float(row["csd_style_sim"]), r_k_x, r_k_ref_val)

        print(f"[csd+] CSLS computed: gallery_size={n_gallery}, k_requested={k_requested}, k_effective={k_effective}")

        # --- Discrimination gap: DEFERRED, not fabricated ---------------------
        # Frochte's discrimination gap g_k = w_k - c_k requires, per artist k, a
        # within-class median cosine w_k computed over multiple own-artist anchor
        # artworks (off-diagonal pairs) and a cross-class statistic c_k = max over
        # every OTHER artist j of the median cosine between k's anchors and j's
        # anchors (typically ~90 other classes in their 91-artist corpus). Our
        # corpus provides exactly ONE reference image per style_id, so there is no
        # within-class pool to take an off-diagonal median over -- w_k is
        # undefined from reference images alone. Substituting our handful of
        # *generated* images sharing a style_id as a proxy anchor pool would
        # answer a different question (does our generator cluster by intended
        # style?) rather than Frochte's question (is CSD cosine intrinsically an
        # absolute same/different score on real artist artworks?), and with only
        # 3-8 other styles as the candidate "other artists" set, c_k = max over
        # 2-7 comparisons is far too noisy to support the paper's threshold
        # language (their gap>=0.15 "cleanly separated" bucket was estimated over
        # 90 other classes). We do not compute csd_discrimination_gap here; see
        # the run report for the honesty-gate reasoning.

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
            "csd_plus_csls": aggregate(rows, "csd_plus_csls"),
            "clip_t": aggregate(rows, "clip_t"),
            "dino_style_sim": aggregate(rows, "dino_style_sim"),
            "vgg_gram_dist": aggregate(rows, "vgg_gram_dist"),
            "dino_layout_leak": aggregate(rows, "dino_layout_leak"),
        },
        "csd_discrimination_gap": {
            "status": "deferred",
            "reason": (
                "Frochte (2026) g_k = w_k - c_k needs a multi-image within-class anchor pool per "
                "style (for the off-diagonal within-class median w_k) and ~90 other classes for the "
                "cross-class max c_k. Our corpus has exactly one reference image per style_id (no "
                "within-class pool) and only a handful of other styles per manifest, so the diagnostic "
                "is not computable in its published form here without fabricating a proxy. See code "
                "comment above MetricRunner._compute_csd_plus_csls for the full reasoning."
            ),
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
