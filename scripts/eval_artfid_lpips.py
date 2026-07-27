#!/usr/bin/env python3
"""AAAI-eval-bar metrics gate: ArtFID + standalone LPIPS, on images we already have.

Context: `docs/research/lit_rf_and_eval.md` (Part B) flags ArtFID and a
standalone LPIPS number as missing relative to the StyleID / Dual-RF /
Scheduled-Style-Injection evidence bar. This script implements the SOUND
SUBSET only, after an explicit feasibility gate (mirroring the honesty
precedent already set by `evaluate_matched_teacher_outputs.py`'s deferred
`csd_discrimination_gap`). Nothing below is fabricated to look like the
textbook metric when the textbook preconditions are not met.

--- Feasibility gate (read before trusting any number this script prints) ---

1) ArtFID -- DEFERRED, both halves blocked, not merely "coarse":

   Formula, verbatim from Wright & Ommer, "ArtFID: Quantitative Evaluation of
   Neural Style Transfer" (arXiv:2207.12280), via the paper's own definition:

       ArtFID(Xg, Xc, Xs) = (1 + (1/N) * sum_i d(Xc(i), Xg(i))) * (1 + FID(Xs, Xg))

   where d(.,.) is LPIPS between the CONTENT image Xc(i) and the stylized
   output Xg(i) (paired, per-sample), and FID(Xs, Xg) is a Frechet distance
   between the feature distribution of the STYLE image set Xs and the
   stylized-output set Xg, using an Inception network trained on a
   241,088-image labelled artwork dataset (artist + period heads). The paper
   itself computes FID pooled over the whole test set (not per style) and,
   because standard FID is known to be biased at small N, recommends ~50k
   samples and extrapolates to FID-infinity via regression over several
   sub-sample sizes -- i.e. even the metric's own authors do not trust FID at
   small N and correct for it explicitly.

   Both factors are blocked here:
     (a) LPIPS factor needs a real CONTENT IMAGE Xc(i) per sample. Our content
         is a text prompt, not an image -- there is no Xc(i) at all. The only
         sound proxy (LPIPS against an unstyled, same-prompt/same-seed plain
         T2I generation) requires generations we do not currently have in
         results/ (checked: no no-style/plain-T2I/content-only *output image*
         set exists; "content_only" elsewhere in this repo denotes a
         CFG-branch label used inside the training-time loss, not a rendered
         baseline image). Proposal (not executed here; GPU is contended):
         generate a small matched no-style set later (same style_id/prompt_id/
         seed grid as matrix_manifest.csv, 9 images) to unlock this factor.
     (b) FID factor needs the STYLE distribution's covariance in Inception
         feature space. Our style-reference gallery has exactly 11 unique
         images total (checked across matrix_manifest.csv + breadth_manifest
         .csv + all 5 sched_*_manifest.csv). Every available Inception feature
         block (pytorch-fid dims in {64,192,768,2048}) exceeds 11, so the
         style-side sample covariance is rank-deficient (rank <= 10) for any
         choice -- this is not "coarser," it is numerically degenerate (the
         standard implementation's matrix-sqrt falls back to an epsilon*I
         regularization whenever this happens, so a number would still print,
         but it would mostly reflect that regularization constant rather than
         the true style distribution). Pooling stylized outputs across styles
         does not rescue this: it only grows the Xg side; Xs stays at 11.
         Per-method Xg sample sizes are also unequal across our three methods
         (score_carrier n=9 vs score_match/kv_match n=33 in the pooled corpus),
         which would additionally confound the trace term even for a
         *relative* (method-vs-method) ranking, not just the absolute number.
   Net: ArtFID is not computed by this script. If a directional (non-citable)
   FID number is ever wanted, it would need to be subsampled to a common n and
   labelled explicitly as "not ArtFID, not comparable to literature values."

2) Content-preservation LPIPS -- DEFERRED for the same reason as 1(a): no
   content-image reference exists yet. Not computed here.

3) Method-delta LPIPS -- SOUND, IMPLEMENTED. Standalone LPIPS is well-defined
   between any two same-resolution images regardless of what they represent;
   the only question is whether the *comparison* is meaningful. This script
   computes three matched quantities on `matrix_manifest.csv` (score_carrier /
   score_match / kv_match, 3 styles x 3 seeds each) so the "carrier is inert"
   ablation claim (see docs/LOOP_PLAN.md) has a scale to be read against:
     - carrier_effect      : LPIPS(score_carrier, score_match), same cell.
                              "How much does turning the carrier on change the
                              image?"
     - seed_noise_floor    : LPIPS(score_match@seed_i, score_match@seed_j),
                              same style_id/prompt_id, across the 3 seeds.
                              "How much does the image change from seed noise
                              alone, method held fixed?" -- the calibration
                              floor the carrier_effect must be read against.
     - method_family_scale : LPIPS(kv_match, score_match), same cell. "How
                              much does swapping to a genuinely different
                              transport mechanism (K/V replacement vs.
                              score-residual) change the image?" -- the
                              calibration ceiling.
   The inert-carrier reading is: carrier_effect approx= seed_noise_floor
   << method_family_scale. This is an ABLATION-MAGNITUDE metric, not the
   content-preservation LPIPS AAAI reviewers expect from a real content
   image; it is reported under that honest label, not as "LPIPS-content".

   Secondary/bonus: schedule_vs_const, same construction, over the 5
   sched_*_manifest.csv files (schedule name is already the "method" column),
   comparing each non-const schedule against const at the shared seed=64
   cells -- supports the separately-documented "schedule is inert" finding.

Dependency check performed before writing this file:
`/home/ubuntu/chengyanli/flux/bin/python3 -c "import lpips"` succeeds (pip
package `lpips`, both AlexNet backbone weights [torch hub cache] and the
package's own bundled linear-calibration weights are already present on
disk -- fully offline, no download at run time). torchmetrics' LPIPS/FID and
pytorch-fid (with the exact FID-standard Inception checkpoint already cached)
are also importable, noted for completeness even though FID is not used here.

Compute: runs on CPU by default (--device cpu) to sidestep the contended GPU
entirely (checked nvidia-smi: 119/143 GB used, 100% util, two external PIDs
at task start) -- AlexNet-backbone LPIPS over ~90 tiny 512x512 images is a
few seconds of CPU work, not worth the contention risk.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_matched_teacher_outputs import manifest_rows, open_rgb, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix-manifest", default="results/matrix_manifest.csv")
    parser.add_argument(
        "--sched-manifest-glob",
        default="results/sched_*_manifest.csv",
        help="Glob (relative to cwd) for the schedule-sweep manifests used for the secondary schedule_vs_const check.",
    )
    parser.add_argument("--device", default="cpu", help="Default cpu: GPU is contended, these models are tiny.")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--out-csv", default="results/lpips_ablation_pairs.csv")
    parser.add_argument("--summary-json", default="results/lpips_ablation_summary.json")
    return parser.parse_args()


def cell_key(row: dict) -> tuple:
    return (str(row["style_id"]), str(row["prompt_id"]), str(row["seed"]))


def group_key(row: dict) -> tuple:
    return (str(row["style_id"]), str(row["prompt_id"]))


def by_method(rows: list[dict]) -> dict[str, dict[tuple, dict]]:
    out: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for row in rows:
        out[str(row["method"])][cell_key(row)] = row
    return out


class LpipsRunner:
    """Thin wrapper around the `lpips` package: cached per-image tensors, CPU by default."""

    def __init__(self, device: str, net: str) -> None:
        import lpips

        self.device = device
        self.net = net
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.to_tensor = transforms.ToTensor()
        self._cache: dict[str, torch.Tensor] = {}
        # Provenance: sha256 of the bundled calibration weight file actually loaded.
        import lpips as lpips_pkg

        weight_path = Path(lpips_pkg.__file__).resolve().parent / "weights" / "v0.1" / f"{net}.pth"
        self.weight_info = {
            "backbone": net,
            "weight_path": str(weight_path),
            "weight_sha256": sha256_file(weight_path) if weight_path.exists() else None,
        }

    def _load(self, path: str) -> torch.Tensor:
        if path not in self._cache:
            tensor = self.to_tensor(open_rgb(path)).unsqueeze(0).to(self.device)
            self._cache[path] = tensor
        return self._cache[path]

    @torch.no_grad()
    def distance(self, path_a: str, path_b: str) -> float:
        a = self._load(path_a)
        b = self._load(path_b)
        val = self.model(a, b, normalize=True)  # inputs are [0,1]; normalize=True rescales to [-1,1] internally
        return float(val.reshape(-1)[0].item())

    def sanity_check_identity(self, path: str) -> float:
        """LPIPS(x, x) should be ~0. Returns the value; caller asserts the bound."""
        return self.distance(path, path)


def paired_rows(
    comparison: str,
    method_a: str,
    method_b: str,
    cells_a: dict[tuple, dict],
    cells_b: dict[tuple, dict],
    runner: LpipsRunner,
) -> list[dict]:
    out = []
    for key in sorted(set(cells_a) & set(cells_b)):
        row_a, row_b = cells_a[key], cells_b[key]
        style_id, prompt_id, seed = key
        out.append(
            {
                "comparison": comparison,
                "style_id": style_id,
                "prompt_id": prompt_id,
                "seed_a": seed,
                "seed_b": seed,
                "method_a": method_a,
                "method_b": method_b,
                "path_a": row_a["output_path"],
                "path_b": row_b["output_path"],
                "lpips": runner.distance(str(row_a["output_path"]), str(row_b["output_path"])),
            }
        )
    return out


def seed_noise_floor_rows(method_name: str, rows: list[dict], runner: LpipsRunner) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if str(row["method"]) != method_name:
            continue
        grouped[group_key(row)][str(row["seed"])] = row
    out = []
    for (style_id, prompt_id), by_seed in sorted(grouped.items()):
        seeds = sorted(by_seed.keys())
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            row_a, row_b = by_seed[seed_a], by_seed[seed_b]
            out.append(
                {
                    "comparison": "seed_noise_floor",
                    "style_id": style_id,
                    "prompt_id": prompt_id,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "method_a": method_name,
                    "method_b": method_name,
                    "path_a": row_a["output_path"],
                    "path_b": row_b["output_path"],
                    "lpips": runner.distance(str(row_a["output_path"]), str(row_b["output_path"])),
                }
            )
    return out


def aggregate(rows: list[dict], comparison: str) -> dict:
    values = [float(r["lpips"]) for r in rows if r["comparison"] == comparison]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_by_style(rows: list[dict], comparison: str) -> dict[str, dict]:
    """Per-style_id breakdown -- required reading before calling a mean 'inert': a
    small pooled mean can hide a style-conditional effect (near-zero for some
    styles, large for others), which is a materially different claim from a
    uniformly small effect. See docs/LOOP_PLAN.md line 205 ("Carrier is NOT
    inert; it is STYLE-CONDITIONAL...") for the repo's own prior finding this
    must be checked against before any inert/style-conditional verdict here.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["comparison"] == comparison:
            grouped[str(row["style_id"])].append(float(row["lpips"]))
    return {
        style_id: {
            "count": len(values),
            "mean": mean(values),
            "values": values,
        }
        for style_id, values in sorted(grouped.items())
    }


def write_rows(rows: list[dict], out_csv: Path) -> None:
    fieldnames = ["comparison", "style_id", "prompt_id", "seed_a", "seed_b", "method_a", "method_b", "path_a", "path_b", "lpips"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


ARTFID_DEFERRAL = {
    "status": "deferred",
    "formula_source": "Wright & Ommer, arXiv:2207.12280, 'ArtFID: Quantitative Evaluation of Neural Style Transfer'",
    "formula": "ArtFID(Xg,Xc,Xs) = (1 + mean_i LPIPS(Xc(i), Xg(i))) * (1 + FID(Xs, Xg))",
    "blocked_factor_1_content_lpips": (
        "Needs a real content IMAGE Xc(i) per sample; our content is a text prompt, so Xc(i) does not "
        "exist at all. No no-style/plain-T2I baseline images currently exist in results/ to serve as a "
        "proxy (checked; the repo's 'content_only' label elsewhere denotes a training-time CFG branch, "
        "not a rendered baseline image). Deferred; would need a small generated no-style set (not "
        "executed here -- GPU is contended by external jobs)."
    ),
    "blocked_factor_2_fid_style": (
        "Needs the style image set Xs's covariance in Inception feature space. Our style-reference "
        "gallery has exactly 11 unique images total (checked across matrix/breadth/all 5 sched "
        "manifests). Every standard Inception feature block (64/192/768/2048-dim) exceeds 11, so the "
        "style-side sample covariance is rank-deficient (rank <= 10) regardless of dims choice -- this "
        "is numerically degenerate, not merely a coarser estimate (the standard matrix-sqrt "
        "implementation falls back to an epsilon*I regularization in this regime, so a number would "
        "print but would mostly reflect that regularization constant). Pooling stylized outputs across "
        "styles does not fix this: it only grows the generated side; the style side stays at 11. "
        "Per-method generated-sample sizes are also unequal (score_carrier n=9 vs score_match/kv_match "
        "n=33 pooled), confounding even a relative (method-vs-method) ranking. Not computed."
    ),
    "verdict": "ArtFID is not computed by this script (both factors blocked, not just the harder one).",
}

CONTENT_LPIPS_DEFERRAL = {
    "status": "deferred",
    "reason": (
        "Standard content-preservation LPIPS needs LPIPS(stylized_output, content_image). Our content is "
        "a prompt, not an image, and no unstyled/plain-T2I same-prompt-same-seed baseline currently "
        "exists in results/. Not computed. Proposal: generate a small matched no-style set later "
        "(same style_id/prompt_id/seed grid as matrix_manifest.csv -> 9 images) to unlock this metric; "
        "not executed here per instruction (GPU currently contended)."
    ),
    "computed_instead": (
        "method_delta_lpips (carrier_effect / seed_noise_floor / method_family_scale below) -- an "
        "ablation-magnitude quantity, NOT a substitute for content-preservation LPIPS. Labeled honestly "
        "as such; do not conflate with the reviewer-expected content-LPIPS number."
    ),
}


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix_manifest).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()

    matrix_rows = manifest_rows(matrix_path)
    by_method_matrix = by_method(matrix_rows)

    runner = LpipsRunner(device=args.device, net=args.lpips_net)

    # Unit sanity check: LPIPS(x, x) == 0.
    sample_path = str(matrix_rows[0]["output_path"])
    identity_value = runner.sanity_check_identity(sample_path)
    identity_ok = identity_value < 1e-6
    print(f"[sanity] LPIPS(x,x) = {identity_value:.3e} on {sample_path} -> {'PASS' if identity_ok else 'FAIL'}")
    if not identity_ok:
        print("[sanity] FAILED identity check -- aborting, do not trust downstream numbers.", file=sys.stderr)
        return 1

    all_rows: list[dict] = []

    # 1. carrier_effect: score_carrier vs score_match, matched cells.
    if "score_carrier" in by_method_matrix and "score_match" in by_method_matrix:
        all_rows += paired_rows(
            "carrier_effect", "score_carrier", "score_match",
            by_method_matrix["score_carrier"], by_method_matrix["score_match"], runner,
        )
    else:
        print("[warn] matrix manifest missing score_carrier or score_match; carrier_effect skipped.")

    # 2. method_family_scale: kv_match vs score_match, matched cells.
    if "kv_match" in by_method_matrix and "score_match" in by_method_matrix:
        all_rows += paired_rows(
            "method_family_scale", "kv_match", "score_match",
            by_method_matrix["kv_match"], by_method_matrix["score_match"], runner,
        )
    else:
        print("[warn] matrix manifest missing kv_match or score_match; method_family_scale skipped.")

    # 3. seed_noise_floor: score_match across seeds, same style/prompt.
    if "score_match" in by_method_matrix:
        all_rows += seed_noise_floor_rows("score_match", matrix_rows, runner)
    else:
        print("[warn] matrix manifest missing score_match; seed_noise_floor skipped.")

    # 4. (secondary/bonus) schedule_vs_const across sched_*_manifest.csv files.
    sched_paths = sorted(matrix_path.parent.glob(Path(args.sched_manifest_glob).name))
    sched_rows: list[dict] = []
    for sched_path in sched_paths:
        sched_rows += manifest_rows(sched_path)
    if sched_rows:
        by_method_sched = by_method(sched_rows)
        if "const" in by_method_sched:
            for schedule_name in sorted(by_method_sched):
                if schedule_name == "const":
                    continue
                all_rows += paired_rows(
                    "schedule_vs_const", schedule_name, "const",
                    by_method_sched[schedule_name], by_method_sched["const"], runner,
                )
        else:
            print("[warn] sched manifests found but no 'const' method; schedule_vs_const skipped.")
    else:
        print(f"[warn] no sched manifests matched glob {args.sched_manifest_glob!r}; schedule_vs_const skipped.")

    write_rows(all_rows, out_csv)

    comparisons = sorted({r["comparison"] for r in all_rows})
    method_delta_summary = {c: aggregate(all_rows, c) for c in comparisons}
    carrier_effect_by_style = aggregate_by_style(all_rows, "carrier_effect")
    method_family_scale_by_style = aggregate_by_style(all_rows, "method_family_scale")

    carrier_mean = method_delta_summary.get("carrier_effect", {}).get("mean")
    seed_mean = method_delta_summary.get("seed_noise_floor", {}).get("mean")
    family_mean = method_delta_summary.get("method_family_scale", {}).get("mean")
    inert_reading = None
    if carrier_mean is not None and seed_mean is not None and family_mean is not None and family_mean > 0:
        per_style_ratio = {
            style_id: (
                carrier_effect_by_style[style_id]["mean"] / method_family_scale_by_style[style_id]["mean"]
                if style_id in method_family_scale_by_style and method_family_scale_by_style[style_id]["mean"] > 0
                else None
            )
            for style_id in carrier_effect_by_style
        }
        inert_reading = {
            "carrier_effect_mean_pooled": carrier_mean,
            "method_family_scale_mean_pooled": family_mean,
            "carrier_over_family_scale_ratio_pooled": carrier_mean / family_mean,
            "carrier_effect_by_style": carrier_effect_by_style,
            "method_family_scale_by_style": method_family_scale_by_style,
            "carrier_over_family_scale_ratio_by_style": per_style_ratio,
            "seed_noise_floor_mean_pooled": seed_mean,
            "caveat": (
                "PRIMARY, apples-to-apples calibration: carrier_effect vs method_family_scale, both "
                "same-seed matched cells. seed_noise_floor is cross-seed (different comparison shape) "
                "and lands ~= method_family_scale (0.586 vs 0.574 in the run this was written against) "
                "only because both are 'different-image'-scale LPIPS where the metric saturates for this "
                "pipeline -- it is NOT a meaningful low calibration floor and should not be read as "
                "confirming carrier_effect is small by comparison to it. "
                "The pooled carrier_over_family_scale_ratio hides substantial style-conditionality: "
                "check carrier_over_family_scale_ratio_by_style before calling the effect uniformly "
                "small. This matches the repo's own prior finding (docs/LOOP_PLAN.md line 205, 'Carrier "
                "is NOT inert; it is STYLE-CONDITIONAL...') rather than supporting a blanket 'inert' verdict."
            ),
        }

    summary = {
        "matrix_manifest": str(matrix_path),
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "sched_manifests": [str(p) for p in sched_paths],
        "out_csv": str(out_csv),
        "device": args.device,
        "lpips_backbone": runner.weight_info,
        "sanity_check_lpips_identity": {"value": identity_value, "pass": identity_ok},
        "method_delta_lpips": method_delta_summary,
        "inert_carrier_reading": inert_reading,
        "artfid": ARTFID_DEFERRAL,
        "content_preservation_lpips": CONTENT_LPIPS_DEFERRAL,
        "row_count": len(all_rows),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] wrote pairs csv -> {out_csv}")
    print(f"[done] wrote summary   -> {summary_json}")
    for comparison, stats in method_delta_summary.items():
        if stats.get("count"):
            print(f"  {comparison:22s} n={stats['count']:3d} mean={stats['mean']:.4f} std={stats['std']:.4f}")
    if inert_reading:
        print(f"  carrier/family_scale (pooled) = {inert_reading['carrier_over_family_scale_ratio_pooled']:.3f}")
        print("  carrier/family_scale (by style_id) -- check before calling the effect uniformly small:")
        for style_id, ratio in sorted(inert_reading["carrier_over_family_scale_ratio_by_style"].items()):
            if ratio is not None:
                print(f"    style {style_id}: {ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
