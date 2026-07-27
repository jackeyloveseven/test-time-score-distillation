#!/usr/bin/env python3
"""Third-judge validation with the already-running local open-family VLM endpoint."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import statistics as stats
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STYLE_DIR = Path("/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty")
SOURCE = ROOT / "results/vlm_judge/verdicts.jsonl"
OUTDIR = ROOT / "results/vlm_judge"
OUTPUT = OUTDIR / "open_local_verdicts.jsonl"
SUMMARY = OUTDIR / "open_local_agreement.json"
BASE_URL = "http://127.0.0.1:18082/v1"
MODEL = "kaonai/grpo-oe-hint-c100"

PROMPT = (
    "Image 1 is a STYLE REFERENCE. Image 2 is a GENERATED image intentionally depicting "
    "a different subject. Ignore subject mismatch. Judge only artistic style transfer: "
    "medium, texture, brushwork or mark-making, palette, line quality, and rendering technique. "
    "Return one integer style_match from 1 to 5, where 5 means the same medium and technique "
    "are reproduced convincingly and 1 means the reference style is absent. Return ONLY valid "
    'JSON with this exact schema: {"style_match": 1, "note": "at most 15 words"}.'
)


def data_uri(path: Path, pixels: int = 512, quality: int = 84) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((pixels, pixels))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_json(text: str) -> dict[str, object]:
    matches = list(re.finditer(r"\{.*\}", text, re.DOTALL))
    if not matches:
        raise ValueError("response contains no JSON object")
    return json.loads(matches[-1].group(0))


def normalize_style_match(value: object) -> int | None:
    if isinstance(value, (int, float)):
        return max(1, min(5, int(round(float(value)))))
    if isinstance(value, dict):
        numeric = [float(item) for item in value.values() if isinstance(item, (int, float))]
        if numeric:
            return max(1, min(5, int(round(float(np.mean(numeric))))))
    return None


def judge(reference: Path, output: Path, retries: int = 4) -> dict[str, object]:
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri(reference)}},
                    {"type": "image_url", "image_url": {"url": data_uri(output)}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 350,
    }
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                BASE_URL + "/chat/completions",
                json=body,
                timeout=240,
            )
            if response.status_code in (408, 429) or response.status_code >= 500:
                last_error = f"http {response.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            payload = extract_json(content)
            score = normalize_style_match(payload.get("style_match"))
            if score is None:
                raise ValueError("style_match is missing or non-numeric")
            return {
                "style_match": score,
                "note": str(payload.get("note", ""))[:240],
                "raw": content,
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"[:240]
            time.sleep(2 * (attempt + 1))
    return {"style_match": None, "note": "ERR: " + last_error, "raw": ""}


def run(max_workers: int) -> None:
    source_rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8")]
    completed = {}
    if OUTPUT.exists():
        for line in OUTPUT.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                completed[row["output"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [row for row in source_rows if row["output"] not in completed]
    print(f"[open-judge] source={len(source_rows)} done={len(completed)} todo={len(todo)}")

    def work(row: dict[str, object]) -> dict[str, object]:
        output = ROOT / "results" / str(row["output"])
        correct = STYLE_DIR / str(row["correct_ref"])
        wrong = STYLE_DIR / str(row["wrong_ref"])
        return {
            "output": row["output"],
            "style_id": row.get("style_id"),
            "prompt_id": row.get("prompt_id"),
            "seed": row.get("seed"),
            "model": MODEL,
            "correct": judge(correct, output),
            "wrong_control": judge(wrong, output),
        }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(work, row): row for row in todo}
        with OUTPUT.open("a", encoding="utf-8") as handle:
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[open-judge] {index}/{len(todo)} {result['output'][:48]} "
                    f"correct={result['correct']['style_match']} "
                    f"wrong={result['wrong_control']['style_match']}"
                )


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return math.nan
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def agreement(left: list[float], right: list[float]) -> dict[str, float]:
    differences = [abs(a - b) for a, b in zip(left, right)]
    return {
        "n": len(differences),
        "pearson_r": pearson(left, right),
        "spearman_rho": pearson(rankdata(left).tolist(), rankdata(right).tolist()),
        "mean_absolute_difference": float(stats.mean(differences)) if differences else math.nan,
        "exact_agreement": float(np.mean(np.asarray(differences) == 0)) if differences else math.nan,
        "agreement_within_1": float(np.mean(np.asarray(differences) <= 1)) if differences else math.nan,
    }


def analyze() -> None:
    source = {row["output"]: row for row in (json.loads(line) for line in SOURCE.open(encoding="utf-8"))}
    opened = {row["output"]: row for row in (json.loads(line) for line in OUTPUT.open(encoding="utf-8"))}
    shared = sorted(set(source) & set(opened))

    triples = []
    correct_scores = []
    wrong_scores = []
    for key in shared:
        source_row = source[key]
        open_row = opened[key]
        local = open_row.get("correct", {}).get("style_match")
        wrong = open_row.get("wrong_control", {}).get("style_match")
        pro = source_row.get("verdicts", {}).get("gemini-pro", {}).get("style_match")
        flash = source_row.get("verdicts", {}).get("gemini-flash", {}).get("style_match")
        if all(isinstance(value, (int, float)) for value in (local, pro, flash)):
            triples.append((float(local), float(pro), float(flash)))
        if isinstance(local, (int, float)) and isinstance(wrong, (int, float)):
            correct_scores.append(float(local))
            wrong_scores.append(float(wrong))

    local_scores = [row[0] for row in triples]
    pro_scores = [row[1] for row in triples]
    flash_scores = [row[2] for row in triples]
    payload = {
        "model": MODEL,
        "endpoint": BASE_URL,
        "prompt": PROMPT,
        "n_source": len(source),
        "n_open": len(opened),
        "n_three_judge_shared": len(triples),
        "agreement": {
            "local_vs_gemini_pro": agreement(local_scores, pro_scores),
            "local_vs_gemini_flash": agreement(local_scores, flash_scores),
            "gemini_pro_vs_flash": agreement(pro_scores, flash_scores),
        },
        "wrong_reference_control": {
            "n": len(correct_scores),
            "correct_mean": float(stats.mean(correct_scores)) if correct_scores else math.nan,
            "wrong_mean": float(stats.mean(wrong_scores)) if wrong_scores else math.nan,
            "paired_gap": (
                float(stats.mean(a - b for a, b in zip(correct_scores, wrong_scores)))
                if correct_scores
                else math.nan
            ),
            "correct_higher_rate": (
                float(np.mean(np.asarray(correct_scores) > np.asarray(wrong_scores)))
                if correct_scores
                else math.nan
            ),
        },
    }
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.run:
        run(args.max_workers)
    if args.analyze:
        analyze()
    if not args.run and not args.analyze:
        parser.error("choose --run and/or --analyze")


if __name__ == "__main__":
    main()
