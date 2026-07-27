#!/usr/bin/env python3
"""Run 3-triplet benchmark (8 dims including style_fidelity) and print HTML+text report."""

import json, os
import benchmark as bm

for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

# 3 clean triplets with known-good URLs
TRIPLETS = [
    {
        "id": "ST_NIJI_PORTRAIT",
        "desc": "写实人像 → Niji Anime",
        "content": "https://image-cdn.flowgpt.com/generated_images/image-fe1fae8e-f7b1-44de-895b-0461954dd779.png",
        "style":   "https://image-cdn.flowgpt.com/generated_images/image-78a9f2e8-ddf8-43fe-bfce-70c8e9d44041.png",
        "output":  "https://image-cdn.flowgpt.com/generated_images/image-409bc5d9-0a09-41f7-8a9e-cb9c2b89f0da.jpg",
    },
    {
        "id": "ST_WATERCOLOR_LANDSCAPE",
        "desc": "写实风景 → 水彩插画",
        "content": "https://image-cdn.flowgpt.com/generated_images/image-0bc6ac78-7ac8-486c-aa39-80cc4b1a4aa1.jpg",
        "style":   "https://image-cdn.flowgpt.com/generated_images/image-405d80c6-79c4-4270-8833-8d2715b9b509.jpg",
        "output":  "https://image-cdn.flowgpt.com/generated_images/image-fd5265e2-4c6c-4a05-82da-31a7fb8e632d.jpg",
    },
    {
        "id": "ST_INKWASH_PORTRAIT",
        "desc": "写实人像 → 中国水墨",
        "content": "https://image-cdn.flowgpt.com/generated_images/image-a9b14487-a0cb-42f9-ab7c-c6ea772e208e.jpg",
        "style":   "https://image-cdn.flowgpt.com/generated_images/image-f07d6426-c7bc-404d-84bc-3f16b6fc31a9.jpg",
        "output":  "https://image-cdn.flowgpt.com/generated_images/image-31b6487e-1112-4a77-8a7f-204dbb39d6ec.jpg",
    },
]

results = []
for t in TRIPLETS:
    print(f"\n{'='*60}")
    print(f"[{t['id']}] {t['desc']}")
    r = bm.evaluate(pair_id=t["id"], content_url=t["content"],
                    style_url=t["style"], output_url=t["output"])
    r["desc"] = t["desc"]
    r["urls"] = {"content": t["content"], "style": t["style"], "output": t["output"]}
    results.append(r)

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open("/Users/jackeylove/style/final_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ── Text report ───────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("STYLE TRANSFER BENCHMARK REPORT  —  8-Dimension CoT Evaluation")
print("="*70)

for r in results:
    st = r.get("style_tags", {})
    ct = r.get("content_tags", {})
    tt = r.get("transfer_tags", {})
    fi = r.get("fidelity", {})
    sf = r["dim_results"].get("style_fidelity", {})

    score = fi.get("score", "?")
    score_bar = "★" * int(score) + "☆" * (5 - int(score)) if isinstance(score, int) else "?"

    print(f"\n┌─ {r['pair_id']}  {r['desc']}")
    print(f"│")
    print(f"│  STYLE IMAGE TAGS")
    print(f"│    art_style  : {st.get('art_style_l1')} / {st.get('art_style_l2')}")
    print(f"│    technique  : {st.get('technique')}")
    print(f"│    color_tone : {st.get('color_tone')}")
    print(f"│    lighting   : {st.get('lighting')}")
    print(f"│    realism    : {st.get('realism')}")
    print(f"│")
    print(f"│  CONTENT IMAGE TAGS")
    print(f"│    content_type : {ct.get('content_type')}")
    print(f"│")
    print(f"│  TRANSFER COMPLEXITY")
    print(f"│    complexity : {tt.get('transfer_complexity')}")
    print(f"│    gap        : {tt.get('realism_gap')}")
    print(f"│    domains    : {tt.get('domain_pair')}")
    print(f"│")
    print(f"│  FIDELITY SCORE  {score_bar}  ({score}/5)")
    print(f"│    style_capture        : {fi.get('style_capture')}")
    print(f"│    content_preservation : {fi.get('content_preservation')}")
    print(f"│    artifacts            : {fi.get('artifacts')}")
    print(f"│    weakest_dim          : {fi.get('weakest_dim')}")
    if sf.get("reasoning"):
        print(f"│    reasoning: {sf['reasoning'][:120]}...")
    print(f"└{'─'*60}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\nSUMMARY TABLE")
print(f"{'ID':<28} {'desc':<20} {'complexity':<14} {'score':>5}  {'style':<8} {'content':<8} {'artifacts'}")
print("─" * 100)
for r in results:
    tt = r.get("transfer_tags", {})
    fi = r.get("fidelity", {})
    print(
        f"{r['pair_id']:<28} "
        f"{r['desc']:<20} "
        f"{str(tt.get('transfer_complexity','?')):<14} "
        f"{str(fi.get('score','?')):>5}  "
        f"{str(fi.get('style_capture','?')):<8} "
        f"{str(fi.get('content_preservation','?')):<8} "
        f"{str(fi.get('artifacts','?'))}"
    )

print("\nFull results → /Users/jackeylove/style/final_results.json")
