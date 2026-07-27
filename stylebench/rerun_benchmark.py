#!/usr/bin/env python3
"""Re-run benchmark using already-generated image URLs (skip generation)."""

import json
import benchmark as bm

# URLs from the previous successful generation run
TRIPLETS = [
    {
        "id": "ST_WATERCOLOR_LANDSCAPE",
        "desc": "写实风景 → 水彩插画",
        "content": None,  # filled below
        "style": None,
        "output": None,
    },
]

# Manually set the URLs from batch_test.py output
# (all 17 images succeeded; only ST_VANGOGH style timed out)
URL_MAP = {
    "ST_WATERCOLOR_LANDSCAPE": {
        "content": None,
        "style":   None,
        "output":  None,
    },
}

# Re-submit only by re-running the full batch with already-collected URLs injected
# We'll parse the output from the previous run
import os, time, requests

for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

YATA_KEY = "yata-a7fca6599d5f6e95adfb5ace7c9d021e5094f530d5ed3deb879f03c9587b32ef"
STATUS = "https://chat-image-api.yata.art/api/image-generation/status/{}"

# Task IDs from the previous run (all images)
TASK_IDS = {
    "ST_WATERCOLOR_LANDSCAPE": {
        "content": "4ff7fb7f-139e-4f15-9627-28fcf1a2d51f",
        "style":   "d8270089-7ff8-47d2-8ea9-8f020968784e",
        "output":  "0e1c1673-9859-41b2-b189-04a394014666",
    },
    "ST_INKWASH_PORTRAIT": {
        "content": "d34d813a-ac0e-4c99-bfc5-ca6f60f70427",
        "style":   "9ae533e5-2812-4350-a182-0a08e1ee9f85",
        "output":  "f7adce0a-5c6d-4e88-aeea-57da0ab6dd6a",
    },
    "ST_CYBERPUNK_CITY": {
        "content": "c1eaa8e8-6d34-4780-aad0-7ca0a93ee92d",
        "style":   "92f09ed8-8a90-4cfe-8ba9-b2979af5c72a",
        "output":  "603a2479-b805-4a2e-aa31-1997cd47b960",
    },
    "ST_OILPAINT_ANIMAL": {
        "content": "c4b2c616-ee4e-4d5e-b5e5-7a0dc26228ed",
        "style":   "4275ee40-5d0f-41c5-a3e2-9ccc6152fde6",
        "output":  "22f72d60-c877-404d-8110-d613ab6bd43d",
    },
    "ST_FLATDESIGN_ARCHITECTURE": {
        "content": "5ce0b81a-6c65-4f59-af80-6039063bf9a5",
        "style":   "13ea6543-3687-4874-9317-7727e1bb600f",
        "output":  "77f1fb32-e86c-4a8e-9f36-01bcdf4032b5",
    },
    "ST_VANGOGH_STILL": {
        "content": "26609521-b11e-4fb1-afcf-a57d8bd495ef",
        "style":   "3d48dfec-cfe2-4da4-9f84-5150b222e1f6",  # timed out before, retry
        "output":  "48307ec2-afed-474e-9e97-dee42bbade89",
    },
}

DESCS = {
    "ST_WATERCOLOR_LANDSCAPE": "写实风景 → 水彩插画",
    "ST_INKWASH_PORTRAIT":     "写实人像 → 中国水墨",
    "ST_CYBERPUNK_CITY":       "白天城市 → 赛博朋克",
    "ST_OILPAINT_ANIMAL":      "动物照片 → 古典油画",
    "ST_FLATDESIGN_ARCHITECTURE": "建筑照片 → 扁平插画",
    "ST_VANGOGH_STILL":        "静物照片 → 梵高风格",
}


def get_url(task_id, max_wait=120):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = requests.get(
            STATUS.format(task_id),
            headers={"Authorization": f"Bearer {YATA_KEY}"},
            timeout=10
        )
        d = r.json()
        if d["status"] == "completed":
            return d["image_urls"][0]
        if d["status"] == "failed":
            return None
        time.sleep(3)
    return None


# Collect all URLs
print("Collecting image URLs from task IDs...")
urls = {}
for triplet_id, keys in TASK_IDS.items():
    urls[triplet_id] = {}
    for key, tid in keys.items():
        u = get_url(tid)
        urls[triplet_id][key] = u
        status = "OK" if u else "MISSING"
        print(f"  [{triplet_id}/{key}] {status}" + (f" {u}" if u else ""))

# Run benchmark
print("\n" + "=" * 60)
print("Running benchmark...")
print("=" * 60)

all_results = []
for triplet_id, img_urls in urls.items():
    c = img_urls.get("content")
    s = img_urls.get("style")
    o = img_urls.get("output")
    if not all([c, s, o]):
        print(f"\n[{triplet_id}] SKIP — missing URL")
        continue
    print(f"\n[{triplet_id}] {DESCS[triplet_id]}")
    result = bm.evaluate(pair_id=triplet_id, content_url=c, style_url=s, output_url=o)
    result["desc"] = DESCS[triplet_id]
    all_results.append(result)

# Summary table
print("\n" + "=" * 60)
print("BATCH SUMMARY")
print("=" * 60)
print(f"{'ID':<30} {'art_style':<22} {'technique':<18} {'realism':<16} {'complexity':<16} {'color'}")
print("-" * 115)
for r in all_results:
    st = r["style_tags"]
    tt = r["transfer_tags"]
    print(
        f"{r['pair_id']:<30} "
        f"{str(st.get('art_style_l1','?')):<22} "
        f"{str(st.get('technique','?')):<18} "
        f"{str(st.get('realism','?')):<16} "
        f"{str(tt.get('transfer_complexity','?')):<16} "
        f"{str(st.get('color_tone','?'))}"
    )

out = "/Users/jackeylove/style/batch_results.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nFull results → {out}")
