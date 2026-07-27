#!/usr/bin/env python3
"""Batch test: generate multiple style-transfer triplets and run benchmark on each."""

import os, time, json, requests
import benchmark as bm

for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

YATA_KEY = "yata-a7fca6599d5f6e95adfb5ace7c9d021e5094f530d5ed3deb879f03c9587b32ef"
SUBMIT = "https://chat-image-api.yata.art/api/image-generation/submit"
STATUS = "https://chat-image-api.yata.art/api/image-generation/status/{}"
H = {"Authorization": f"Bearer {YATA_KEY}", "Content-Type": "application/json"}

# 6 triplets covering diverse style × content × complexity
TRIPLETS = [
    {
        "id": "ST_WATERCOLOR_LANDSCAPE",
        "desc": "写实风景 → 水彩插画",
        "content": "photorealistic mountain landscape, green valley, blue sky with clouds, golden hour, DSLR photo, sharp detail",
        "style":   "loose watercolor landscape painting, wet-on-wet technique, soft bleeds, paper texture visible, impressionistic brushwork, warm sunset palette",
        "output":  "mountain landscape rendered in loose watercolor style, wet-on-wet bleeds, paper texture, soft impressionistic washes, warm sunset colors",
        "ratio": "3:2",
    },
    {
        "id": "ST_INKWASH_PORTRAIT",
        "desc": "写实人像 → 中国水墨",
        "content": "photorealistic portrait of an elderly man, strong facial features, dramatic side lighting, black and white photography style",
        "style":   "traditional Chinese ink wash painting, sumi-e style, calligraphic brushstrokes, strong contrast, minimal composition, white background, black ink gradients",
        "output":  "portrait of an elderly man in Chinese ink wash style, sumi-e brushstrokes, high contrast black ink, white space, calligraphic line quality",
        "ratio": "2:3",
    },
    {
        "id": "ST_CYBERPUNK_CITY",
        "desc": "白天城市 → 赛博朋克",
        "content": "daytime photograph of a modern city street, glass skyscrapers, pedestrians, clear blue sky, realistic urban scene",
        "style":   "cyberpunk neon city at night, vibrant purple and teal neon lights, rain-slicked streets, holographic advertisements, blade runner aesthetic, dark dystopian atmosphere",
        "output":  "city street transformed into cyberpunk aesthetic, neon purple and teal lighting, rain-slicked reflections, holographic signs, dark night atmosphere",
        "ratio": "3:2",
    },
    {
        "id": "ST_OILPAINT_ANIMAL",
        "desc": "动物照片 → 古典油画",
        "content": "photorealistic photograph of a majestic lion, close-up portrait, golden savanna background, natural lighting",
        "style":   "17th century Dutch Golden Age oil painting, dramatic chiaroscuro, rich impasto brushwork, dark background, warm golden tones, baroque lighting",
        "output":  "lion portrait in 17th century Dutch oil painting style, chiaroscuro lighting, thick impasto strokes, dark baroque background, warm golden palette",
        "ratio": "1:1",
    },
    {
        "id": "ST_FLATDESIGN_ARCHITECTURE",
        "desc": "建筑照片 → 扁平插画",
        "content": "photorealistic photograph of a European cathedral, gothic architecture, stone facade, overcast sky",
        "style":   "modern flat design illustration, geometric shapes, limited color palette, no shadows, clean vector aesthetic, minimal graphic poster style",
        "output":  "European cathedral rendered in flat design style, geometric simplification, limited palette, no gradients, clean vector shapes",
        "ratio": "2:3",
    },
    {
        "id": "ST_VANGOGH_STILL",
        "desc": "静物照片 → 梵高风格",
        "content": "photorealistic still life photograph, ceramic vase with sunflowers, wooden table, soft natural window light",
        "style":   "Van Gogh post-impressionist painting, thick expressive swirling brushstrokes, vivid impasto texture, bold yellows and blues, emotional turbulent strokes",
        "output":  "sunflower still life in Van Gogh style, thick swirling impasto strokes, vivid yellow and blue palette, turbulent expressive brushwork",
        "ratio": "2:3",
    },
]


def submit_task(prompt, ratio):
    r = requests.post(SUBMIT, headers=H, json={
        "prompt": prompt,
        "model": "gemini-3-pro-image-preview",
        "temperature": 0.7,
        "aspect_ratio": ratio,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["task_id"]


def wait_for(task_id, max_wait=180):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = requests.get(STATUS.format(task_id), headers={"Authorization": f"Bearer {YATA_KEY}"}, timeout=10)
        d = r.json()
        if d["status"] == "completed":
            return d["image_urls"][0]
        if d["status"] == "failed":
            raise RuntimeError(f"Task failed: {d}")
        time.sleep(4)
    raise TimeoutError(f"Timed out waiting for {task_id}")


# ── Step 1: submit all generation tasks ──────────────────────────────────────
print("=" * 60)
print(f"Submitting {len(TRIPLETS) * 3} image generation tasks...")
print("=" * 60)

pending = []  # (triplet_id, key, task_id)
for t in TRIPLETS:
    for key in ("content", "style", "output"):
        tid = submit_task(t[key], t["ratio"])
        pending.append((t["id"], key, tid))
        print(f"  [{t['id']}] {key}: {tid}")
    time.sleep(0.5)  # mild rate limiting

# ── Step 2: collect all image URLs ───────────────────────────────────────────
print(f"\nWaiting for {len(pending)} images...")
url_map = {}  # (triplet_id, key) -> url
for triplet_id, key, tid in pending:
    label = f"{triplet_id}/{key}"
    print(f"  [{label}] ...", end=" ", flush=True)
    try:
        url = wait_for(tid)
        url_map[(triplet_id, key)] = url
        print(f"OK")
    except Exception as e:
        url_map[(triplet_id, key)] = None
        print(f"FAILED: {e}")

# ── Step 3: run benchmark on each triplet ────────────────────────────────────
print("\n" + "=" * 60)
print("Running 7-dimension benchmark on each triplet...")
print("=" * 60)

all_results = []
for t in TRIPLETS:
    content_url = url_map.get((t["id"], "content"))
    style_url   = url_map.get((t["id"], "style"))
    output_url  = url_map.get((t["id"], "output"))

    if not all([content_url, style_url, output_url]):
        print(f"\n[{t['id']}] SKIP — missing image URL")
        continue

    print(f"\n[{t['id']}] {t['desc']}")
    result = bm.evaluate(
        pair_id=t["id"],
        content_url=content_url,
        style_url=style_url,
        output_url=output_url,
    )
    result["desc"] = t["desc"]
    all_results.append(result)

# ── Step 4: summary report ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BATCH SUMMARY")
print("=" * 60)
print(f"{'ID':<32} {'art_style':<22} {'technique':<18} {'realism':<16} {'complexity':<16} {'color'}")
print("-" * 120)
for r in all_results:
    st = r["style_tags"]
    tt = r["transfer_tags"]
    print(
        f"{r['pair_id']:<32} "
        f"{str(st.get('art_style_l1','?')):<22} "
        f"{str(st.get('technique','?')):<18} "
        f"{str(st.get('realism','?')):<16} "
        f"{str(tt.get('transfer_complexity','?')):<16} "
        f"{str(st.get('color_tone','?'))}"
    )

# save full results
out = "/Users/jackeylove/style/batch_results.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nFull results saved → {out}")
