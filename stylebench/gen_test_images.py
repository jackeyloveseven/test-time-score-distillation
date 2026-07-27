#!/usr/bin/env python3
"""Generate a style transfer test triplet using Yata API:
  content: realistic portrait photo style
  style:   Midjourney Niji anime illustration
  output:  portrait rendered in Niji anime style
"""

import os, time, requests, json

for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

YATA_KEY = "yata-a7fca6599d5f6e95adfb5ace7c9d021e5094f530d5ed3deb879f03c9587b32ef"
SUBMIT = "https://chat-image-api.yata.art/api/image-generation/submit"
STATUS = "https://chat-image-api.yata.art/api/image-generation/status/{}"
HEADERS = {"Authorization": f"Bearer {YATA_KEY}", "Content-Type": "application/json"}

IMAGES = [
    {
        "key": "content",
        "prompt": "photorealistic portrait of a young woman, natural soft lighting, outdoor golden hour, Canon 85mm lens, shallow depth of field, realistic skin texture, no makeup, casual clothing",
        "aspect_ratio": "2:3",
    },
    {
        "key": "style",
        "prompt": "Midjourney niji 6 anime illustration, beautiful anime girl, soft cel shading, clean linework, pastel color palette, dreamy atmosphere, flowing hair, detailed eyes, shoujo manga style, white background",
        "aspect_ratio": "2:3",
    },
    {
        "key": "output",
        "prompt": "Midjourney niji 6 anime style portrait of a young woman, cel shading, soft pastel colors, clean linework, anime face proportions with large detailed eyes, flowing hair, shoujo manga aesthetic, dreamy soft background",
        "aspect_ratio": "2:3",
    },
]


def submit(prompt, aspect_ratio):
    r = requests.post(SUBMIT, headers=HEADERS, json={
        "prompt": prompt,
        "model": "gemini-3-pro-image-preview",
        "temperature": 0.7,
        "aspect_ratio": aspect_ratio,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["task_id"]


def wait(task_id, max_wait=180):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = requests.get(STATUS.format(task_id), headers={"Authorization": f"Bearer {YATA_KEY}"}, timeout=10)
        data = r.json()
        if data["status"] == "completed":
            return data["image_urls"][0]
        if data["status"] == "failed":
            raise RuntimeError(f"failed: {data}")
        time.sleep(4)
    raise TimeoutError("timed out")


results = {}
print("Submitting 3 generation tasks...")
tasks = {}
for img in IMAGES:
    tid = submit(img["prompt"], img["aspect_ratio"])
    tasks[img["key"]] = tid
    print(f"  {img['key']}: task_id={tid}")

print("\nWaiting for results...")
for key, tid in tasks.items():
    print(f"  [{key}] polling...", end=" ", flush=True)
    url = wait(tid)
    results[key] = url
    print(url)

print("\n=== Generated URLs ===")
print(json.dumps(results, indent=2))

out = "/Users/jackeylove/style/test_triplet.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")
