#!/usr/bin/env python3
"""
run_stylebench.py — apply the user's StyleBench-CoT (stylebench/benchmark.py) to
THIS project's local outputs. Our method is text-driven (no content image), so:
  - path-invariant dims (target style/output: art_style, technique, color_tone,
    lighting, realism) run VERBATIM on local (style_ref | output).
  - style_fidelity is adapted for the no-content-image case: fed (style, output)
    plus the text prompt subject as the content spec (content-preservation = does
    the output clearly depict the prompted subject).
  - content_type / transfer_complexity (need a content IMAGE) are SKIPPED in the
    pilot; enabled later if we generate no-style baselines as content refs.

Integration pilot goal: verify StyleBench's fidelity_score + weakest_dim reproduce
our finding (failures = weakest_dim texture/structure; palette kept) on our data.

Usage:
  python3 scripts/run_stylebench.py --pilot           # stratified 12, adapted dims
  python3 scripts/run_stylebench.py --pilot-analyze
"""
import os, io, re, json, time, base64, argparse, statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stylebench"))
import benchmark as B  # their DIMENSIONS, KAON key/url/model

ROOT = "/home/ubuntu/chengyanli/items/style_residual_distill_20260709"
STY = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty"
OUTDIR = f"{ROOT}/results/stylebench"
PILOT_F = f"{OUTDIR}/pilot.jsonl"
SUBJ = {"kp0": "a giant panda", "kp1": "a castle by the sea", "kp2": "a cat",
        "kp3": "a field of sunflowers", "kp4": "a fisherman"}
DIMS = {d["dim"]: d for d in B.DIMENSIONS}
PATH_INVARIANT = ["art_style", "technique", "color_tone", "lighting", "realism"]

# adapted style_fidelity prompt (no content image): style + output + text subject
SF_ADAPTED = """You are a style transfer quality judge evaluating how successfully the style was transferred.

You are given TWO images IN ORDER: STYLE IMAGE -> OUTPUT IMAGE. The output was asked (by a text prompt) to depict: "{subject}". There is no content image; judge content preservation against that intended subject.

Your job: score how well the OUTPUT captures the STYLE, while clearly depicting the intended subject.

Think step by step:
Step 1 - Style Capture (does the output look like the style image?):
  Check: brushwork/stroke character, color palette match, texture/medium feel, lighting mood. Rate each: fully / partially / not transferred.
Step 2 - Content Preservation (is "{subject}" clearly and correctly depicted?):
  Rate: fully / partially / lost.
Step 3 - Artifact Assessment: none / minor / major.
Step 4 - Fidelity Score (1-5):
  5=style fully transferred, subject clear, no artifacts; 4=strong style, subject clear, minor artifacts;
  3=style partial OR subject partly unclear, some artifacts; 2=style barely visible OR subject degraded; 1=style not transferred OR subject unrecognizable.
Also output: style_capture (full/partial/none), content_preservation (full/partial/lost), artifacts (none/minor/major), weakest_dim (which style sub-dim transferred least: color / texture / structure / lighting).

Output ONLY valid JSON:
{{"fidelity_score": <1-5>, "style_capture": "<label>", "content_preservation": "<label>", "artifacts": "<label>", "weakest_dim": "<label>", "confidence": <0.0-1.0>, "reasoning": "<3-4 sentences>"}}"""


def data_uri(path, px=768, q=85):
    from PIL import Image
    im = Image.open(path).convert("RGB"); im.thumbnail((px, px))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def call(prompt, img_paths, retries=4, timeout=120):
    parts = [{"type": "text", "text": prompt}]
    for p in img_paths:
        parts.append({"type": "image_url", "image_url": {"url": data_uri(p)}})
    body = {"model": B.MODEL, "messages": [{"role": "user", "content": parts}],
            "temperature": 0.1, "max_tokens": 700}
    last = ""
    for a in range(retries + 1):
        try:
            r = requests.post(B.KAON_URL, headers={"Authorization": "Bearer " + B.KAON_KEY,
                              "Content-Type": "application/json"}, json=body, timeout=timeout)
            if r.status_code in (408, 429) or r.status_code >= 500:
                last = f"http {r.status_code}"; time.sleep(3 * (a + 1)); continue
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                last = "no json"; time.sleep(2 * (a + 1)); continue
            return json.loads(m.group())
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:150]; time.sleep(2 * (a + 1))
    return {"error": last}


def stratified_12():
    recs = [json.loads(l) for l in open(f"{ROOT}/results/vlm_judge/verdicts.jsonl")]
    def pro(r):
        v = r["verdicts"].get("gemini-pro", {}).get("style_match"); return v if isinstance(v, int) else None
    for r in recs: r["pro"] = pro(r)
    recs = [r for r in recs if r["pro"] is not None]
    succ = sorted([r for r in recs if r["pro"] == 5], key=lambda r: -r["csd"])[:4]
    fail = sorted([r for r in recs if r["pro"] <= 1], key=lambda r: r["csd"])[:4]
    under = sorted([r for r in recs if r["pro"] >= 4 and r["csd"] < 0.45], key=lambda r: r["csd"])[:4]
    out = []
    for grp, tag in [(succ, "success"), (fail, "failure"), (under, "under-rated")]:
        for r in grp:
            out.append({"output": r["output"], "style_id": str(r["style_id"]),
                        "prompt_id": r["prompt_id"], "csd": r["csd"],
                        "prior_judge": r["pro"], "stratum": tag})
    return out


def eval_cell(c):
    ref = f"{STY}/{c['style_id']}.png"; gen = f"{ROOT}/results/{c['output']}"
    subj = SUBJ.get(c["prompt_id"], "the intended subject")
    res = dict(c); res["dims"] = {}
    for dn in PATH_INVARIANT:
        d = DIMS[dn]
        imgs = [ref] if d["target"] == "style" else [gen]
        res["dims"][dn] = call(d["prompt"], imgs)
    res["dims"]["style_fidelity"] = call(SF_ADAPTED.format(subject=subj), [ref, gen])
    return res


def run_pilot():
    os.makedirs(OUTDIR, exist_ok=True)
    cells = stratified_12()
    print(f"[sb-pilot] {len(cells)} cells x {len(PATH_INVARIANT)+1} adapted dims ({B.MODEL})")
    out = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        for fu in as_completed([ex.submit(eval_cell, c) for c in cells]):
            r = fu.result(); out.append(r)
            sf = r["dims"].get("style_fidelity", {})
            print(f"[sb-pilot] {r['stratum']:11s} {r['output'][:40]:41s} "
                  f"fid={sf.get('fidelity_score')} cap={sf.get('style_capture')} weak={sf.get('weakest_dim')}")
    with open(PILOT_F, "w") as f:
        for r in sorted(out, key=lambda x: (x["stratum"], x["output"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sb-pilot] -> {PILOT_F}")


def pilot_analyze():
    recs = [json.loads(l) for l in open(PILOT_F)]
    def sf(r, k): return r["dims"].get("style_fidelity", {}).get(k)
    print(f"\n===== StyleBench integration pilot (n={len(recs)}) =====")
    # 1) does fidelity_score track our prior judge / separate strata?
    print("\n[1] StyleBench fidelity_score vs stratum & prior judge:")
    for tag in ["success", "under-rated", "failure"]:
        rs = [r for r in recs if r["stratum"] == tag]
        fids = [sf(r, "fidelity_score") for r in rs if isinstance(sf(r, "fidelity_score"), (int, float))]
        pj = [r["prior_judge"] for r in rs]
        print(f"  {tag:11s} n={len(rs)}  StyleBench fid mean={st.mean(fids):.2f} {sorted(fids)}  "
              f"(prior style-judge {sorted(pj)})")
    # 2) weakest_dim on failures (the money reproduction)
    print("\n[2] weakest_dim distribution (does 'texture/structure' dominate failures?):")
    from collections import Counter
    for tag in ["success", "under-rated", "failure"]:
        wk = [sf(r, "weakest_dim") for r in recs if r["stratum"] == tag]
        print(f"  {tag:11s}: {dict(Counter(wk))}")
    # 3) style_capture on failures — is it 'partial/none' with palette kept?
    print("\n[3] failure cells detail (style_capture / weakest / color_tone label):")
    for r in [x for x in recs if x["stratum"] == "failure"]:
        print(f"  {r['output'][:38]:39s} cap={sf(r,'style_capture')} weak={sf(r,'weakest_dim')} "
              f"tech={r['dims'].get('technique',{}).get('technique')} "
              f"art={r['dims'].get('art_style',{}).get('art_style_l1')}")
    # 4) any dim errors?
    errs = sum(1 for r in recs for d in r["dims"].values() if "error" in d)
    print(f"\n[4] dim errors: {errs}/{len(recs)*(len(PATH_INVARIANT)+1)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-analyze", action="store_true")
    a = ap.parse_args()
    if a.pilot: run_pilot()
    elif a.pilot_analyze: pilot_analyze()
    else: print("use --pilot | --pilot-analyze")
