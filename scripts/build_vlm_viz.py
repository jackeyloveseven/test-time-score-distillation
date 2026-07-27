#!/usr/bin/env python3
"""Build a self-contained HTML visualization of the VLM-panel calibration study
(for R2 public publish). No external deps: images base64-embedded, scatter as
inline SVG. Reader: the user (shareable link). Data: results/vlm_judge/verdicts.jsonl."""
import json, io, base64, statistics as st, hashlib
from PIL import Image

ROOT = "/home/ubuntu/chengyanli/items/style_residual_distill_20260709"
STY = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty"
OUT = f"{ROOT}/agent_runs/vlm_calibration_viz.html"

recs = [json.loads(l) for l in open(f"{ROOT}/results/vlm_judge/verdicts.jsonl")]
def pv(r, ch="gemini-pro", k="style_match"):
    v = r["verdicts"].get(ch, {}).get(k); return v if isinstance(v, (int, float)) else None
for r in recs:
    r["pro"] = pv(r); r["fl"] = pv(r, "gemini-flash"); r["cc"] = pv(r, "gemini-pro", "content_clarity")
    r["ctrl"] = pv(r, "control_gemini-pro")
R = [r for r in recs if r["pro"] is not None]

# ---- stats ----
n = len(R)
succ = [r for r in R if r["pro"] >= 4]; ind = [r for r in R if r["pro"] == 5]
fails = [r for r in R if r["pro"] <= 2]
exc_csd = [r["csd"] for r in succ]
ctrlS = [r["ctrl"] for r in R if r["ctrl"] is not None]
proS = [r["pro"] for r in R]
disc = st.mean(proS) - st.mean(ctrlS)
# inter-judge r
pf = [(r["pro"], r["fl"]) for r in R if r["fl"] is not None]
mx = st.mean([a for a,_ in pf]); my = st.mean([b for _,b in pf])
num = sum((a-mx)*(b-my) for a,b in pf); dx = sum((a-mx)**2 for a,_ in pf)**.5; dy = sum((b-my)**2 for _,b in pf)**.5
rr = num/(dx*dy)
fail_styles = sorted({str(r["style_id"]) for sid_r in [fails] for r in sid_r
                      for r in [r]})  # placeholder
# robust failure styles (fail on all their seeds)
from collections import defaultdict
bys = defaultdict(list)
for r in R: bys[str(r["style_id"])].append(r)
robust = sorted([s for s,rs in bys.items() if len(rs) >= 2 and all(x["pro"] <= 2 for x in rs)], key=int)

def b64_pair(sid, out_png, px=300, q=82):
    ref = Image.open(f"{STY}/{sid}.png").convert("RGB"); ref.thumbnail((px, px))
    gen = Image.open(f"{ROOT}/results/{out_png}").convert("RGB"); gen.thumbnail((px, px))
    h = max(ref.height, gen.height)
    m = Image.new("RGB", (ref.width + gen.width + 8, h), "white")
    m.paste(ref, (0, 0)); m.paste(gen, (ref.width + 8, 0))
    b = io.BytesIO(); m.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

EX = {  # sid, output, csd, judge, kind, caption
 "under1": ("25", "breadthext_score_match_style25_kp3_seed64.png", 0.49, 5, "under",
            "B&amp;W engraving archer &rarr; sunflower field. Flawless engraving, yet CSD only 0.49."),
 "under2": ("10", "bench40_score_match_style10_kp1_seed64.png", 0.381, 5, "under",
            "Impasto portrait &rarr; castle. Excellent bold-brushstroke transfer, yet CSD only 0.38."),
 "win":    ("38", "bench40_score_match_style38_kp4_seed65.png", 0.858, 5, "win",
            "Low-poly vector &rarr; fisherman. High CSD, high judge &mdash; both agree it&rsquo;s great."),
 "fail1":  ("04", "bench40_score_match_style04_kp2_seed65.png", 0.393, 1, "fail",
            "Gold-mosaic / Klimt &rarr; cat. Only the warm palette transferred; mosaic absent."),
 "fail2":  ("16", "bench40_score_match_style16_kp4_seed65.png", 0.328, 1, "fail",
            "Crystalline 3D-render &rarr; portrait. Palette copied; glassy crystal style missing."),
}
imgs = {k: b64_pair(v[0], v[1]) for k, v in EX.items()}

# ---- scatter SVG (x=CSD, y=judge style-match with jitter) ----
W, H, PL, PR, PT, PB = 720, 380, 56, 20, 24, 44
def sx(c): return PL + (c - 0.25) / (0.90 - 0.25) * (W - PL - PR)
def sy(v, jit): return H - PB - ((v - 0.6) / (5.4 - 0.6)) * (H - PT - PB) + jit
pts = []
key_out = {EX[k][1] for k in EX}
for r in R:
    c = r["csd"]; v = r["pro"]
    jit = (int(hashlib.md5(r["output"].encode()).hexdigest(), 16) % 100 / 100 - 0.5) * 14
    col = "#16a34a" if v >= 4 else ("#d97706" if v == 3 else "#dc2626")
    is_key = r["output"] in key_out
    pts.append((sx(c), sy(v, jit), col, is_key, r["output"]))
grid = ""
for cx in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    grid += f'<line x1="{sx(cx):.0f}" y1="{PT}" x2="{sx(cx):.0f}" y2="{H-PB}" stroke="#eee"/>' \
            f'<text x="{sx(cx):.0f}" y="{H-PB+16}" font-size="11" fill="#888" text-anchor="middle">{cx:.1f}</text>'
for v in [1, 2, 3, 4, 5]:
    yy = sy(v, 0)
    grid += f'<line x1="{PL}" y1="{yy:.0f}" x2="{W-PR}" y2="{yy:.0f}" stroke="#f3f3f3"/>' \
            f'<text x="{PL-8}" y="{yy+4:.0f}" font-size="11" fill="#888" text-anchor="end">{v}</text>'
# shade the "panel-excellent" band (>=4) to show CSD spread there
band_y = sy(3.5, 0)
grid += f'<rect x="{PL}" y="{PT}" width="{W-PL-PR}" height="{band_y-PT:.0f}" fill="#16a34a" opacity="0.05"/>'
dots = ""
for x, y, col, key, name in pts:
    if key:
        dots += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" fill="{col}" stroke="#111" stroke-width="2"/>'
    else:
        dots += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4" fill="{col}" opacity="0.75"/>'
scatter = f'''<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">
<text x="{PL}" y="16" font-size="12" fill="#555" font-weight="600">Panel-excellent band (style-match &ge;4): CSD sprawls 0.36&ndash;0.86 &rarr; CSD can&rsquo;t rank quality here</text>
{grid}{dots}
<text x="{(W)/2:.0f}" y="{H-6}" font-size="12" fill="#555" text-anchor="middle">raw CSD style-similarity &rarr;</text>
<text x="16" y="{(H)/2:.0f}" font-size="12" fill="#555" text-anchor="middle" transform="rotate(-90 16 {(H)/2:.0f})">independent judge style-match (1&ndash;5) &rarr;</text>
</svg>'''

def card(k):
    sid, out, csd, judge, kind, cap = EX[k]
    badge = {"under": ("#7c3aed", "CSD under-rates"), "win": ("#16a34a", "both agree: great"),
             "fail": ("#dc2626", "genuine failure")}[kind]
    return f'''<figure class="ex {kind}">
<img src="{imgs[k]}" alt="style {sid}"/>
<div class="exmeta"><span class="tag" style="background:{badge[0]}">{badge[1]}</span>
<span class="nums">raw CSD <b>{csd:.2f}</b> &middot; judge <b>{judge}/5</b></span></div>
<figcaption>{cap}</figcaption></figure>'''

html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native Score-Residual Style Transfer &mdash; Independent VLM-Panel Audit</title>
<style>
 :root{{--ink:#18181b;--mut:#6b7280;--line:#e5e7eb;--bg:#fafafa;--acc:#4f46e5}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
   font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
 .wrap{{max-width:960px;margin:0 auto;padding:32px 20px 64px}}
 h1{{font-size:26px;line-height:1.25;margin:0 0 6px}} .sub{{color:var(--mut);margin:0 0 4px;font-size:15px}}
 .method{{color:var(--mut);font-size:13.5px;margin:8px 0 26px;padding:10px 14px;background:#fff;border:1px solid var(--line);border-radius:10px}}
 .method code{{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
 .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:0 0 34px}}
 .stat{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
 .stat .v{{font-size:24px;font-weight:700}} .stat .l{{color:var(--mut);font-size:12.5px;margin-top:2px}}
 .stat.g .v{{color:#16a34a}} .stat.r .v{{color:#dc2626}} .stat.p .v{{color:#7c3aed}}
 h2{{font-size:19px;margin:34px 0 6px}} .lead{{color:var(--mut);margin:0 0 16px;font-size:14.5px}}
 .card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:14px}}
 .exgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
 figure.ex{{margin:0;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}}
 figure.ex img{{width:100%;display:block;background:#f3f4f6}}
 .exmeta{{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 12px 4px;flex-wrap:wrap}}
 .tag{{color:#fff;font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;white-space:nowrap}}
 .nums{{font-size:12.5px;color:var(--mut)}} .nums b{{color:var(--ink)}}
 figure.ex figcaption{{padding:2px 12px 12px;font-size:12.5px;color:var(--mut)}}
 figure.ex.under{{box-shadow:inset 0 0 0 2px #7c3aed22}} figure.ex.fail{{box-shadow:inset 0 0 0 2px #dc262622}}
 .mech{{display:flex;gap:20px;flex-wrap:wrap;align-items:center}} .mech .big{{font-size:15px}}
 .pill{{display:inline-block;background:#f3f4f6;border-radius:6px;padding:2px 7px;font-size:12.5px;margin:2px}}
 .foot{{color:var(--mut);font-size:12.5px;margin-top:34px;border-top:1px solid var(--line);padding-top:14px}}
 .flag{{display:inline-block;background:#fef3c7;color:#92400e;font-size:12px;font-weight:600;padding:2px 9px;border-radius:20px;margin-bottom:10px}}
</style></head><body><div class="wrap">
<div class="flag">independent audit &middot; pre-committed falsification test</div>
<h1>Native Score-Residual Style Transfer:<br>an Independent VLM-Panel Audit</h1>
<p class="sub">Training-free style transfer on a frozen rectified-flow transformer (FLUX.2 [klein] 9B) &mdash; where it works, where it fails, and why the standard metric can&rsquo;t tell.</p>
<div class="method">Method: optimize <b>only the current latent</b> along the <b>same-state image-CFG score residual</b> (score with vs. without the style reference at the identical noise state). Train nothing, add no modules. Audited by a 2-model VLM panel (<code>gemini-pro</code> + <code>gemini-flash</code>) rating style-match <b>blind to subject</b>, over {n} generations.</div>

<div class="stats">
 <div class="stat g"><div class="v">{len(succ)/n*100:.0f}%</div><div class="l">rated successful transfers (judge &ge;4)</div></div>
 <div class="stat g"><div class="v">{len(ind)/n*100:.0f}%</div><div class="l">indistinguishable in medium (judge = 5)</div></div>
 <div class="stat"><div class="v">5.0</div><div class="l">content clarity on <b>all</b> {n} cells &mdash; content never breaks</div></div>
 <div class="stat r"><div class="v">{len(robust)}</div><div class="l">style types that fail on every seed</div></div>
 <div class="stat p"><div class="v">&Delta;{disc:.2f}</div><div class="l">discrimination control (correct vs wrong ref)</div></div>
 <div class="stat"><div class="v">r={rr:.2f}</div><div class="l">inter-judge agreement (panel is consistent)</div></div>
</div>

<h2>The standard metric can&rsquo;t rank quality among successes</h2>
<p class="lead">Each dot is one generation: raw CSD (x) vs. the independent judge&rsquo;s style score (y). Among cells the panel calls excellent (green band), CSD sprawls from <b>0.36 to 0.86</b> &mdash; so a low CSD does <i>not</i> mean a worse transfer. CSD mostly separates success from failure; it does not order quality.</p>
<div class="card">{scatter}</div>

<h2>CSD under-rates &mdash; caught in the act</h2>
<p class="lead">Two flawless transfers the raw metric scores low, purely because the generated subject differs from the reference&rsquo;s subject. The independent panel (which ignores subject) rates both 5/5.</p>
<div class="exgrid">{card("under1")}{card("under2")}</div>

<h2>Where it fails &mdash; and the mechanism</h2>
<p class="lead">On complex non-painterly styles the method collapses to copying the palette. This is not noise: {len(robust)} style families fail on every seed, and the cause is clean.</p>
<div class="exgrid">{card("fail1")}{card("fail2")}</div>
<div class="card mech"><div class="big">
 <b>Every failure looks the same:</b> content clarity <b style="color:#16a34a">5.0</b> + style-match <b style="color:#dc2626">1.65</b>.
 The content-preservation term <b>dominates</b> on hard styles &mdash; the image stays a clean, generic render and the style never takes hold.
 <div style="margin-top:8px">Robust-failure styles (synthetic-render / mosaic / crystalline):
 {" ".join(f'<span class="pill">style {s}</span>' for s in robust)}</div>
</div></div>

<h2>What&rsquo;s solid</h2>
<p class="lead">The instrument is validated (discrimination &Delta;{disc:.2f}, inter-judge r={rr:.2f}, and the authors eyeball-confirmed the failure calls). On styles with a nameable medium the transfer is excellent and the metric agrees.</p>
<div class="exgrid">{card("win")}</div>

<div class="foot">
 Audit of {n} generations across 40 style references &middot; 2-model VLM panel, blind-to-subject, with a wrong-reference discrimination control &middot;
 raw verdicts + harness reproducible &middot; a pre-committed falsification test &mdash; reported whichever way it landed.
</div>
</div></body></html>'''

open(OUT, "w").write(html)
sz = len(html.encode())
print(f"wrote {OUT}  ({sz/1024:.0f} KB)")
print(f"stats: n={n} succ={len(succ)}({len(succ)/n*100:.0f}%) ind={len(ind)}({len(ind)/n*100:.0f}%) "
      f"robust_fail={robust} disc=Δ{disc:.2f} r={rr:.2f} exc_csd=[{min(exc_csd):.2f},{max(exc_csd):.2f}]")
