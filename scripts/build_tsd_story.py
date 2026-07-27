#!/usr/bin/env python3
"""Self-contained 中文 'visual paper' HTML of the TSD story, for R2.
Readable: per-section takeaways, trimmed non-redundant figures, aspect-ratio preserved."""
import base64, io, os
from PIL import Image

SC = "/tmp/claude-1000/-home-ubuntu-chengyanli/0da1ecad-87ce-4e83-abe2-cbcdbb020c1e/scratchpad"
OUT = "/home/ubuntu/chengyanli/items/style_residual_distill_20260709/agent_runs/tsd_story.html"

def img(name, maxw=1000, q=82):
    p = f"{SC}/{name}"
    if not os.path.exists(p): return ""
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

# only the 5 non-redundant result figures (cut worst_04/16, cabatch_alpha, renoise2, ceil04)
IM = {k: img(k + ".png") for k in ["caprobe","cabatch_gen","cfggate","b2_cmp","dmsched_cmp"]}
def figure(key, cap):
    if not IM.get(key): return f"<div class='miss'>[{key} 缺失]</div>"
    return f"<figure><img src='{IM[key]}'/><figcaption>{cap}</figcaption></figure>"

mech_svg = """<svg viewBox="0 0 560 250" width="100%" style="max-width:520px;height:auto">
<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#111"/></marker></defs>
<line x1="40" y1="215" x2="520" y2="215" stroke="#ccc"/><text x="280" y="238" font-size="12" fill="#888" text-anchor="middle">迭代次数 →</text>
<text x="18" y="125" font-size="12" fill="#888" transform="rotate(-90 18 125)">风格搬运量 →</text>
<circle cx="430" cy="148" r="5" fill="#94a3b8"/><text x="440" y="143" font-size="11.5" fill="#64748b">弱不动点 (v₊=v₋)</text>
<path d="M40,205 C160,148 300,150 430,148" fill="none" stroke="#dc2626" stroke-width="3" marker-end="url(#ar)"/>
<text x="150" y="184" font-size="12.5" fill="#dc2626" font-weight="600">score_match:插值→饱和</text>
<path d="M40,205 C170,148 300,108 500,40" fill="none" stroke="#16a34a" stroke-width="3" marker-end="url(#ar)"/>
<text x="285" y="68" font-size="12.5" fill="#16a34a" font-weight="600">TSD:外推走过它→持续累积</text></svg>"""

framework_svg = """<svg viewBox="0 0 900 140" width="100%" style="max-width:880px;height:auto">
<defs><marker id="af" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#475569"/></marker></defs>
<rect x="8" y="18" width="118" height="30" rx="8" fill="#fff7ed" stroke="#f59e0b"/><text x="67" y="37" font-size="12" text-anchor="middle" fill="#92400e">风格参考图</text>
<rect x="8" y="62" width="118" height="30" rx="8" fill="#eff6ff" stroke="#3b82f6"/><text x="67" y="81" font-size="12" text-anchor="middle" fill="#1e40af">文字 prompt</text>
<line x1="126" y1="43" x2="158" y2="52" stroke="#475569" stroke-width="1.5" marker-end="url(#af)"/><line x1="126" y1="77" x2="158" y2="68" stroke="#475569" stroke-width="1.5" marker-end="url(#af)"/>
<rect x="158" y="36" width="120" height="46" rx="9" fill="#f8fafc" stroke="#334155"/><text x="218" y="56" font-size="12" text-anchor="middle" fill="#0f172a" font-weight="700">冻结整流流</text><text x="218" y="73" font-size="12" text-anchor="middle" fill="#0f172a" font-weight="700">DiT</text>
<line x1="278" y1="48" x2="316" y2="34" stroke="#475569" stroke-width="1.5" marker-end="url(#af)"/><line x1="278" y1="72" x2="316" y2="88" stroke="#475569" stroke-width="1.5" marker-end="url(#af)"/>
<rect x="316" y="18" width="186" height="30" rx="8" fill="#f0fdf4" stroke="#16a34a"/><text x="409" y="37" font-size="12" text-anchor="middle" fill="#166534">v₊ = DiT(有风格条件)</text>
<rect x="316" y="76" width="186" height="30" rx="8" fill="#fef2f2" stroke="#dc2626"/><text x="409" y="95" font-size="12" text-anchor="middle" fill="#991b1b">v₋ = DiT(无风格条件)</text>
<line x1="502" y1="33" x2="540" y2="54" stroke="#475569" stroke-width="1.5" marker-end="url(#af)"/><line x1="502" y1="91" x2="540" y2="70" stroke="#475569" stroke-width="1.5" marker-end="url(#af)"/>
<rect x="540" y="36" width="176" height="46" rx="9" fill="#eef2ff" stroke="#6366f1"/><text x="628" y="56" font-size="12" text-anchor="middle" fill="#3730a3">g=(α−1)(v₊−v₋)</text><text x="628" y="73" font-size="11.5" text-anchor="middle" fill="#6366f1">rms 归一(矛)</text>
<line x1="716" y1="59" x2="752" y2="59" stroke="#475569" stroke-width="1.5" marker-end="url(#af)"/>
<rect x="752" y="36" width="140" height="46" rx="9" fill="#f0fdf4" stroke="#16a34a"/><text x="822" y="56" font-size="12" text-anchor="middle" fill="#166534">latent 外推步</text><text x="822" y="73" font-size="11.5" text-anchor="middle" fill="#16a34a">x₀ += s·ĝ</text>
<text x="409" y="128" font-size="11.5" text-anchor="middle" fill="#94a3b8">同一噪声态、只差"有无风格条件" = same-state CFG-augmentation</text></svg>"""

loop_svg = """<svg viewBox="0 0 900 145" width="100%" style="max-width:880px;height:auto">
<defs><marker id="a2" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#475569"/></marker></defs>
<rect x="10" y="45" width="200" height="56" rx="10" fill="#eef2ff" stroke="#6366f1"/><text x="110" y="69" font-size="13" text-anchor="middle" fill="#3730a3" font-weight="700">冻结去噪一步</text><text x="110" y="88" font-size="12" text-anchor="middle" fill="#6366f1">= 盾(免费)</text>
<line x1="210" y1="73" x2="255" y2="73" stroke="#475569" stroke-width="2" marker-end="url(#a2)"/>
<rect x="255" y="45" width="230" height="56" rx="10" fill="#f0fdf4" stroke="#16a34a"/><text x="370" y="67" font-size="13" text-anchor="middle" fill="#166534" font-weight="700">latent 上做 CFG-aug 外推</text><text x="370" y="87" font-size="11.5" text-anchor="middle" fill="#16a34a">g=(α−1)(v₊−v₋),rms 归一 = 矛</text>
<line x1="485" y1="73" x2="530" y2="73" stroke="#475569" stroke-width="2" marker-end="url(#a2)"/>
<rect x="530" y="45" width="150" height="56" rx="10" fill="#fff" stroke="#cbd5e1"/><text x="605" y="77" font-size="12" text-anchor="middle" fill="#475569">内 5 次 × 外 25 步</text>
<path d="M680,73 C760,73 760,20 400,20 C120,20 120,45 110,45" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#a2)"/>
<text x="400" y="14" font-size="11" fill="#94a3b8" text-anchor="middle">循环 — 不训练任何权重、不用 fake 网络</text></svg>"""

recast_svg = """<svg viewBox="0 0 900 215" width="100%" style="max-width:880px;height:auto">
<defs><marker id="ab" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#64748b"/></marker></defs>
<line x1="450" y1="8" x2="450" y2="205" stroke="#e5e7eb" stroke-dasharray="4 4"/>
<text x="215" y="22" font-size="13.5" text-anchor="middle" fill="#0f172a" font-weight="800">DMD · 训练生成器</text>
<rect x="20" y="38" width="170" height="38" rx="8" fill="#f0fdf4" stroke="#16a34a"/><text x="105" y="62" font-size="11.5" text-anchor="middle" fill="#166534">CA 矛 (α−1)(v₊−v₋)</text>
<rect x="230" y="38" width="185" height="38" rx="8" fill="#eff6ff" stroke="#3b82f6"/><text x="322" y="55" font-size="11" text-anchor="middle" fill="#1e40af">DM 盾 = 训练出的</text><text x="322" y="69" font-size="11" text-anchor="middle" fill="#1e40af">fake 网络</text>
<line x1="105" y1="76" x2="180" y2="116" stroke="#64748b" stroke-width="1.5" marker-end="url(#ab)"/><line x1="322" y1="76" x2="250" y2="116" stroke="#64748b" stroke-width="1.5" marker-end="url(#ab)"/>
<rect x="110" y="118" width="220" height="38" rx="8" fill="#fff" stroke="#334155"/><text x="220" y="142" font-size="12" text-anchor="middle" fill="#0f172a">生成器 θ(参数会漂移)</text>
<text x="220" y="186" font-size="12" text-anchor="middle" fill="#b91c1c">盾必需 —— 挡训练里的参数漂移</text>
<text x="685" y="22" font-size="13.5" text-anchor="middle" fill="#0f172a" font-weight="800">TSD · 推理 · 单样本</text>
<rect x="490" y="38" width="170" height="38" rx="8" fill="#f0fdf4" stroke="#16a34a"/><text x="575" y="62" font-size="11.5" text-anchor="middle" fill="#166534">CA 矛(保留)</text>
<rect x="700" y="38" width="190" height="38" rx="8" fill="#eff6ff" stroke="#3b82f6"/><text x="795" y="55" font-size="11" text-anchor="middle" fill="#1e40af">盾 = 冻结反向过程</text><text x="795" y="69" font-size="11" text-anchor="middle" fill="#1e40af">(免费)</text>
<line x1="575" y1="76" x2="650" y2="116" stroke="#64748b" stroke-width="1.5" marker-end="url(#ab)"/><line x1="795" y1="76" x2="720" y2="116" stroke="#64748b" stroke-width="1.5" marker-end="url(#ab)"/>
<rect x="580" y="118" width="230" height="38" rx="8" fill="#fff" stroke="#334155"/><text x="695" y="142" font-size="12" text-anchor="middle" fill="#0f172a">单个 latent(有界·无漂移)</text>
<text x="695" y="186" font-size="12" text-anchor="middle" fill="#b91c1c" font-weight="600">显式 DM 盾 → 多余且有害</text></svg>"""

_neglabels = ["score_match","低-τ 重加噪","fake-score DM","注意力/K-V","摊销 student","carrier"]
_boxes = "".join(f'<rect x="14" y="{20+i*40}" width="150" height="30" rx="7" fill="#fef2f2" stroke="#dc2626"/><text x="26" y="{40+i*40}" font-size="11.5" fill="#991b1b">✗ {l}</text><line x1="164" y1="{35+i*40}" x2="352" y2="140" stroke="#cbd5e1" stroke-width="1.4" marker-end="url(#ac)"/>' for i,l in enumerate(_neglabels))
converge_svg = f"""<svg viewBox="0 0 620 270" width="100%" style="max-width:560px;height:auto">
<defs><marker id="ac" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#cbd5e1"/></marker></defs>
{_boxes}
<rect x="356" y="115" width="230" height="50" rx="11" fill="#f0fdf4" stroke="#16a34a" stroke-width="2"/><text x="471" y="139" font-size="14" text-anchor="middle" fill="#166534" font-weight="800">CA-alone (TSD) ✓</text><text x="471" y="156" font-size="11" text-anchor="middle" fill="#16a34a">唯一活下来的</text>
<text x="300" y="258" font-size="12" text-anchor="middle" fill="#64748b">六个"显而易见"的加法 → 全部收敛回 CA-alone</text></svg>"""

NEG = [("score-matching","插值到弱不动点 v₊=v₋ → 纹理没出就饱和","2/partial → 4–5/full"),
 ("低-τ 重加噪","低 σ 图已定 → 方向指向局部内容,非全局风格","比 same-state 更差"),
 ("fake-score DM","往自然主义流形拉 → 过度正则 → 抹掉风格","马赛克丢失"),
 ("注意力/K-V 匹配","能搬纹理但内容崩(连参考构图一起搬)","一代未发表(禁用)"),
 ("摊销式 student","抓不住迭代外推(只学到 ~40% 方向)","delta-cos 0.40"),
 ("content-affinity carrier","对残差乘性重加权,inert","5 路探针")]
negrows = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td class='ev'>{c}</td></tr>" for a,b,c in NEG)

def take(t): return f"<div class='take'><b>要点</b> · {t}</div>"

# LaTeX formulas as RAW strings (so f-string doesn't mangle \n \t \f etc. in \nabla \frac ...)
EQ1  = r"$$\Delta s \;=\; -\frac{1-\sigma}{\sigma}\,(v_{+}-v_{-}) \;=\; \nabla \log \frac{p_{+}}{p_{-}}$$"
EQ1b = r"$$g \;=\; (\alpha-1)\,\Delta s$$"
EQ2a = r"$$\mathcal{L}=\big\lVert\,\mathcal{N}\!\left[\,v_{-}-\operatorname{sg}(v_{+})\,\right]\big\rVert^{2}\;\;\Longrightarrow\;\;\min\iff v_{-}=v_{+}$$"
EQ2b = r"$$x_{0}\;\leftarrow\;x_{0}+s\cdot\operatorname{rmsnorm}(g)$$"
EQ3  = r"$$\nabla_{\text{d-DMD}}\;\propto\;\underbrace{\big(s_{\text{cond}}-s_{\text{fake}}\big)}_{\text{DM (shield, needs fake)}}\;+\;\underbrace{(\alpha-1)\big(s_{\text{cond}}-s_{\text{uncond}}\big)}_{\text{CA (spear, frozen)}}$$"
EQ3b = r"$$\text{TSD}\;:\quad g\;=\;(\alpha-1)\big(v_{+}-v_{-}\big)\qquad(\text{CA only — drop DM})$$"

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test-Time Score Distillation — 整篇论文思路</title>
<style>
 :root{{--ink:#1e293b;--mut:#64748b;--line:#e8ecf1;--bg:#fbfcfd;--gd:#16a34a;--rd:#dc2626;--acc:#4f46e5}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16.5px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Segoe UI",Roboto,Helvetica,Arial,sans-serif;letter-spacing:.1px}}
 .wrap{{max-width:760px;margin:0 auto;padding:34px 22px 80px}}
 h1{{font-size:27px;line-height:1.3;margin:0 0 10px;letter-spacing:0}}
 .sub{{color:var(--mut);font-size:15px;margin:0 0 4px}}
 .flag{{display:inline-block;background:#eef2ff;color:#4338ca;font-size:12px;font-weight:700;padding:4px 11px;border-radius:20px;margin-bottom:16px}}
 h2{{font-size:20px;margin:46px 0 6px;padding-top:20px;border-top:1px solid var(--line);letter-spacing:0}}
 .num{{color:var(--acc);font-weight:800;margin-right:8px}}
 p{{margin:12px 0}} .lead{{color:#475569;font-size:15.5px}}
 .take{{background:#f0fdf4;border-left:4px solid var(--gd);border-radius:0 8px 8px 0;padding:10px 15px;margin:14px 0;font-size:14.5px;color:#14532d}}
 .take b{{color:var(--gd)}}
 .tldr{{background:#eef2ff;border:1px solid #dbe0ff;border-radius:14px;padding:18px 20px;margin:18px 0;font-size:15px;line-height:1.8}}
 figure{{margin:20px 0;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(15,23,42,.04)}}
 figure img{{width:100%;height:auto;display:block;background:#f3f4f6}}
 figcaption{{padding:11px 15px;font-size:13.5px;line-height:1.6;color:var(--mut)}} .miss{{color:#b91c1c;font-size:13px}}
 .diagram{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0;text-align:center}}
 .dcap{{font-size:12.5px;color:#94a3b8;margin-top:8px;text-align:center}}
 table{{width:100%;border-collapse:collapse;font-size:13.5px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0}}
 th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}} th{{background:#f8fafc;font-size:12px;color:var(--mut)}} tr:last-child td{{border-bottom:none}}
 td.ev{{color:var(--mut);font-size:12.5px;white-space:nowrap}}
 .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}}
 .stat{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}} .stat .v{{font-size:20px;font-weight:800}} .stat .l{{color:var(--mut);font-size:12px;margin-top:3px;line-height:1.4}}
 .stat.g .v{{color:var(--gd)}} .stat.r .v{{color:var(--rd)}}
 code{{background:#f1f5f9;padding:1px 6px;border-radius:5px;font-size:13.5px}} b.g{{color:var(--gd)}} b.r{{color:var(--rd)}}
 .eq{{background:#f8fafc;border:1px solid #e8ecf1;border-radius:8px;padding:4px 14px;margin:9px 0;overflow-x:auto;overflow-y:hidden;color:#0f172a}}
 .step{{font-weight:700;color:var(--acc);font-size:15.5px;display:block;margin-top:6px}}
 .foot{{color:var(--mut);font-size:13px;line-height:1.7;margin-top:44px;border-top:1px solid var(--line);padding-top:16px}}
</style>
<script>window.MathJax={{tex:{{displayMath:[['$$','$$']],inlineMath:[]}},chtml:{{scale:1.04}},startup:{{typeset:true}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body><div class="wrap">
<div class="flag">思路可视化 · 论文(main.tex)撰写中</div>
<h1>Test-Time Score Distillation (TSD)</h1>
<p class="sub" style="font-size:17px;color:#334155;font-weight:600">用反向过程替代 fake-score 的风格迁移</p>
<p class="sub">冻结整流流(rectified-flow)文生图 transformer 上的免训练风格迁移。下面是完整思路——一节一个要点。</p>
<div class="tldr"><b>一句话。</b>推理时只优化<b>单个 latent</b>,用模型自己的 <b>CFG-augmentation</b> 梯度(带/不带风格参考的速度差)
当作<b>定幅外推步</b>迭代。它有效,是因为外推走<i>过</i>了 score-matching 只能插值到并饱和的那个点;而核心发现是:
DMD 训练时需要的分布匹配(fake-score)项,在推理下 <b class="r">既没必要、还有害</b>——冻结反向过程本身就是免费的"盾"。</div>

<h2><span class="num">1</span>问题</h2>
<p class="lead">把参考图的<b>风格</b>迁到文字 prompt 的主体上,免训练。最朴素的信号——同噪态分数残差用 MSE 匹配
(<code>score_match</code>)——在难风格(马赛克、晶体、3D 渲染)上只搬<b>调色板</b>、不搬<b>纹理</b>。</p>
{take("score_match 只会搬调色板,搬不动纹理。要解决的就是这个。")}

<h2><span class="num">2</span>方法 —— TSD(矛 + 免费的盾)</h2>
<p class="lead">每个去噪步,把模型自己的冻结去噪(<b style="color:#6366f1">盾</b>)与 latent 上的 CFG-augmentation 外推
(<b class="g">矛</b>)交替进行,只优化 latent。</p>
<div class="diagram">{framework_svg}<div class="dcap">图 A · 信号怎么算:same-state 两支 → g</div></div>
<div class="diagram">{loop_svg}<div class="dcap">图 B · 每步怎么迭代:盾 → 矛 → 循环</div></div>
{take("矛 = CFG-aug 外推(驱动风格);盾 = 冻结反向过程(免费,保住不漂)。不训权重、无 fake 网络。")}

<h2><span class="num">2.5</span>公式推导(轻量版)</h2>
<p class="lead">三步就到,不苦大仇深。</p>
<p><span class="step">① same-state 残差 → CFG-augmentation</span>
同一噪声态两次预测 v₊/v₋,速度↔分数 s=−(z+(1−σ)v)/σ,共享的 z/σ 抵消:</p>
<div class="eq">{EQ1}</div><div class="eq">{EQ1b}</div>
<p class="lead" style="font-size:13.5px;color:#94a3b8;margin:2px 0 0">↑ 同一噪声态,共享项抵消 = 图像版 CFG;再放大 (α−1) 成矛。</p>
<p><span class="step">② 插值饱和 vs 外推走过</span>
score_match 最小化残差,最小点在两支相等处 → 朝不动点<b>插值</b>,难风格上那点弱纹理、于是饱和。
TSD 每步施加<b>定幅</b>步,rms 把幅度钉死 → 不管收敛与否都继续走 = <b class="g">外推</b>(α 又被 rms 洗掉,故 cfg4≈20):</p>
<div class="eq">{EQ2a}</div><div class="eq">{EQ2b}</div>
<p class="lead" style="font-size:13.5px;color:#94a3b8;margin:2px 0 0">↑ 上:score_match 的弱不动点(饱和);下:TSD 的定幅步(外推,不随收敛衰减)。</p>
<p><span class="step">③ DMD 分解 & test-time 重铸</span>
decoupled-DMD 把梯度<b>加性</b>拆成两项(原论文 Eq.6),两项都以 s<sub>cond</sub> 为基准——DM 盾 + CA 矛:</p>
<div class="eq">{EQ3}</div>
<div class="eq">{EQ3b}</div>
<p class="lead" style="font-size:13.5px;color:#94a3b8;margin:2px 0 0">↑ DM 盾 = (s<sub>cond</sub>−s<sub>fake</sub>),需训练 fake;CA 矛 = (α−1)(s<sub>cond</sub>−s<sub>uncond</sub>),只需冻结 teacher。<b>TSD 只留 CA、丢 DM</b>(推理下反向过程免费顶替了盾)。</p>
{take("推理·单样本下,反向过程每步把样本收缩回流形 = 盾的角色,免费 → DM 多余;显式再加 → 有害(见 §5)。")}

<h2><span class="num">3</span>突破 —— CFG-augmentation 外推</h2>
<p class="lead">把 g 当<b>定幅步</b>施加,纹理回来了、<b>同时保住内容</b>——这正是区别于"注意力替换"(搬纹理但内容崩)的那条轴。</p>
{figure('caprobe','左起:参考 · score_match(只有调色板)· TSD。马赛克 tesserae、晶体棱面都回来了,而猫 / 老人的内容被保住。')}
{figure('cabatch_gen','推广 + 回归:TSD 修复"纹理贴写实图"的失败,且不砸已经能 work 的风格。少数本就 graphic 的风格保持中性 = 诚实局限。')}
{take("纹理回来 + 内容保住,而且不砸原本 work 的风格。")}

<h2><span class="num">4</span>为什么有效 —— 外推 vs 插值</h2>
<p class="lead">机制(代码验证):score_match 朝"两支相等"的弱不动点递减地拉 → 饱和;TSD 定幅步走过它、持续累积。</p>
<div class="diagram">{mech_svg}<div class="dcap">图 C · 两条轨迹:插值(红,饱和) vs 外推(绿,累积)</div></div>
<p class="lead">直接证据:α 被 <b>rms 归一化洗掉</b>——从 cfg 4 到 20 输出几乎一致,且从不过饱和。</p>
{figure('cfggate','α 推到 7 / 15 / 20 依然几乎一致、无过饱和 —— α 被洗掉,且样本从不漂离流形。')}
{take("杠杆是外推几何,不是放大 α;也不是「就调高了 CFG」。")}

<h2><span class="num">5</span>核心结果 —— 反向过程就是免费的盾</h2>
<p class="lead">DMD 分解为 CFG-aug"矛"+ fake-score"盾",但那是<b>训练生成器</b>的语境(盾挡参数漂移)。我们搬到<b>推理·单样本</b>:
没有参数漂移,冻结反向过程已经把样本收缩回流形 → 盾免费,显式 DM 多余、且有害。</p>
<div class="diagram">{recast_svg}<div class="dcap">图 D · 训练 DMD(盾=训练出的 fake 网) vs 推理 TSD(盾=免费反向过程)</div></div>
{figure('b2_cmp','CA-alone(TSD) vs CA+DM(真实在线 fake critic)。DM 把输出拽回自然主义窗边猫,马赛克被抹掉。')}
{take("在推理下,DM 不只是没必要,而是有害——它过度正则、把风格拉没。")}

<h2><span class="num">6</span>系统研究:什么<i>不行</i>——以及为什么</h2>
<p class="lead">六个"显而易见"的加法,每个都失败、且<b>收敛回 CA-alone</b>,各有机制。这个收敛本身就是结果,也代替了外部对比。</p>
<div class="diagram">{converge_svg}<div class="dcap">图 E · 六路收敛</div></div>
<table><tr><th>替代方案</th><th>为什么不行(机制)</th><th>证据</th></tr>{negrows}</table>
{figure('dmsched_cmp','连"早开晚关"调度的 DM,在边际风格上也没有增益 —— DM 这条杠杆诚实闭环。')}
<p class="lead" style="font-size:14px;color:#94a3b8">注:早期我们曾错误断定"模型无法渲染这些纹理"(基于未外推的单步预测),后经迭代外推证伪并撤回——失败-纠正也是证据。</p>
{take("六路全灭、只剩 CA-alone。收敛即结论。")}

<h2><span class="num">7</span>结果与诚实局限</h2>
<div class="stats">
 <div class="stat g"><div class="v">partial→full</div><div class="l">难风格 style_capture(VLM 基准)</div></div>
 <div class="stat g"><div class="v">full</div><div class="l">内容保住(纹理 AND 内容)</div></div>
 <div class="stat"><div class="v">cfg 4–20</div><div class="l">α 被洗掉(外推,非放大)</div></div>
 <div class="stat r"><div class="v">6 / 6</div><div class="l">替代方案全失败→收敛 CA-alone</div></div>
</div>
<p class="lead"><b>局限(明说):</b>TSD 不训权重,但<b>基于优化</b>(每步几次 AdamW latent 迭代)——不是前馈白嫖。弱方法也能处理的、
已 graphic 的风格保持中性。快速前馈 student 是真实但天级的扩展。</p>

<div class="foot"><b>贡献。</b>(1) TSD — CFG-augmentation 外推的免训练风格迁移;(2) 外推-vs-插值机制;
(3) 反向过程即免费盾(DMD 的 fake-score 在推理下不必要且有害);(4) 展示"纹理 AND 内容"的多维 VLM 基准;
(5) 六路负向系统研究。评测跑在一个冻结的整流流 transformer 上,架构无关。论文(main.tex)撰写中。</div>
</div></body></html>"""

open(OUT, "w").write(html)
print(f"wrote {OUT} ({len(html.encode())/1024:.0f} KB), figures: {sum(1 for v in IM.values() if v)}/{len(IM)}")
