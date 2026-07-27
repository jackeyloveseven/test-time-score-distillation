# Idea Quality Review — Controllable Style Locality (TSD)

Skill: `idea-quality` (= microsoft/ResearchStudio → ResearchStudio-Idea). Input: `idea_for_quality_eval.md`. Run: 2026-07-22. Absolute track (single idea).

## Idea Review — Controllable Style Locality: one frozen-model score spear

**Decomposition**
- Problem / gap (from Motivation): training-free style transfer is treated as one fixed behavior, flattening two distinct useful tasks (mask-free Semantic Style Editing that keeps the reference composition, and Global Style Transfer that dissolves it); the two strong routes each cost something — DMD trains a fake-score critic network, K/V methods do attention surgery. Gap = a *single frozen-model operator* reaching *both ends* with locality as a *controllable knob*.
- Method (the proposed move): recast style guidance as decoupled-DMD's CFG-augmentation spear `g=(α−1)(v_with−v_without)`, an rms-normalized fixed-step on the latent; claim the frozen T-step reverse process *is* the distribution-matching regularizer so DMD's trained shield is unnecessary **and** harmful (drop it → TSD); make locality a free knob (full spear = semantic, fewer reverse steps = global).
- Why it should work: velocity=conditional-score affine relation makes the spear a principled log-density-ratio gradient; the reverse process already pulls samples toward the data manifold, so a separate trained critic is redundant and over-restrictive.

**Axis scores**

| Axis | Score (1–5) | Evidence (quoted from the idea) | Reason |
|------|-------------|----------------------------------|--------|
| A — Problem position | 4 | "reach *both* ends of the style-locality axis, and can the locality itself be a *controllable knob* rather than a fixed behavior" | 把"风格迁移=一根可控 locality 轴上的两个不同任务"这一 reframing 本身有洞见(命名轴=贡献),且"单冻结算子、免训练、不动 attention"的 gap 真实且未闭合;非 5 是因为该轴紧邻大量 controllable/guided-style 工作,重要性是"有用的能力"而非卡住整个领域的瓶颈。 |
| B — Method quality | 3 | "the frozen T-step reverse process *itself* is the distribution-matching regularizer … the trained fake-score 'shield' … is (i) **unnecessary** … and (ii) **harmful**" | depth: 正则化替代(reverse process 顶替 DMD 训练判别器)是货真价实的 reframing 非增量补丁,单这一点够 4;但外壳=已确立的 CFG 外推 + latent 优化,拉低。soundness: velocity=score 仿射正确,但两个承重命题都停在"经验/合理"层——(a)"reverse process ≡ DMD 项"的等价是断言非证明、有未言明的粒度前提;(b)step-4"少步数溶构图留介质"是把副作用抬成机制,为何少步丢构图却留介质没论证。feasibility: 高(冻结、两次前向、无 oracle/数据),但按 skill 只作 tie-breaker。综合压到 3。 |
| C — Problem-fit | 4 | "run *fewer* reverse steps to dissolve the reference composition while keeping the medium = Global Style Transfer. One operator, one continuous control" | 语义端 + "单冻结算子/免训练/不动 attention"契合极强,正则化替代直接命中"无需训练判别器"子 gap;但 gap 要的是"可控 knob 而非固定行为",全局端却靠"少步数"这一**粗糙副作用**旋钮达成、非原理性 style/content 解耦→"可控 locality"这个野心只部分兑现,故非 5。 |

**Overall: 67 / 100  ·  Verdict: strong(擦线,实为 borderline-strong)**
A/C 门未触发(A=4>2, C=4>2,无封顶)。但 67 恰落在 strong≥67 的临界点上——诚实读法是"刚迈进 strong 的 borderline-strong",B 的 soundness 缺口 + 全局端契合的软肋正是把它按在临界线的原因。

**Strongest point:** 正则化替代——论证冻结多步反向过程本身就是 DMD 的分布匹配正则器,使训练 fake-score 判别器既没必要又有害——是审稿人能一眼认出的真概念贡献,不是调参补丁。

**Most fixable weakness:** 全局迁移端目前压在"少步数"这个粗糙副作用上;要么给它一个**推导出来的机制级旋钮**(原理性 style/content 解耦),要么诚实地把全局迁移界定为"免费但钝"的杠杆、把贡献重心压在语义编辑 + 正则化替代上——两者都能同时抬高 B-soundness 与 C。
