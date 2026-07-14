# Adaptation Notes

Status: historical migration record. Current ownership, method, and commands
are in `PROJECT_STATE.md` and `AGENT_HANDOFF.md`.

## 1) 已从旧项目复制/适配的组件

- 从 `custom_main_klein9b_t2i.py` 中提取并重构：
  - `Klein9BCache`
  - `Klein9BController`
  - `register_klein9b_attn_hooks`
    - 保留 `cache` / `inject` / `sinkhorn` 三种模式。
    - `inject` 与 `sinkhorn` 分别对应旧逻辑的硬注入与 Sinkhorn 注入。
  - 关键张量算子：
    - `sinkhorn_attn`
    - `sinkhorn_attn_freq`
    - `sinkhorn_style_pattern`
    - `feature_adain`
    - `adain_1d`
    - `sinkhorn_latent_transport`
- 新建轻量 wrapper：
  - `compute_normal_velocity`
  - `compute_styled_velocity`
  - `compute_style_velocity_residual`
  - `apply_velocity_guidance`
  - `compute_velocity_tuple`

## 2) 本次修正（优先修复点）

1. `scripts/sanity_import.py` 路径与异常处理修正
   - 已在脚本头部将项目根目录加入 `sys.path`，避免 `import src.*` 误报为 `ModuleNotFoundError`。
   - `torch` 仅在 `import torch` 失败时打印 `OK (torch not installed; skipped)`。
   - `src.*` 的 import 与后续调用不再被吞掉，缺失时会直接抛错，便于真实定位。

2. `sinkhorn_attn` / `sinkhorn_attn_freq` 长度语义修正
   - 这两个函数不再截断 `Q`。
   - 只对 `K_style` 和 `V_style` 做长度对齐（按 `min(Lk, Lv)`）；输出长度始终与输入 `Q` 的 `Lq` 一致。
   - 这与旧流程中“Q 为内容 token、K/V 为 style text+image token、Q 长度不应丢失”的语义一致。

3. `residuals` 的注入参数隔离
   - `compute_velocity_tuple` 改为显式参数：
     `mode`, `guidance_scale`, `inject_beta`, `sinkhorn_tau`, `sinkhorn_iters`,
     `sinkhorn_only_last_n`, `sinkhorn_use_pattern`, `model_kwargs`。
   - normal/styled forward 共享 `model_kwargs`，注入参数只进入 styled 逻辑，不会注入到 dit forward。
   - `scheduler.step` 仍未进入 wrapper；仅输出 velocity 并在调用方一次性 step。

## 3) 额外新增 sanity 机制

- `scripts/sanity_import.py` 新增轻量 fake-DiT 验证：
  - 检查 `compute_velocity_tuple` 的代数关系：`residual = styled - normal`、
    `guided = normal + guidance_scale * (styled - normal)`。
  - 同时检测注入参数不泄漏到 dit forward（通过 fake-DiT 的参数白名单断言）。

## 4) 本次刻意未复制的内容

- 未复制旧脚本中的完整实验循环、参数解析、采样循环和模型加载逻辑。
- 未复制训练、落盘、可视化、日志与评估脚本。
- 未加载任何大模型；sanity 文件仅做张量级 smoke test。

## 5) 待主线程确认

1. `sinkhorn_style_pattern` 在长度边界下（例如 `txt_len` 与内容长度差异较大）是否需要与旧实现完全一致的 padding/crop 行为。
2. `sinkhorn_latent_transport` 的 `K_style/V_style` 输入长度不一致时是否需要更严格的断言，当前保持最小重叠长度策略。
3. `attention_hooks.py` 仍保持 `diffsynth.models.flux2_dit` 的运行时导入，若希望完全脱离可考虑改为可注入的 processor 抽象。

## 6) 验收环境与优先级（更新）

- 采用的验收顺序更新为：
  1. `python3 -m py_compile src/*.py scripts/sanity_import.py`
  2. `/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sanity_import.py`
  3. `/home/ubuntu/miniconda3/envs/yzh-dmd/bin/python scripts/sanity_import.py`
- 依据远端环境检查：
  - `diffsynth` 为首选（包含 diffsynth 及 CUDA 运行栈，适配后续 Klein9B runtime）。
  - `yzh-dmd` 为备选（同样包含 diffsynth）。
  - `sensenova` 仅保留为纯 torch 张量测试用途，缺少 diffsynth，不用于后续 Klein9B 接口验证。
