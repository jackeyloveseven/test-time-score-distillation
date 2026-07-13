# Residual Field Profiling Notes

## 1) 脚本目的

`scripts/profile_residual_fields.py` 用于在不启动完整采样实验的前提下，快速得到残差场（residual field）诊断指标，判断第二篇方法中的核心问题：

- OT-CFG residual 是否比 hard-SCFG 更接近 Hyperion 的 style 方向。
- 不同方案下残差场在空间一致性与频率结构上的差异。

默认行为为 dry-run，生成可复现的随机张量样例，不加载 Klein9B。

## 2) 定义与字段

输出字段（按 `ResidualFieldRecord`）

- `style_id`: 风格 ID。
- `prompt_id`: 提示词 ID。
- `timestep`: 当前去噪步长时间（本脚本 dry-run 为给定离散值）。
- `variant`: 残差变体名称（如 `hard_scfg`, `otcfg`, `lrs_otcfg`）。
- `residual_norm`: 残差场 L2 范数。
- `rho`: 空间邻域余弦一致性。
- `cos_to_hyperion_grad`: 与 Hyperion 梯度方向余弦。
- `cos_to_hyperion_delta`: 与 Hyperion 位移方向余弦。
- `hi_freq_ratio`: 高频能量占比。
- `low_freq_ratio`: 低频能量占比。

### hard-SCFG / OT-CFG / Hyperion teacher

- `hard-SCFG`：当前残差场定义为 `v_inject - v_normal`，保留在注意力 token 的硬注入逻辑。
- `OT-CFG`：当前残差场定义为 `v_sinkhorn - v_normal`，使用 Sinkhorn 风格注入。
- `Hyperion teacher`：
  - `hyperion_grad`：Hyperion 的风格梯度（teacher 梯度）。
  - `hyperion_delta`：一次 Hyperion 步后隐变量位移（`z_{t+1} - z_t`）

## 3) dry-run 命令

```bash
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709

diffsynth: /home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/profile_residual_fields.py --dry-run

yzh-dmd:  /home/ubuntu/miniconda3/envs/yzh-dmd/bin/python scripts/profile_residual_fields.py --dry-run
```

默认将生成：

- `results/residual_profile_dryrun.csv`
- `results/residual_profile_dryrun_summary.json`

加 `--env-note` 可打印推荐环境清单。

## 4) Klein9B runtime 接入点

当前脚本保留了模型路径的接口占位（`run_model_path`）。下一阶段接入时：

1. 在 `run_model_path` 中完成以下工作：
   - 加载 diffusers/diffsynth pipeline 与调度器。
   - 对每个 style/prompt/step 获取
     - `v_normal`
     - `v_hard` / `v_otcfg`（基于 hooks）
     - Hyperion teacher field（若可复现）。
   - 调用 `compare_fields` 聚合为 `ResidualFieldRecord`。
2. 将模型路径中的字段转换为统一字典结构传给 `compare_fields`。
3. 输出同名 csv/json 供与 dry-run 的输出结构一致对齐。
4. 保持 profile 端与采样端的 scheduler step 一致性：同一 `(z_t, t)` 下计算 normal 与 styled。
