# Klein9B Probe Notes

Status: chronological experiment ledger. Use `RESULTS_CATALOG.md` for current
paper numbers and `PROJECT_STATE.md` for decisions.

## 当前状态

`scripts/probe_klein9b_residuals.py` 已支持三类路径：

- dry-run（默认）：合成张量，生成 profiler-compatible csv / summary / meta。
- `--run-model --runtime-smoke-only`：加载 DiffSynth/Klein9B，验证 prompt/style latent/scheduler 形状，不做 DiT forward。
- `--run-model`：真实单样本 residual probe，计算同一 `(z_t, t)` 上的 `v_normal`、`v_hard`、`v_ot`，输出 `hard_scfg` 与 `otcfg`。

同时已新增 pairwise residual diagnostics：

- `<out-prefix>_pairwise.csv`
- `<out-prefix>_pairwise_summary.json`

当前 pairwise 记录 `hard_scfg` vs `otcfg`：

- flattened cosine
- `diff_norm = ||otcfg - hard_scfg||`
- `norm_ratio_b_over_a = ||otcfg|| / ||hard_scfg||`
- diff field 的 high/low frequency ratio

## 重要参数

- `--height`, `--width`：仅用于 synthetic dry-run，默认 `32x32`。
- `--runtime-height`, `--runtime-width`：真实 Klein9B 路径图像尺寸，默认 `512x512`。
- `--selected-timesteps`：请求采样的 scheduler timestep，例如 `700` 或 `700,500,300`。
- `--self-layer-start`, `--self-layer-end`：hook 的 self-attention layer 范围，默认 `[12, 28)`。
- `--inject-beta`：hard K/V injection blend 系数，默认 `1.0`。
- `--sinkhorn-tau`, `--sinkhorn-iters`：OT/Sinkhorn residual 参数。

## 已验证命令

### dry-run

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --dry-run \
  --out-prefix residual_profile_klein9b_probe_dryrun_realpathcheck
```

输出：

- `results/residual_profile_klein9b_probe_dryrun_realpathcheck.csv`
- `results/residual_profile_klein9b_probe_dryrun_realpathcheck_summary.json`
- `results/residual_profile_klein9b_probe_dryrun_realpathcheck_meta.json`

### runtime smoke

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --runtime-smoke-only \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --out-prefix residual_profile_klein9b_runtime_smoke_dimcheck
```

关键结果：

- `latent_shape = [1, 1024, 128]`
- `style_latent_shape = [1, 1024, 128]`
- requested timestep `[700, 500, 300]` 对应实际 scheduler timestep `[718.3221, 472.0755, 363.1470]`

### real residual smoke

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700 \
  --out-prefix residual_profile_klein9b_smoke \
  --no-hyperion-grad \
  --sinkhorn-iters 6
```

输出：

- `results/residual_profile_klein9b_smoke.csv`
- `results/residual_profile_klein9b_smoke_summary.json`
- `results/residual_profile_klein9b_smoke_meta.json`

CSV 当前结果：

```text
variant,timestep,residual_norm,rho,hi_freq_ratio,low_freq_ratio
hard_scfg,718.3220825195312,469.0926208496094,0.35127267241477966,0.4304223954677582,0.5695776343345642
otcfg,718.3220825195312,479.05914306640625,0.32293057441711426,0.42153793573379517,0.5784620642662048
```

meta 关键字段：

- `mode = run_model`
- `prompt_text = A panda eat bamboo`
- `selected_indices = [18]`
- `actual_timesteps = [718.3220825195312]`
- `latent_shape = [1, 1024, 128]`
- `style_latent_shape = [1, 1024, 128]`
- `image_ids_shape = [1, 1024, 4]`
- `self_layers = [12, 28]`

### real residual 3-step smoke

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_klein9b_3step_smoke \
  --no-hyperion-grad \
  --sinkhorn-iters 6
```

输出：

- `results/residual_profile_klein9b_3step_smoke.csv`
- `results/residual_profile_klein9b_3step_smoke_summary.json`
- `results/residual_profile_klein9b_3step_smoke_meta.json`

实际 scheduler timesteps：

- requested: `[700.0, 500.0, 300.0]`
- selected indices: `[18, 22, 23]`
- actual: `[718.3220825195312, 472.07550048828125, 363.1470031738281]`

当前 3-step summary：

```text
variant,n,residual_norm_mean,residual_norm_std,rho_mean,hi_freq_ratio_mean
hard_scfg,3,367.7356872558594,73.8099406998484,0.24581635991732279,0.4033626814683278
otcfg,3,429.7097473144531,38.37044049964392,0.252120574315389,0.4075594941775004
```

初步观察：

- `otcfg` 的 residual norm 均值更高，并且跨 timestep 方差更低。
- `hard_scfg` 随 timestep 降低衰减更明显。
- 这支持后续把 OT/Sinkhorn residual 当作更稳定的 teacher 候选，而不是只蒸馏 hard injection。

### real residual 3-step pairwise smoke

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_klein9b_3step_pairwise \
  --no-hyperion-grad \
  --sinkhorn-iters 6
```

输出：

- `results/residual_profile_klein9b_3step_pairwise.csv`
- `results/residual_profile_klein9b_3step_pairwise_summary.json`
- `results/residual_profile_klein9b_3step_pairwise_meta.json`
- `results/residual_profile_klein9b_3step_pairwise_pairwise.csv`
- `results/residual_profile_klein9b_3step_pairwise_pairwise_summary.json`

当前 pairwise summary：

```text
pair,n,cosine_mean,cosine_std,diff_norm_mean,norm_ratio_b_over_a_mean,diff_hi_freq_ratio_mean
hard_scfg__vs__otcfg,3,0.8051637808481852,0.009738486271378384,257.98016357421875,1.193231225013733,0.7176442344983419
```

当前 pairwise 观察：

- `hard_scfg` 与 `otcfg` 方向高度相关，但不是同一个 residual：平均 cosine 约 `0.805`。
- `otcfg` 平均范数约为 `hard_scfg` 的 `1.19x`。
- 二者差异主要集中在高频：`otcfg - hard_scfg` 的 high-frequency ratio 约 `0.718`。
- 这说明 OT/Sinkhorn 不只是放大 hard injection；它可能补充了更偏纹理/笔触的高频 style residual。后续蒸馏时可以考虑 `hard + high-frequency OT correction`，而不是直接全量蒸一个混合残差。

### highpass teacher sweep

已在代码中新增候选 teacher：

```text
teacher_hp(lambda) = hard_scfg + lambda * highpass(otcfg - hard_scfg)
```

相关 CLI：

- `--teacher-lambdas`：默认 `0,0.25,0.5,0.75,1.0`
- `--teacher-highpass-kernel`：默认 `5`
- `--vram-limit-gb`：传给 DiffSynth `from_pretrained(vram_limit=...)`，用于低显存环境。

已验证 dry-run：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --dry-run \
  --out-prefix residual_profile_teacher_sweep_dryrun \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5
```

输出：

- `results/residual_profile_teacher_sweep_dryrun.csv`
- `results/residual_profile_teacher_sweep_dryrun_summary.json`
- `results/residual_profile_teacher_sweep_dryrun_pairwise.csv`
- `results/residual_profile_teacher_sweep_dryrun_pairwise_summary.json`
- `results/residual_profile_teacher_sweep_dryrun_meta.json`

dry-run 记录数：

- main records: `24`
- pairwise records: `33`
- variants: `hard_scfg`, `otcfg`, `teacher_hp_lam0`, `teacher_hp_lam0p25`, `teacher_hp_lam0p5`, `teacher_hp_lam0p75`, `teacher_hp_lam1`, `lrs_otcfg`

### offline teacher sweep postprocess

## Residual adapter distillation smoke

已新增最小 student adapter 训练闭环：

- `src/student_adapter.py`
- `scripts/train_residual_adapter.py`

当前 adapter 设计：

```text
pred_teacher = hard_scfg + adapter(hard_scfg, timestep, style_id, prompt_id)
target_teacher = hard_scfg + highpass(otcfg - hard_scfg)
```

也就是保留 hard CFG residual 作为基线，只蒸馏 teacher 的高频修正项。这样后续接回采样器时可以先用较小系数注入 student correction，而不是替换完整 velocity/residual 路径。

使用的数据集：

```text
results/residual_distill_style_sweep_small_lam1.jsonl
```

数据规模：

- `27` samples
- styles: `00`, `13`, `27`
- prompts: `kp0`, `kp2`, `kp4`
- timesteps: `363.1470`, `472.0755`, `718.3221`
- field shape: `[1, 1024, 128]`
- teacher: `teacher_hp_lam1`

短训 smoke：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_residual_adapter.py \
  --manifest results/residual_distill_style_sweep_small_lam1.jsonl \
  --out-dir results/residual_adapter_smoke \
  --epochs 8 \
  --batch-size 3 \
  --device cpu \
  --hidden-dim 128 \
  --cond-dim 48 \
  --time-embed-dim 48 \
  --lr 1e-3 \
  --log-every 1
```

结果：

- best epoch: `8`
- val MSE: `0.43904`
- hard baseline val MSE: `0.44768`
- val MSE improvement: `0.00865`
- val cosine: `0.82996`
- hard baseline val cosine: `0.82829`
- outputs:
  - `results/residual_adapter_smoke/adapter_best.pt`
  - `results/residual_adapter_smoke/adapter_last.pt`
  - `results/residual_adapter_smoke/config.json`
  - `results/residual_adapter_smoke/metrics.json`
  - `results/residual_adapter_smoke/train_log.csv`

稍长训练：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_residual_adapter.py \
  --manifest results/residual_distill_style_sweep_small_lam1.jsonl \
  --out-dir results/residual_adapter_smoke_long \
  --epochs 60 \
  --batch-size 3 \
  --device cpu \
  --hidden-dim 256 \
  --cond-dim 64 \
  --time-embed-dim 64 \
  --lr 2e-3 \
  --log-every 10
```

最佳验证结果：

- best epoch: `12`
- val MSE: `0.43417`
- hard baseline val MSE: `0.44768`
- val MSE improvement: `0.01351`, about `3.0%`
- val cosine: `0.83137`
- hard baseline val cosine: `0.82829`
- best checkpoint: `results/residual_adapter_smoke_long/adapter_best.pt`

过拟合观察：

- epoch 60 train MSE 降到 `0.36163`，但 val MSE 回升到 `0.45408`。
- 说明当前 27-sample manifest 足以验证训练闭环和 teacher 可学习性，但不足以支持最终 student。
- 下一步优先扩充 style/prompt/timestep 数据，再考虑低秩 adapter、spectral loss、delta norm regularization，最后再接回 Klein9B sampler 做图像级评估。

## Residual adapter ablation

已新增训练脚本指标/约束：

- `--spectral-weight`：对 `pred_delta` 与 `target_delta` 的 highpass 分量加 MSE。
- `--spectral-kernel`：highpass kernel，默认 `5`。
- `--delta-norm-weight`：约束 correction norm 的 log-ratio。
- 训练日志新增 `highpass_delta_mse`、`delta_norm_ratio`、`loss_spectral_mse`、`loss_delta_norm_loss`。

已生成 teacher ablation manifest：

- `results/residual_distill_style_sweep_small_hard.jsonl`
- `results/residual_distill_style_sweep_small_ot.jsonl`
- `results/residual_distill_style_sweep_small_lam1.jsonl`

已生成汇总：

- `results/residual_adapter_ablation_summary.csv`
- `results/residual_adapter_ablation_summary.json`

当前同 split / 同 backbone 的 best checkpoint 对比：

```text
run          best_epoch  val_mse   baseline_val_mse  improvement  val_cos
hard_noop    30          0.00000   0.00000           ~0           1.00001
ot_full      24          0.48286   0.52231           0.03945      0.82201
hp_plain     12          0.43417   0.44768           0.01351      0.83137
hp_spectral  19          0.44044   0.44768           0.00724      0.82817
```

解释：

- `hard_noop` 是 sanity check：teacher 等于输入 hard residual，目标 correction 为零；结果接近零误差，说明训练/manifest 路径没有错。
- `ot_full` 相比自己的 hard baseline 有明显改善，但绝对 val MSE `0.48286`，比 `hp_plain` 的 `0.43417` 更难学。
- `hp_plain` 仍是当前最强 teacher/student target，支持论文主线：不要直接蒸 full OT residual，而是蒸 `hard_scfg + highpass(otcfg - hard_scfg)`。
- 当前 `hp_spectral` 配置没有收益，可能是 27 samples 下 highpass loss 与 MSE 目标冲突，暂不作为默认训练配置。

下一步 adapter 侧优先级：

- 扩数据，而不是先继续调小样本训练超参。
- 对 `hp_plain` 做更大 style/prompt/timestep manifest。
- 再试低秩/门控 adapter，目标是减少 27-sample 下的过拟合。
- 接采样器时先用 `student_scale < 1` 注入 correction，避免小样本 student 过强破坏 content。

## 45-sample prompt expansion

GPU 可用后补了一轮真实 residual sweep：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_style_sweep.py \
  --probe-mode run-model \
  --style-ids 00,13,27 \
  --prompt-ids kp1,kp3 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_style_sweep_expand_kp13 \
  --results-dir results \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 6 \
  --runtime-height 512 \
  --runtime-width 512 \
  --min-free-gb 70
```

新增数据：

- styles: `00`, `13`, `27`
- prompts: `kp1`, `kp3`
- timesteps: `363.1470`, `472.0755`, `718.3221`
- new dumps: `18`
- combined with previous `27` dumps -> `45` samples

新增 prompt sweep 结果：

```text
style00 kp1 best = teacher_hp_lam1 score=0.886169
style00 kp3 best = teacher_hp_lam1 score=0.894441
style13 kp1 best = teacher_hp_lam1 score=0.887509
style13 kp3 best = teacher_hp_lam1 score=0.897019
style27 kp1 best = teacher_hp_lam1 score=0.882311
style27 kp3 best = teacher_hp_lam1 score=0.891510
```

新增 18-sample global ranking：

```text
best teacher = teacher_hp_lam1
score = 0.8897353269322312
n = 18
cosine_mean = 0.9795136551062266
diff_norm_mean = 87.80111482408311
norm_ratio_mean = 1.0237165590127308
```

合并 45-sample global ranking：

```text
best teacher = teacher_hp_lam1
score = 0.8885413021374088
n = 45
cosine_mean = 0.9792421685324775
diff_norm_mean = 88.59452633327908
norm_ratio_mean = 1.0263540612326727
diff_hi_freq_ratio_mean = 0.3603623398476177
```

相关输出：

- `results/residual_profile_style_sweep_expand_kp13_global_teacher_ranking.json`
- `results/residual_profile_style_sweep_45_global_pairwise.csv`
- `results/residual_profile_style_sweep_45_global_teacher_ranking.json`

manifest builder 已更新，`--field-dumps` 支持重复传入多个 glob。45-sample manifest：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/build_distill_manifest.py \
  --field-dumps "results/field_dumps/residual_profile_style_sweep_small_style*_probe_*.pt" \
  --field-dumps "results/field_dumps/residual_profile_style_sweep_expand_kp13_style*_probe_*.pt" \
  --out-prefix residual_distill_style_sweep_45_lam1 \
  --teacher-kind highpass_delta \
  --teacher-lambda 1.0 \
  --teacher-highpass-kernel 5 \
  --ranking-json results/residual_profile_style_sweep_expand_kp13_global_teacher_ranking.json \
  --verify-teacher
```

summary:

```text
num_samples = 45
styles = 00:15, 13:15, 27:15
prompts = kp0:9, kp1:9, kp2:9, kp3:9, kp4:9
timesteps = 363.147003:15, 472.075500:15, 718.322083:15
teacher = teacher_hp_lam1
```

## 45-sample adapter structure ablation

已新增：

- `LowRankResidualAdapter`
- `--adapter-type mlp|lowrank`
- `--low-rank-dim`
- `--no-gate`

45-sample structure summary：

- `results/residual_adapter_hp45_structure_summary.csv`
- `results/residual_adapter_hp45_structure_summary.json`

当前 best checkpoint 对比：

```text
run                    best_epoch  val_mse   baseline_val_mse  improvement  val_cos
hp27_mlp               12          0.43417   0.44768           0.01351      0.83137
hp45_mlp               18          0.42484   0.43757           0.01273      0.83313
hp45_lowrank32_gated   21          0.42915   0.43757           0.00842      0.83176
hp45_lowrank64_nogate  19          0.42553   0.43757           0.01203      0.83277
```

观察：

- 扩 prompt 后，MLP adapter 的 best val MSE 从 `0.43417` 降到 `0.42484`，说明扩数据比小样本调参更有效。
- `lowrank32_gated` 后期过拟合更轻，但 best 不如 MLP，说明当前 rank/gate 约束偏保守。
- `lowrank64_nogate` 接近 MLP，但仍略差；低秩结构可作为稳定性 ablation，不应替代当前 best。
- 当前最强 student checkpoint 是 `results/residual_adapter_hp45_plain/adapter_best.pt`。

## Adapter scale sweep and runtime

已新增：

- `scripts/evaluate_residual_adapter.py`
- `src/adapter_runtime.py`
- `ResidualAdapterRuntime`

scale sweep 用途：在接回 sampler 前，确定 `hard_scfg + student_scale * adapter_delta` 的注入系数。`student_scale=0` 等价于 hard baseline。

MLP checkpoint 细粒度 sweep：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/evaluate_residual_adapter.py \
  --manifest results/residual_distill_style_sweep_45_lam1.jsonl \
  --checkpoint results/residual_adapter_hp45_plain/adapter_best.pt \
  --out-prefix residual_adapter_hp45_plain_eval_fine \
  --student-scales 0,0.25,0.5,0.75,1,1.25,1.5,1.75,2 \
  --batch-size 3 \
  --device cpu \
  --seed 43 \
  --val-fraction 0.2
```

MLP val scale sweep：

```text
scale  val_mse  improvement  val_cos   delta_norm_ratio
0.00   0.43757  0.00000      0.82994   0.000
0.50   0.42488  0.01269      0.83379   0.118
0.75   0.42328  0.01429      0.83406   0.177
1.00   0.42484  0.01273      0.83313   0.236
1.50   0.43745  0.00012      0.82745   0.354
2.00   0.46271 -0.02514      0.81644   0.473
```

LowRank64/no-gate best scale：

```text
best scale = 0.75
val_mse = 0.42490
val_cos = 0.83335
```

结论：

- 当前 sampler 默认建议使用 `student_scale=0.75`。
- `scale=1.0` 不是坏点，但 residual-space MSE/cosine 都略低于 `0.75`。
- 不应按 delta norm ratio 盲目放大到 `4x`；scale `2.0+` 已明显破坏 teacher matching。
- 图像级评估建议比较 `student_scale = 0.5, 0.75, 1.0` 三档。

runtime smoke：

```text
adapter_type = mlp
student_scale = 0.75
hard_shape = (1, 1024, 128)
delta_shape = (1, 1024, 128)
out_shape = (1, 1024, 128)
finite = True
```

接 sampler 的最小调用形式：

```python
from src.adapter_runtime import ResidualAdapterRuntime

runtime = ResidualAdapterRuntime.from_checkpoint(
    "results/residual_adapter_hp45_plain/adapter_best.pt",
    device="cuda",
    student_scale=0.75,
)
styled_residual = runtime.apply(
    hard_residual,
    timestep=t,
    style_id="27",
    prompt_id="kp0",
)
```

相关输出：

- `results/residual_adapter_hp45_plain_eval_fine_scale_sweep.csv`
- `results/residual_adapter_hp45_plain_eval_fine_scale_sweep_summary.json`
- `results/residual_adapter_hp45_lowrank64_eval_scale_sweep.csv`
- `results/residual_adapter_hp45_lowrank64_eval_scale_sweep_summary.json`

## Online student residual probe

`scripts/probe_klein9b_residuals.py` 已支持在线加载 adapter：

- `--adapter-checkpoint`
- `--adapter-device`
- `--student-scales`
- `--student-main-scale`

其中 `--student-scales` 只增加 profile variants；`--student-main-scale` 才会把 student residual 放入当前 selected timestep 的 scheduler prediction：

```text
v_student = v_normal + hard_scfg + student_main_scale * adapter_delta
```

在线 3-step probe：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_online_student_style27_kp0_3step \
  --results-dir results \
  --teacher-lambdas 1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 6 \
  --runtime-height 512 \
  --runtime-width 512 \
  --adapter-checkpoint results/residual_adapter_hp45_plain/adapter_best.pt \
  --adapter-device cpu \
  --student-scales 0.5,0.75,1.0 \
  --no-hyperion-grad
```

3-step pairwise summary:

```text
pair                                      cosine   diff_norm
hard_scfg -> teacher_hp_lam1             0.82150  239.620
student_scfg_s0p5 -> teacher_hp_lam1     0.82958  232.925
student_scfg_s0p75 -> teacher_hp_lam1    0.83193  230.981
student_scfg_s1 -> teacher_hp_lam1       0.83302  230.014

hard_scfg -> otcfg                       0.80516  257.980
student_scfg_s0p5 -> otcfg               0.81421  250.879
student_scfg_s0p75 -> otcfg              0.81709  248.619
student_scfg_s1 -> otcfg                 0.81876  247.262
teacher_hp_lam1 -> otcfg                 0.97900   88.088
```

解释：

- 在线 forward 路径里，student residual 相比 hard residual 更接近 `teacher_hp_lam1` 和 `otcfg`。
- 在 style27/kp0 这个 3-step 点上 `s1` 最强，但全 45-sample residual-space validation 中 `s0.75` 最稳。
- 图像级默认仍建议先用 `student_main_scale=0.75`，并保留 `0.5/1.0` 作为 ablation。

主路径 scheduler-step smoke：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700 \
  --out-prefix residual_profile_online_student_main_s0p75_style27_kp0_t700 \
  --results-dir results \
  --teacher-lambdas 1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 6 \
  --runtime-height 512 \
  --runtime-width 512 \
  --adapter-checkpoint results/residual_adapter_hp45_plain/adapter_best.pt \
  --adapter-device cpu \
  --student-scales 0.5,0.75,1.0 \
  --student-main-scale 0.75 \
  --no-hyperion-grad
```

结果：

```text
student_main_scale = 0.75
num_records = 6
num_pairwise_records = 12
invariant = same (z_t,t), one scheduler.step per main-path timestep
```

相关输出：

- `results/residual_profile_online_student_style27_kp0_3step.csv`
- `results/residual_profile_online_student_style27_kp0_3step_pairwise.csv`
- `results/residual_profile_online_student_style27_kp0_3step_student_summary.json`
- `results/residual_profile_online_student_main_s0p75_style27_kp0_t700_meta.json`

已新增：

```text
scripts/postprocess_teacher_sweep.py
```

用途：读取 `probe_klein9b_residuals.py --dump-fields` 生成的 `.pt`，离线重算 highpass teacher sweep，不需要加载 DiffSynth/Klein9B，也不需要 GPU。

推荐真实采样时打开 dump：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_klein9b_base_dump \
  --no-hyperion-grad \
  --sinkhorn-iters 6 \
  --teacher-lambdas 0 \
  --dump-fields
```

然后离线扫 lambda：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/postprocess_teacher_sweep.py \
  --field-dumps "results/field_dumps/residual_profile_klein9b_base_dump_*.pt" \
  --out-prefix residual_profile_klein9b_base_dump_teacher_offline \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5
```

已验证 dry-run fixture：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --dry-run \
  --dump-fields \
  --out-prefix residual_profile_postprocess_fixture \
  --selected-timesteps 700,500 \
  --teacher-lambdas 0,0.5,1.0 \
  --teacher-highpass-kernel 5

/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/postprocess_teacher_sweep.py \
  --field-dumps "results/field_dumps/residual_profile_postprocess_fixture_*.pt" \
  --out-prefix residual_profile_postprocess_fixture_offline \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5
```

验证输出：

- processed dumps: `2`
- main records: `14`
- pairwise records: `22`
- `results/residual_profile_postprocess_fixture_offline.csv`
- `results/residual_profile_postprocess_fixture_offline_pairwise.csv`
- `results/residual_profile_postprocess_fixture_offline_meta.json`

### teacher lambda selection

已新增：

```text
scripts/select_teacher_lambda.py
```

用途：读取 `*_pairwise.csv`，筛选 `teacher_hp_lam* -> otcfg` 行，自动给 lambda 排名。

默认评分：

```text
score = cosine_mean
        - 0.25 * normalized_diff_norm_mean
        - 0.15 * abs(log(norm_ratio_mean))
        - 0.10 * cosine_std
```

含义：

- 奖励 teacher 与 `otcfg` 方向一致。
- 惩罚 teacher 到 `otcfg` 的差异范数过大。
- 惩罚范数比例明显偏离 1。
- 惩罚跨 timestep/style/prompt 不稳定。

fixture 验证命令：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/select_teacher_lambda.py \
  --pairwise-csv results/residual_profile_postprocess_fixture_offline_pairwise.csv \
  --out-prefix results/residual_profile_postprocess_fixture_offline
```

输出：

- `results/residual_profile_postprocess_fixture_offline_teacher_ranking.csv`
- `results/residual_profile_postprocess_fixture_offline_teacher_ranking.json`

fixture 当前排名：

```text
rank,teacher,score,cosine_mean,diff_norm_mean,norm_ratio_mean
1,teacher_hp_lam1,0.9160217094582241,0.9714874923229218,29.547205924987793,1.015462875366211
2,teacher_hp_lam0p75,0.829571717566213,0.9509609341621399,44.991512298583984,1.3094457387924194
3,teacher_hp_lam0p5,0.6303274130100126,0.8431923389434814,74.0489273071289,1.7007777094841003
4,teacher_hp_lam0p25,0.2304589087491744,0.5230695009231567,106.06603240966797,1.971575379371643
5,teacher_hp_lam0,-0.27427110679648814,0.06179848499596119,139.0127716064453,1.7736573219299316
```

注意：fixture 是 synthetic dry-run，只验证脚本逻辑；真实 teacher 选择以后应以 Klein9B dump 的 postprocess 输出为准。

### end-to-end teacher pipeline

已新增：

```text
scripts/run_teacher_pipeline.py
```

用途：串联三步：

1. `probe_klein9b_residuals.py --dump-fields`
2. `postprocess_teacher_sweep.py`
3. `select_teacher_lambda.py`

dry-run 回归命令：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_teacher_pipeline.py \
  --probe-mode dry-run \
  --out-prefix residual_profile_pipeline_dryrun \
  --selected-timesteps 700,500 \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5 \
  --height 32 \
  --width 32
```

已验证输出：

- `results/residual_profile_pipeline_dryrun_probe.csv`
- `results/residual_profile_pipeline_dryrun_teacher.csv`
- `results/residual_profile_pipeline_dryrun_teacher_pairwise.csv`
- `results/residual_profile_pipeline_dryrun_teacher_teacher_ranking.csv`
- `results/residual_profile_pipeline_dryrun_teacher_teacher_ranking.json`
- `results/residual_profile_pipeline_dryrun_pipeline_manifest.json`

dry-run pipeline 当前 best：

```text
teacher_hp_lam1, score=0.9022763161482419
```

GPU 可用后的标准真实命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_teacher_pipeline.py \
  --probe-mode run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_klein9b_teacher_pipeline \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 6 \
  --min-free-gb 40
```

`run-model` 模式默认会检查 GPU free memory；不足 `--min-free-gb` 时拒绝启动，避免在 transformer 加载阶段再次 OOM。若已有 dumps，可直接：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_teacher_pipeline.py \
  --probe-mode postprocess-only \
  --field-dumps "results/field_dumps/residual_profile_klein9b_teacher_pipeline_probe_*.pt" \
  --out-prefix residual_profile_klein9b_teacher_pipeline_reselect \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0
```

真实 pipeline 已于 GPU 释放后跑通：

```text
out_prefix = residual_profile_klein9b_teacher_pipeline
style_id = 27
prompt_id = kp0
selected_timesteps = 700,500,300
actual_timesteps = 718.3220825195312, 472.07550048828125, 363.1470031738281
```

关键产物：

- `results/residual_profile_klein9b_teacher_pipeline_probe.csv`
- `results/residual_profile_klein9b_teacher_pipeline_probe_meta.json`
- `results/field_dumps/residual_profile_klein9b_teacher_pipeline_probe_kp0_27_t718.pt`
- `results/field_dumps/residual_profile_klein9b_teacher_pipeline_probe_kp0_27_t472.pt`
- `results/field_dumps/residual_profile_klein9b_teacher_pipeline_probe_kp0_27_t363.pt`
- `results/residual_profile_klein9b_teacher_pipeline_teacher.csv`
- `results/residual_profile_klein9b_teacher_pipeline_teacher_pairwise.csv`
- `results/residual_profile_klein9b_teacher_pipeline_teacher_teacher_ranking.csv`
- `results/residual_profile_klein9b_teacher_pipeline_teacher_teacher_ranking.json`
- `results/residual_profile_klein9b_teacher_pipeline_pipeline_manifest.json`

真实 pipeline ranking：

```text
rank,teacher,score,cosine_mean,diff_norm_mean,norm_ratio_mean,diff_hi_freq_ratio_mean
1,teacher_hp_lam1,0.8888838992150765,0.9789955019950867,88.08778635660808,1.0320684909820557,0.27143000066280365
2,teacher_hp_lam0p75,0.8510941453019555,0.9705471197764078,108.13178253173828,1.1014761527379353,0.36026037732760113
3,teacher_hp_lam0p5,0.7724209418179605,0.9410001238187155,151.00580342610678,1.1575587193171184,0.52182404200236
4,teacher_hp_lam0p25,0.6630755633252124,0.8858975768089294,202.7070770263672,1.190677523612976,0.6415926615397135
5,teacher_hp_lam0,0.5276901908531572,0.8051637808481852,257.98016357421875,1.193231225013733,0.7176442344983419
```

当前结论：

- `teacher_hp_lam1` 是这组真实 Klein9B 单样本 3-step 的最佳候选。
- 相比 hard baseline (`teacher_hp_lam0`)，`teacher_hp_lam1` 到 `otcfg` 的 cosine 从 `0.805` 提升到 `0.979`。
- diff norm 从 `257.98` 降到 `88.09`。
- norm ratio 从 `1.193` 降到 `1.032`，更接近同尺度 teacher。
- 剩余差异主要偏低频（`diff_low_freq_ratio ~= 0.729`），说明 highpass correction 已经覆盖了 hard/OT 之间的大部分高频差异。

### prompt sweep

已新增：

```text
scripts/run_prompt_sweep.py
```

用途：对多个 prompt 串行运行 `run_teacher_pipeline.py`，合并各 prompt 的 `*_teacher_pairwise.csv`，再全局调用 `select_teacher_lambda.py`。

dry-run 验证命令：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_prompt_sweep.py \
  --probe-mode dry-run \
  --style-id 27 \
  --prompt-ids kp0,kp1 \
  --selected-timesteps 700 \
  --out-prefix residual_profile_prompt_sweep_dryrun \
  --teacher-lambdas 0,0.5,1 \
  --height 32 \
  --width 32
```

真实 prompt sweep 命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_prompt_sweep.py \
  --probe-mode run-model \
  --style-id 27 \
  --prompt-ids kp0,kp1,kp2,kp3,kp4 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_style27_prompt_sweep \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 6 \
  --min-free-gb 40
```

真实 prompt sweep 已完成：

- style: `27`
- prompts: `kp0,kp1,kp2,kp3,kp4`
- timesteps per prompt: `718.3220825195312, 472.07550048828125, 363.1470031738281`
- total teacher-to-OT observations: `15`

全局 ranking：

```text
rank,teacher,score,n,cosine_mean,cosine_std,diff_norm_mean,norm_ratio_mean,diff_hi_freq_ratio_mean
1,teacher_hp_lam1,0.8851675220239321,15,0.9775532086690267,0.0015849908835707217,89.22043863932292,1.0332472006479898,0.3272808209061623
2,teacher_hp_lam0p75,0.8479565872183553,15,0.9690542658170064,0.0023116384574158383,108.38291778564454,1.1036301056543987,0.4187419613202413
3,teacher_hp_lam0p5,0.7693992687605662,15,0.9390903353691101,0.003977058474861757,150.19454549153645,1.1602550824483235,0.5721665501594544
4,teacher_hp_lam0p25,0.659537273740586,15,0.8831613699595133,0.00420814753546617,200.9781290690104,1.193259859085083,0.6784009595712026
5,teacher_hp_lam0,0.5237710618761726,15,0.8013521909713746,0.008471545504678017,255.4374552408854,1.1950959761937459,0.7435821056365967
```

各 prompt 局部 best：

```text
kp0 teacher_hp_lam1 score=0.888884 cosine=0.978996 diff=88.088
kp1 teacher_hp_lam1 score=0.882311 cosine=0.976184 diff=89.357
kp2 teacher_hp_lam1 score=0.878017 cosine=0.975649 diff=91.706
kp3 teacher_hp_lam1 score=0.891510 cosine=0.979349 diff=86.630
kp4 teacher_hp_lam1 score=0.885525 cosine=0.977589 diff=90.322
```

跨 prompt 结论：

- `teacher_hp_lam1` 在 5/5 prompts 上都是局部最优。
- 全局 15 个观测点上 `teacher_hp_lam1` 仍是最优。
- `lam1` 的 cosine std 只有 `0.00158`，说明在这组 content prompt 上稳定。
- 这把 `hard + highpass(ot-hard)` 从单样本假设推进到跨 prompt 稳定假设；下一步应做 style sweep，验证是否跨 style 稳定。

### style sweep

已新增：

```text
scripts/run_style_sweep.py
```

用途：对多个 style 串行运行 `run_prompt_sweep.py`，合并所有 style/prompt/timestep 的 pairwise 结果，再全局调用 `select_teacher_lambda.py`。

dry-run 验证命令：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_style_sweep.py \
  --probe-mode dry-run \
  --style-ids 00,27 \
  --prompt-ids kp0,kp1 \
  --selected-timesteps 700 \
  --out-prefix residual_profile_style_sweep_dryrun \
  --teacher-lambdas 0,0.5,1 \
  --height 32 \
  --width 32
```

真实小规模 style sweep 命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_style_sweep.py \
  --probe-mode run-model \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp2,kp4 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_style_sweep_small \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 6 \
  --min-free-gb 40
```

真实 style sweep 已完成：

- styles: `00,13,27`
- prompts per style: `kp0,kp2,kp4`
- timesteps per prompt: `718.3220825195312, 472.07550048828125, 363.1470031738281`
- total teacher-to-OT observations: `27`

全局 ranking：

```text
rank,teacher,score,n,cosine_mean,cosine_std,diff_norm_mean,norm_ratio_mean,diff_hi_freq_ratio_mean
1,teacher_hp_lam1,0.8877497263226992,27,0.9790611774833115,0.0029930032920738813,89.12346733940973,1.0281123960459675,0.32172606692269995
2,teacher_hp_lam0p75,0.8513957239717257,27,0.9708524147669474,0.00265143416893443,108.69489683928313,1.0924623608589172,0.40967691993271865
3,teacher_hp_lam0p5,0.7751878361244611,27,0.9427478379673428,0.004040110314941863,150.83977819372106,1.143836001555125,0.5613723893960317
4,teacher_hp_lam0p25,0.6692674397014038,27,0.8908485924756085,0.008290844679770619,201.86776394314236,1.1737136156470687,0.668843126959271
5,teacher_hp_lam0,0.5390290112471272,27,0.8151497907108731,0.01868887678614552,256.53396662959346,1.1754831782093755,0.7353424120832373
```

各 style 局部 best：

```text
style 00 teacher_hp_lam1 score=0.887757 cosine=0.979639 diff=91.965 n=9
style 13 teacher_hp_lam1 score=0.891641 cosine=0.980133 diff=85.367 n=9
style 27 teacher_hp_lam1 score=0.884053 cosine=0.977411 diff=90.038 n=9
```

跨 style 结论：

- `teacher_hp_lam1` 在 3/3 styles 上都是局部最优。
- 全局 27 个观测点上 `teacher_hp_lam1` 仍是最优。
- `lam1` 的全局 cosine mean 是 `0.9791`，cosine std 是 `0.0030`，跨 style/prompt/timestep 仍稳定。
- hard baseline (`lam0`) 的 cosine mean 只有 `0.8151`，diff norm 为 `256.53`；`lam1` 将 diff norm 降到 `89.12`。
- 当前证据支持把 `hard_scfg + highpass(otcfg - hard_scfg)` 作为第一版 distillation teacher。

### distillation manifest

已新增：

```text
scripts/build_distill_manifest.py
```

用途：把真实 residual field dumps 汇总成训练数据 manifest。训练代码后续只需要读 JSONL，按每行的 `dump_path` 加载 `hard_scfg/otcfg`，再按 `teacher` 字段构造目标 residual。

生成命令：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/build_distill_manifest.py \
  --field-dumps "results/field_dumps/residual_profile_style_sweep_small_*.pt" \
  --out-prefix residual_distill_style_sweep_small_lam1 \
  --teacher-kind highpass_delta \
  --teacher-lambda 1.0 \
  --teacher-highpass-kernel 5 \
  --ranking-json results/residual_profile_style_sweep_small_global_teacher_ranking.json \
  --verify-teacher
```

输出：

- `results/residual_distill_style_sweep_small_lam1.jsonl`
- `results/residual_distill_style_sweep_small_lam1_summary.json`

manifest summary：

```text
num_samples = 27
teacher = teacher_hp_lam1
teacher formula = hard_scfg + lambda * highpass(otcfg - hard_scfg)
lambda = 1.0
highpass_kernel = 5
field_shape = [1, 1024, 128]
styles = 00:9, 13:9, 27:9
prompts = kp0:9, kp2:9, kp4:9
timesteps = 363.147003:9, 472.075500:9, 718.322083:9
```

ranking evidence embedded in summary:

```text
best teacher = teacher_hp_lam1
score = 0.8877497263226992
n = 27
cosine_mean = 0.9790611774833115
diff_norm_mean = 89.12346733940973
norm_ratio_mean = 1.0281123960459675
```

训练入口：

- input: `hard_scfg`
- optional reference/upper bound: `otcfg`
- target teacher: `hard_scfg + highpass(otcfg - hard_scfg)`
- dataset manifest: `results/residual_distill_style_sweep_small_lam1.jsonl`
- script: `scripts/train_residual_adapter.py`
- first checkpoint: `results/residual_adapter_smoke_long/adapter_best.pt`

真实 Klein9B teacher sweep 命令已准备好：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 43 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_klein9b_teacher_sweep \
  --no-hyperion-grad \
  --sinkhorn-iters 6 \
  --teacher-lambdas 0,0.25,0.5,0.75,1.0 \
  --teacher-highpass-kernel 5
```

当前真实 sweep 未完成，原因是 GPU0 被外部进程占用：

```text
3817845 /mnt/local/chengyanli/miniconda3/envs/firered/firered/bin/python 60864 MiB
2486341 /home/ubuntu/chengyanli/flux/bin/python                         60758 MiB
```

已尝试：

- `--vram-limit-gb 18`
- `--vram-limit-gb 0`

二者均在 transformer 权重加载阶段 OOM。不是 highpass teacher 代码错误；失败发生在进入 probe forward 之前。GPU 释放后直接重跑上面的真实 sweep 命令即可。

## 不变量

真实 residual probe 必须保持：

- `v_normal` 与 `v_styled` 在同一个 `(z_t, t)` 上计算。
- 主路径每个 timestep 最多调用一次 `scheduler.step(...)`。
- profile 阶段不 decode 图像，不引入多步 latent optimization。

## 下一步

- 将 adapter 明确降级为 baseline / ablation，不再作为第二篇主方法。
- 主线转为 direct score/velocity residual distillation：`f_theta(z_t, t, prompt, style) -> Delta v_style`。
- 真实数据重新 dump 时使用 `--dump-fields --dump-latents`，保证每个 teacher residual 都有同一点的 `z_t`。
- 后续 manifest 必须拒绝没有 `latents.z_t` 的样本；旧 residual-only dumps 只用于 adapter baseline。
- 继续保持不变量：`v_normal` / `v_hard` / `v_ot` / `student_delta` 必须在同一个 `(z_t, t)` 上计算，主路径每个 timestep 仍只调用一次 scheduler step。

## Score residual distillation pivot

结论：adapter 不是理论必需。它只是因为早期 dumps 只有 `hard_scfg` / `otcfg` / teacher residual，没有保存 `z_t`，所以工程上只能训练：

```text
pred_teacher = hard_scfg + adapter(hard_scfg, timestep, style_id, prompt_id)
```

在真实数据充足的情况下，第二篇主线应训练更强的 score/velocity residual student：

```text
teacher residual:
Delta v* = hard_scfg + highpass(otcfg - hard_scfg)

student:
f_theta(z_t, t, prompt, style) ~= Delta v*

inference:
v = v_normal + alpha * f_theta(z_t, t, prompt, style)
```

配套设计文档：`docs/score_residual_distillation_plan.md`。

`scripts/probe_klein9b_residuals.py` 已新增 `--dump-latents`。和 `--dump-fields` 一起使用时，每个 `.pt` dump 会额外保存：

```text
latents.z_t
latents.z_t_shape
latents.z_t_dtype
has_latents = true
```

因此后续应新增 direct residual manifest/training 脚本，数据 schema 至少包含：

```text
z_t, timestep, prompt_id/prompt_text, style_id/style_path,
fields.hard_scfg, fields.otcfg, fields.teacher_hp_lam1
```

`scripts/build_score_residual_manifest.py` 已作为 direct score residual 专用 manifest 入口；它默认拒绝没有 `latents.z_t` 的旧 dumps。

## Direct score residual student smoke

新增：

- `src/score_residual_student.py`
- `scripts/train_score_residual_student.py`

模型定义：

```text
pred_residual = f_theta(z_t, timestep, style_id, prompt_id)
v = v_normal + alpha * pred_residual
```

当前 student 是 token MLP + timestep/style/prompt conditioning + learned token position embedding。position embedding 必须保留；没有空间位置时，token MLP 很难表达风格 residual 的空间场。

### 单真实样本 overfit

数据：

```text
dump = results/field_dumps/residual_profile_real_dump_latents_check_kp0_27_t717.pt
manifest = results/score_residual_manifest_real_check.jsonl
z_t = [1, 1024, 128], float16
target = teacher_hp_lam1
```

训练：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_real_check.jsonl \
  --out-dir results/score_residual_student_real_overfit_capacity \
  --epochs 200 \
  --batch-size 1 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.005 \
  --weight-decay 0 \
  --val-fraction 0 \
  --device cuda
```

结果：

```text
epoch 25 train_mse = 0.466502, hard_baseline_mse = 0.556178
epoch 50 train_mse = 0.046623
best_epoch = 163
best_metric ~= 0
```

结论：direct `z_t -> teacher residual` 链路可以拟合真实 Klein9B residual field。

### 三 timestep 留一验证

采样命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/probe_klein9b_residuals.py \
  --run-model \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 44 \
  --runtime-height 512 \
  --runtime-width 512 \
  --num-steps 12 \
  --selected-timesteps 700,500,300 \
  --out-prefix residual_profile_real_dump_latents_3step_style27_kp0 \
  --results-dir results \
  --no-hyperion-grad \
  --sinkhorn-iters 3 \
  --teacher-lambdas 1.0 \
  --teacher-highpass-kernel 5 \
  --dump-fields \
  --dump-latents
```

实际 timesteps：

```text
705.765625
590.027954
395.469971
```

manifest：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/build_score_residual_manifest.py \
  --field-dumps "results/field_dumps/residual_profile_real_dump_latents_3step_style27_kp0_*.pt" \
  --out-prefix score_residual_manifest_real_3step_style27_kp0 \
  --results-dir results \
  --teacher-kind highpass_delta \
  --teacher-lambda 1.0 \
  --teacher-highpass-kernel 5 \
  --verify-teacher
```

训练：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_real_3step_style27_kp0.jsonl \
  --out-dir results/score_residual_student_real_3step_style27_kp0 \
  --epochs 200 \
  --batch-size 1 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.003 \
  --weight-decay 0 \
  --val-fraction 0.34 \
  --device cuda
```

结果：

```text
best_epoch = 48
best_val_mse = 0.780175
final_train_mse = 0.010613
final_train_hard_baseline_mse = 0.456898
final_val_mse = 1.339989
final_val_hard_baseline_mse = 0.583576
final_val_cosine = 0.673273
final_val_hard_baseline_cosine = 0.826521
```

结论：

- direct student 能拟合训练 timestep。
- 3-sample 数据下跨 timestep 泛化不足，验证集仍弱于 hard CFG baseline。
- 后续优先扩真实数据和提升结构容量；adapter 继续只作为 baseline。

## 45-sample direct CFG correction

为验证“CFG base + learned score correction”是否比从零预测 residual 更稳，重新采集了 45 个带 `z_t` 的真实 Klein9B field dumps。

采样入口：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_style_sweep.py \
  --probe-mode run-model \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp2,kp3,kp4 \
  --seed 45 \
  --selected-timesteps 700,500,300 \
  --out-prefix score_residual_latent45_lam1 \
  --results-dir results \
  --teacher-lambdas 1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 3 \
  --sinkhorn-tau 0.02 \
  --runtime-height 512 \
  --runtime-width 512 \
  --min-free-gb 0 \
  --force \
  --dump-latents
```

sweep 透传改动：

- `scripts/run_teacher_pipeline.py` 支持 `--dump-latents`
- `scripts/run_prompt_sweep.py` 支持 `--dump-latents`
- `scripts/run_style_sweep.py` 支持 `--dump-latents`

manifest：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/build_score_residual_manifest.py \
  --field-dumps "results/field_dumps/score_residual_latent45_lam1_style*_kp*_probe_*.pt" \
  --out-prefix score_residual_manifest_latent45_lam1 \
  --results-dir results \
  --teacher-kind highpass_delta \
  --teacher-lambda 1.0 \
  --teacher-highpass-kernel 5 \
  --verify-teacher
```

manifest summary：

```text
num_samples = 45
styles = 00:15, 13:15, 27:15
prompts = kp0:9, kp1:9, kp2:9, kp3:9, kp4:9
timesteps = 363.147003:15, 472.075500:15, 718.322083:15
latent_dtype = float16
field_shape = [1, 1024, 128]
target = teacher_hp_lam1
```

### Plain direct residual

目标：

```text
pred = f_theta(z_t, t, prompt, style)
```

训练：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_latent45_lam1.jsonl \
  --out-dir results/score_residual_student_latent45_lam1_plain \
  --epochs 100 \
  --batch-size 3 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.001 \
  --weight-decay 0 \
  --val-fraction 0.2 \
  --device cuda
```

best epoch：

```text
best_epoch = 92
val_mse = 0.547354
val_hard_baseline_mse = 0.418083
val_hard_mse_improvement = -0.129271
val_cosine = 0.778109
val_hard_baseline_cosine = 0.838770
```

结论：从 `z_t` 直接预测完整 style residual 在当前数据/结构下弱于 hard CFG。

### Hard CFG correction

目标：

```text
pred = hard_scfg + f_theta(z_t, t, prompt, style)
```

训练：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_latent45_lam1.jsonl \
  --out-dir results/score_residual_student_latent45_lam1_hard_delta \
  --epochs 100 \
  --batch-size 3 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.001 \
  --weight-decay 0 \
  --val-fraction 0.2 \
  --device cuda \
  --prediction-mode hard_delta
```

best epoch：

```text
best_epoch = 12
val_mse = 0.390818
val_hard_baseline_mse = 0.418083
val_hard_mse_improvement = 0.027265
val_cosine = 0.854317
val_hard_baseline_cosine = 0.838770
train_mse = 0.369235
train_hard_baseline_mse = 0.423738
```

结论：

- `hard_scfg + f_theta(z_t,t,prompt,style)` 已经在 45-sample validation split 上优于 hard CFG baseline。
- 这个结果比 plain direct residual 更符合 CFG/score function 推导。
- 后续主线应表述为 learned score correction，而不是 adapter，也不是从零预测完整 residual。

当前方法公式：

```text
Delta v_teacher = hard_scfg + highpass(otcfg - hard_scfg)
Delta v_student = hard_scfg + f_theta(z_t, t, prompt, style)
v = v_normal + alpha * Delta v_student
```

## Group holdout validation for hard CFG correction

`scripts/train_score_residual_student.py` 已新增：

```text
--split-mode random|style|prompt|timestep
--holdout-value VALUE
```

目的：验证 `hard_scfg + f_theta(z_t,t,prompt,style)` 的提升不是随机 split 里相近样本泄漏造成的。

运行命令：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_latent45_lam1.jsonl \
  --out-dir results/score_residual_student_latent45_lam1_hard_delta_holdout_prompt_kp4 \
  --epochs 100 \
  --batch-size 3 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.001 \
  --weight-decay 0 \
  --split-mode prompt \
  --holdout-value kp4 \
  --device cuda \
  --prediction-mode hard_delta
```

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_latent45_lam1.jsonl \
  --out-dir results/score_residual_student_latent45_lam1_hard_delta_holdout_style27 \
  --epochs 100 \
  --batch-size 3 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.001 \
  --weight-decay 0 \
  --split-mode style \
  --holdout-value 27 \
  --device cuda \
  --prediction-mode hard_delta
```

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_latent45_lam1.jsonl \
  --out-dir results/score_residual_student_latent45_lam1_hard_delta_holdout_t363 \
  --epochs 100 \
  --batch-size 3 \
  --hidden-dim 1024 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --lr 0.001 \
  --weight-decay 0 \
  --split-mode timestep \
  --holdout-value 363 \
  --device cuda \
  --prediction-mode hard_delta
```

汇总文件：

```text
results/score_residual_student_latent45_hard_delta_holdout_summary.json
```

best epoch 指标：

```text
random split:
  train/val = 36/9
  best_epoch = 12
  val_mse = 0.390818
  hard_baseline_mse = 0.418083
  mse_improvement = 0.027265
  val_cosine = 0.854317
  hard_baseline_cosine = 0.838770

holdout prompt kp4:
  train/val = 36/9
  best_epoch = 7
  val_mse = 0.397504
  hard_baseline_mse = 0.420769
  mse_improvement = 0.023265
  val_cosine = 0.848027
  hard_baseline_cosine = 0.834696

holdout style 27:
  train/val = 30/15
  best_epoch = 6
  val_mse = 0.410652
  hard_baseline_mse = 0.436566
  mse_improvement = 0.025914
  val_cosine = 0.832427
  hard_baseline_cosine = 0.817265

holdout timestep 363.147003:
  train/val = 30/15
  best_epoch = 7
  val_mse = 0.307673
  hard_baseline_mse = 0.334272
  mse_improvement = 0.026599
  val_cosine = 0.843585
  hard_baseline_cosine = 0.827962
```

结论：

- prompt/style/timestep 三种 group holdout 全部优于 hard CFG baseline。
- 这比单纯 random split 更有说服力，说明 learned correction 不是只在记忆邻近样本。
- 提升幅度仍是小幅但稳定；下一步应升级 correction student 结构或接 DiT/LoRA residual head。

## Score student image sampler

新增：

- `src/score_student_runtime.py`
- `scripts/sample_student_images.py` 支持 `mode=score_student`

运行公式：

```text
hard_scfg = v_hard(z_t,t) - v_normal(z_t,t)
score_delta = f_theta(z_t,t,prompt,style)
v_score_student = v_hard + score_student_scale * score_delta
```

保持不变量：

- 每个 branch 自己的 `v_normal / v_hard / score_delta` 在同一个 `(z_t,t)` 上计算。
- 每个 branch 每个 timestep 只调用一次 `scheduler.step(...)`。
- 旧 `student` mode 仍是 adapter；新 `score_student` mode 是 direct CFG correction student。

runtime smoke：

```text
checkpoint = results/score_residual_student_latent45_lam1_hard_delta/student_best.pt
input z_t = [1, 1024, 128]
output correction = [1, 1024, 128]
finite = true
norm = 88.341965
```

### Image smoke, inject_beta = 1.0

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sample_student_images.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 46 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_image_style27_kp0_s1_steps12 \
  --results-dir results \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --score-student-scale 1.0 \
  --modes normal,hard,score_student
```

输出：

```text
results/student_samples/score_student_image_style27_kp0_s1_steps12_normal.png
results/student_samples/score_student_image_style27_kp0_s1_steps12_hard.png
results/student_samples/score_student_image_style27_kp0_s1_steps12_score_student.png
results/student_samples/score_student_image_style27_kp0_s1_steps12_contact.png
results/score_student_image_style27_kp0_s1_steps12_meta.json
```

统计：

```text
normal mean/std = [97.53,100.11,75.46] / [75.68,73.30,71.27]
hard mean/std = [199.68,142.14,57.34] / [55.15,51.52,39.18]
score_student mean/std = [199.33,144.75,52.89] / [48.59,45.36,37.60]
score_delta / hard_residual norm ratio mean = 0.2526
```

观察：`inject_beta=1.0` 下 hard CFG 太强，panda prompt/content 被 style reference content 覆盖；score_student 只是小 correction，不能单独修复 base CFG 的 content leak。

### Image smoke, inject_beta = 0.5

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sample_student_images.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 46 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_image_style27_kp0_s1_beta0p5_steps12 \
  --results-dir results \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --score-student-scale 1.0 \
  --inject-beta 0.5 \
  --modes normal,hard,score_student
```

输出：

```text
results/student_samples/score_student_image_style27_kp0_s1_beta0p5_steps12_normal.png
results/student_samples/score_student_image_style27_kp0_s1_beta0p5_steps12_hard.png
results/student_samples/score_student_image_style27_kp0_s1_beta0p5_steps12_score_student.png
results/student_samples/score_student_image_style27_kp0_s1_beta0p5_steps12_contact.png
results/score_student_image_style27_kp0_s1_beta0p5_steps12_meta.json
```

统计：

```text
normal mean/std = [97.53,100.11,75.46] / [75.68,73.30,71.27]
hard mean/std = [128.68,114.40,75.85] / [73.20,67.59,56.65]
score_student mean/std = [126.87,115.81,73.55] / [66.38,61.19,52.59]
score_delta / hard_residual norm ratio mean = 0.3756
```

观察：`inject_beta=0.5` 明显更保 content，panda 仍然存在，同时有可见 style shift。

结论：

- latent 层面 `hard_delta` 已经证明 correction 有效。
- 图像层面 base CFG 强度是关键控制量；`inject_beta=1.0` 容易 content leak，`0.5` 更合理。
- 下一步图像优化应扫 `inject_beta x score_student_scale`，而不是只扫 correction scale。

## Score student image grid

新增：

- `scripts/run_score_image_grid.py`

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_grid.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 46 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_grid_style27_kp0 \
  --results-dir results \
  --betas 0.35,0.5,0.65 \
  --scales 0.5,1.0,1.5 \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_grid_style27_kp0_summary.json
results/student_samples/score_student_grid_style27_kp0_contact.png
```

统计摘要：

```text
beta=0.35 scale=0.5 score_mean=[111.85,106.37,74.13] hard_mean=[112.58,106.17,75.54] ratio=0.5475
beta=0.35 scale=1.0 score_mean=[111.36,106.71,73.14] hard_mean=[112.58,106.17,75.54] ratio=0.5336
beta=0.35 scale=1.5 score_mean=[110.85,107.40,72.30] hard_mean=[112.58,106.17,75.54] ratio=0.5218

beta=0.5 scale=0.5 score_mean=[127.93,115.14,74.62] hard_mean=[128.68,114.40,75.85] ratio=0.3849
beta=0.5 scale=1.0 score_mean=[126.87,115.81,73.55] hard_mean=[128.68,114.40,75.85] ratio=0.3756
beta=0.5 scale=1.5 score_mean=[125.92,116.32,72.49] hard_mean=[128.68,114.40,75.85] ratio=0.3676

beta=0.65 scale=0.5 score_mean=[179.01,140.03,62.82] hard_mean=[178.84,138.04,66.08] ratio=0.2985
beta=0.65 scale=1.0 score_mean=[178.33,141.64,60.42] hard_mean=[178.84,138.04,66.08] ratio=0.2928
beta=0.65 scale=1.5 score_mean=[176.37,142.29,57.44] hard_mean=[178.84,138.04,66.08] ratio=0.2871
```

视觉结论：

- `inject_beta=0.35`：最保 content，但 style 较轻。
- `inject_beta=0.5`：当前最好 tradeoff；panda content 清楚，style shift 明显。
- `inject_beta=0.65`：明显 content leak，开始复制 style reference 的房子/背景。
- `score_student_scale` 在 0.5 到 1.5 内主要影响局部纹理/色调强度，远弱于 `inject_beta` 对 content/style 的影响。

当前推荐：

```text
default image setting:
inject_beta = 0.5
score_student_scale = 1.0

content-conservative setting:
inject_beta = 0.35
score_student_scale = 1.0 or 1.5
```

### Latent120 grid runner update

`scripts/run_score_image_grid.py` 已适配当前 direct-score 主线：

- 默认 checkpoint 改为
  `results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt`
- 支持 `--score-student-style-scales`
- 支持 `--score-student-timestep-min/max`
- summary 额外记录：
  `score_student_effective_scale`, `scaled_ratio_*`, `step_scale_mean`,
  `active_step_count`, `hard_vs_normal_diff`, `score_vs_hard_diff`,
  `score_vs_normal_diff`

smoke：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_grid.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 53 \
  --height 512 \
  --width 512 \
  --num-steps 8 \
  --out-prefix score_student_grid_latent120_smoke_s53 \
  --results-dir results \
  --betas 0.45,0.55 \
  --scales 0.75,1.0 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_grid_latent120_smoke_s53_summary.json
results/student_samples/score_student_grid_latent120_smoke_s53_contact.png
```

关键字段：

```text
checkpoint = results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt
score_student_style_scales = {}
score_student_timestep_min/max = null/null
```

summary：

```text
beta=0.45 scale=0.75 ratio=0.4193 scaled=0.3145 active=8 score_vs_hard_l1=7.310 hard_vs_normal_l1=40.210
beta=0.45 scale=1.00 ratio=0.4148 scaled=0.4148 active=8 score_vs_hard_l1=9.301 hard_vs_normal_l1=40.210
beta=0.55 scale=0.75 ratio=0.3524 scaled=0.2643 active=8 score_vs_hard_l1=6.632 hard_vs_normal_l1=51.065
beta=0.55 scale=1.00 ratio=0.3484 scaled=0.3484 active=8 score_vs_hard_l1=8.496 hard_vs_normal_l1=51.065
```

视觉观察：

- 4 个 smoke sample 都保住 panda content。
- `beta=0.55` 比 `0.45` 风格更强。
- `scale=1.0` 比 `0.75` 的 learned correction 更强，但这组没有明显改变构图。

用途：

- 后续正式扫参直接用这个 updated grid runner。
- 先扫 `beta=0.45,0.5,0.55,0.6` 与 `scale=0.75,1.0,1.25`，再用 batch
  验证候选点。

### Latent120 beta refinement

正式 refined grid：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_grid.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 54 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_grid_latent120_beta_refine_style27_kp0_s54 \
  --results-dir results \
  --betas 0.45,0.5,0.55,0.6 \
  --scales 0.75,1.0,1.25 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_grid_latent120_beta_refine_style27_kp0_s54_summary.json
results/student_samples/score_student_grid_latent120_beta_refine_style27_kp0_s54_contact.png
```

表格：

```text
beta scale ratio  scaled score_hard_l1 hard_normal_l1 score_normal_l1
0.45 0.75  0.4538 0.3404  9.089 43.510 43.830
0.45 1.00  0.4467 0.4467 12.050 43.510 44.339
0.45 1.25  0.4390 0.5488 14.767 43.510 45.041
0.50 0.75  0.4075 0.3056 10.725 48.774 49.343
0.50 1.00  0.4001 0.4001 13.708 48.774 49.882
0.50 1.25  0.3921 0.4901 17.081 48.774 50.708
0.55 0.75  0.3678 0.2758  8.556 57.498 58.409
0.55 1.00  0.3618 0.3618 11.160 57.498 58.714
0.55 1.25  0.3563 0.4454 13.428 57.498 58.732
0.60 0.75  0.3362 0.2521  8.615 64.986 65.854
0.60 1.00  0.3305 0.3305 11.199 64.986 66.147
0.60 1.25  0.3255 0.4069 13.360 64.986 66.353
```

观察：

- `beta=0.60` 对 style27/kp0 过强，panda face/background 明显被 style
  reference 主导。
- `beta=0.55` 风格更强，但 content risk 上升。
- `beta=0.45` 是这一组里最稳的 content-preserving row；配 `scale=1.25`
  可以把 learned correction 拉强，而不明显改变构图。

候选 batch：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 55 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_latent120_beta0p45_scale1p25_s55 \
  --results-dir results \
  --inject-beta 0.45 \
  --score-student-scale 1.25 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_batch_latent120_beta0p45_scale1p25_s55_summary.json
results/student_samples/score_student_batch_latent120_beta0p45_scale1p25_s55_contact.png
```

summary：

```text
scaled_ratio_mean avg = 0.472435
scaled_ratio_mean range = 0.355063 to 0.655053
active_step_count = 12
```

视觉结论：

- 旧 styles `00,13,27` x prompts `kp0,kp1,kp3` 的 9 个组合都保住主要
  prompt content。
- score_student 相比 hard CFG 有更明显的颜色/纹理 refinement。
- 当前把它记为候选配置，不直接替代默认：

```text
default               = inject_beta=0.50, score_student_scale=1.00
conservative candidate= inject_beta=0.45, score_student_scale=1.25
historical status     = stable on old and new cross-seed batches
current status        = superseded by the strict paired comparison below
```

新风格候选验证已完成：

```text
styles  = 02,05,08,22,34
prompts = kp0,kp1,kp3
seed    = 56
steps   = 12

outputs = results/score_student_batch_latent120_beta0p45_scale1p25_newstyles_s56_summary.json
          results/student_samples/score_student_batch_latent120_beta0p45_scale1p25_newstyles_s56_contact.png

ratio_mean avg          = 0.358307
scaled_ratio_mean avg   = 0.447884
scaled_ratio_mean range = 0.327278 to 0.644651
active_step_count       = 12
```

per-style `scaled_ratio_mean`：

```text
02 = 0.473969
05 = 0.407075
08 = 0.562728
22 = 0.364128
34 = 0.431518
```

视觉检查：15 个组合都保住 panda / castle / sunflower 主体。style08 的
correction 最强，但没有明显 reference layout takeover；style22 最弱，仍有可见
风格变化。因此 `beta=0.45, scale=1.25` 可以升级为 robust candidate，但还不能
直接替代默认值：下一步必须用相同 seed 对 `0.50/1.00` 和 `0.45/1.25` 做成对
比较，避免把 seed 差异误判成参数收益。

### Cross-seed normalized image-difference audit

在同 seed default batch 等待显存期间，先对已有四批图计算归一化差异：

```text
refine_fraction = L1(score_student, hard) / L1(hard, normal)
extension_fraction = [L1(score_student, normal) - L1(hard, normal)]
                     / L1(hard, normal)

                         refine mean  refine median  refine std  extension mean
default old, seed48        0.216747      0.214812      0.038697       0.000053
candidate old, seed55      0.318217      0.315683      0.036526      -0.001441
default new, seed49        0.267347      0.239298      0.103885      -0.040083
candidate new, seed56      0.285010      0.277664      0.070898      -0.018979
```

new-style range：

```text
default   = 0.162547 to 0.581006
candidate = 0.171076 to 0.402359
```

解释：旧风格 candidate 的相对 refinement 比 default 高约 `46.8%`，但
`extension_fraction` 仍接近 0，没有表现成单纯把图整体推得更远。新风格的均值
只小幅增加，但标准差与最大 outlier 都下降，说明 candidate 更稳定。由于四批
seed 不同，这些只能作为诊断证据，不能替代最终的 same-seed paired comparison。

已尝试 `style27/kp0, seed55` 的 same-seed default smoke；外部任务占用约
113 GiB / 144 GiB 显存，`--vram-limit-gb 20` 仍在 transformer loading 阶段
OOM。没有生成新样本，该失败不涉及采样或 student 逻辑。

### Lossless low-VRAM loading fix

根因已经定位：旧代码虽然把 `--vram-limit-gb` 传给 pipeline，但 text encoder /
transformer 的 `ModelConfig` 没有设置 offload dtype/device，因此 DiffSynth 根本没
启用 VRAM manager，8B text encoder 和 9B transformer 仍同时加载到 CUDA。

已修改：

```text
src/klein9b_loading.py
scripts/sample_student_images.py
scripts/probe_klein9b_residuals.py
```

提供 `--vram-limit-gb` 后使用 lossless BF16 CPU offload：

```text
offload_device=cpu,  offload_dtype=bfloat16
onload_device=cuda,  onload_dtype=bfloat16
preparing_device=cuda, preparing_dtype=bfloat16
computation_device=cuda, computation_dtype=bfloat16
```

没有 FP8 quantization。meta 新增 `model_loading_mode=cpu_offload_bf16` 和
`vram_limit_gb`。远端 `py_compile`、probe dry-run 和 12-step real sample 均
通过；在只有约 28 GiB GPU headroom 时完整生成 normal/hard/score_student。

### Strict same-seed three-config comparison

统一 `kp0`、12 steps、`cpu_offload_bf16`：

```text
default      = beta0.50 / scale1.00
conservative = beta0.45 / scale1.25
balanced     = beta0.50 / scale1.25

styles 00,13,27: seed55
styles 02,05,08,22,34: seed56
```

输出：

```text
results/score_student_paired_default_vs_candidate_8styles_kp0_offload16_summary.json
results/score_student_paired_three_config_8styles_kp0_offload16_summary.json
results/student_samples/score_student_paired_three_config_8styles_kp0_offload16_contact.png
```

严格配对有效性：style27 两个独立 offload run 的 normal image 逐像素完全一致，
`L1=L2=max=0`。

聚合结果：

```text
                                     default   conservative   balanced
refine_fraction                       0.209310     0.288188     0.254383
palette histogram distance            32.6389      35.4462      33.1203

conservative vs default:
  carrier magnitude                   -8.87%
  learned refinement                 +26.10%
  final displacement                  -9.09%

balanced vs default:
  learned refinement                 +21.62%
  final displacement                  +0.11%
```

视觉观察：conservative 在 `00/13/02` 这类强 palette 风格上会褪色；balanced
保留 default 的颜色/结构 carrier，同时在 `13/27/02/05/22/34` 上给出更清楚的
线条、纹理和 tone refinement。简单 RGB histogram 受 content 分布影响，只保留
作诊断，不能代替视觉和正式 style metric。

当前决策：

```text
baseline              = beta0.50 / scale1.00
balanced leader       = beta0.50 / scale1.25
conservative fallback = beta0.45 / scale1.25
status = 已由下面的 multi-prompt freeze validation 更新
```

方法上的新结论：`inject_beta` 主要控制 carrier / low-frequency palette，
`score_student_scale` 控制 learned teacher correction。两个旋钮应独立校准；把
beta 随 student scale 一起反向降低，会丢掉有用的 carrier 信息。

### Balanced multi-prompt freeze validation

继续在 `kp1` 和 `kp3` 上做了相同的八风格 strict same-seed paired test。连同
前面的 `kp0`，冻结配置所依据的验证集为：

```text
styles  = 00,13,27,02,05,08,22,34
prompts = kp0,kp1,kp3
steps   = 12

baseline = beta0.50 / scale1.00
balanced = beta0.50 / scale1.25
```

这些任务使用 BF16 CPU offload，是因为当时 H200 上的其他任务占用了约 113 GiB，
只剩约 28 GiB 空闲；这不是方法的显存限制。GPU 空闲时可以不传
`--vram-limit-gb`，直接使用正常加载路径。

输出：

```text
results/score_student_balanced_prompt_validation_24pairs_offload16_summary.json
results/student_samples/score_student_balanced_promptval_kp1_8styles_offload16_contact.png
results/student_samples/score_student_balanced_promptval_kp3_8styles_offload16_contact.png
```

24 组聚合结果：

```text
default refine_fraction                 0.217179
balanced refine_fraction                0.262665
learned refinement change               +21.03%
final displacement from normal change    -0.10%
refinement wins                          24 / 24
active student steps                     每组均为 12 / 12
```

按 prompt 分解，learned refinement 在 `kp0/kp1/kp3` 分别提高 `+21.62%`、
`+20.13%`、`+21.34%`，每个 prompt 都是 `8/8` wins；八个 style 各自也都是跨
prompt `3/3` wins。按 style 聚合后，final displacement 的变化约落在
`-0.92%` 到 `+0.62%`，仍接近零。

视觉上，castle 与 sunflower 的主体和空间构图均保持稳定，balanced 相比 default
主要增强风格相关的线条、纹理和 tone，没有明显 reference layout takeover 或
content collapse。简单 RGB histogram 只有 `3/24` wins，但它受到内容像素分布
混淆，无法衡量实际可见的 texture/line gain，因此只能作为诊断指标，不能作为
主 style metric。

当前 latent120 图像实验正式冻结为：

```text
recommended/default = beta0.50 / scale1.25
paired baseline      = beta0.50 / scale1.00
fallback             = beta0.45 / scale1.25
```

边界：这是当前 `8 styles x 3 prompts` 实验的 operating point，不代表 `1.25` 是
所有 reference/style 的普适最优值。论文中应保留 paired scale sweep，并把
reference-conditioned 或 timestep-conditioned calibration 作为后续扩展。

### Reference-information onset 与 prompt collision（2026-07-13）

按照 reference-conditioned 主方法规格，新增了两个同状态审计。所有比较均使用
Klein9B、12 steps、`inject_beta=0.5`、`sinkhorn_iters=3`、相同 seed，target 为：

```text
r* = teacher_hp_lam1 - hard_scfg
   = H_grid(otcfg - hard_scfg)
```

#### 1. `t=1000` prompt collision

固定 `style27`、seed61 和完全相同的初始 `z_T`，只替换
`kp0/kp1/kp3`。三个 dump 的 `z_T` 逐元素一致：

```text
pairwise z_T max_abs = 0
```

但 teacher correction 明显不同：

```text
mean pairwise cosine       = 0.1483
mean pairwise relative L2  = 1.3307
between-prompt energy frac = 57.89%
correction norm kp0/kp1/kp3 = 164.40 / 102.56 / 111.19
```

输出：

```text
results/prompt_collision_tmax_style27_seed61_summary.json
```

结论：删除 `prompt_id` 是正确的，但主 student 不能删除真实 text/content
condition。`R(z_T,t,C_ref)` 在最高噪声处存在不可解的 prompt collision；V1 至少
需要 frozen text embedding，V2 更适合直接使用 hard/content Q。

#### 2. Reference-information onset

固定 `kp0`、seed62 和相同 content trajectory，比较 reference `00/13/27`。
所有 timestep 上不同 reference 的 `z_t` 均逐元素一致。reference 对 teacher
correction 的影响为：

```text
timestep   ref energy frac   pair cosine   relative L2
1000.000       0.00%           1.0000        0.0000
 909.701      38.39%           0.4246        1.0728
 782.513      30.46%           0.5438        0.9559
 705.766      29.34%           0.5608        0.9386
 590.028      28.72%           0.5709        0.9290
 395.470      27.36%           0.5914        0.9069
```

输出：

```text
results/reference_information_onset_styles00_13_27_kp0_seed62.csv
results/reference_information_onset_styles00_13_27_kp0_seed62_summary.json
```

`t=1000` 时 noised style latent 是纯噪声，三个 reference 的 teacher correction
逐元素完全相同；到 `t≈910` reference identity 立即成为强信号。这给出一个可测的
reference-information onset，而不是任意经验 timestep window。

#### 3. 首步 guidance 成图消融

`sample_student_images.py` 新增：

```text
--reference-guidance-timestep-min
--reference-guidance-timestep-max
```

窗口外 hard/student branch 使用同 branch `v_normal`，仍严格每步只调用一次
`scheduler.step`。默认 `None` 保持旧行为。

在 style `00/27`、`kp0`、seed63 上比较：

```text
default:        首步 hard + student 均启用
student-skip:   只在首步关闭 learned correction，保留 hard carrier
all-skip:       首步 hard 与 learned correction 都关闭
```

严格复现检查：独立 run 的 normal 图逐像素一致；student-skip 的 hard 图与
default hard 逐像素一致。

观察：

- student-skip 对最终 score image 影响很小，L1 仅约 `1.13/255`（style00）和
  `1.60/255`（style27），没有稳定 style/content 收益；
- all-skip 影响明显，score image L1 约 `21.69/255` 和 `18.29/255`；
- all-skip 让结果更接近 normal，style27 的 painterly 强度也略降；
- 因此 `t=1000` 的 hard correction 虽不含 reference identity，仍表现为
  prompt/content-dependent generic carrier，不能与 reference goal 混为一谈。

输出：

```text
results/reference_guidance_skip_first_styles00_27_kp0_seed63_summary.json
results/student_correction_skip_first_styles00_27_kp0_seed63_summary.json
results/student_samples/reference_guidance_skip_first_styles00_27_kp0_seed63_contact.jpg
```

当前决策：不把 all-skip 冻结为默认。V2 应比较 dynamic-only 与 dual-code：
一次性 canonical `C_goal(x_ref)` 提供从开局可用的视觉目标，当前 timestep 的
`C_dyn(t)` 与 hard Q 提供 carrier 对齐。canonical timestep 必须由更大规模 onset
curve 选择。首步 learned correction 可以置零作为语义干净的 baseline，但当前
style-ID student 上几乎不改变最终结果。

论文故事和可复现性蓝图已单独整理到：

```text
docs/paper_story_goal_carrier_flow.md
```

核心边界：当前 `style_id` student 只证明 score correction 可学；Think with
Image 主方法必须改成真实 reference image code，并按 unseen reference image /
unseen style family 评估。

## Recommended setting batch

新增：

- `scripts/run_score_image_batch.py`

目的：用推荐设置检查多个 style/prompt 上是否仍保 content。

设置：

```text
inject_beta = 0.5
score_student_scale = 1.0
styles = 00,13,27
prompts = kp0,kp1,kp3
seed = 47
num_steps = 12
```

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 47 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_recommended_s47 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_batch_recommended_s47_summary.json
results/student_samples/score_student_batch_recommended_s47_contact.png
```

统计：

```text
style=00 prompt=kp0 ratio=0.3155 normal=[100.35,104.13,89.75] hard=[106.88,105.35,82.21] score=[110.68,115.39,82.46]
style=00 prompt=kp1 ratio=0.3486 normal=[135.45,151.34,159.73] hard=[139.76,156.90,162.66] score=[144.50,166.16,166.98]
style=00 prompt=kp3 ratio=0.3412 normal=[166.97,134.19,86.54] hard=[167.60,125.64,80.86] score=[175.59,129.84,76.94]

style=13 prompt=kp0 ratio=0.3445 normal=[100.35,104.13,89.75] hard=[109.57,109.95,89.36] score=[113.07,115.52,91.42]
style=13 prompt=kp1 ratio=0.3614 normal=[135.45,151.34,159.73] hard=[133.39,143.75,148.94] score=[137.64,148.65,153.15]
style=13 prompt=kp3 ratio=0.3421 normal=[166.97,134.19,86.54] hard=[145.86,113.67,82.92] score=[151.44,113.85,80.01]

style=27 prompt=kp0 ratio=0.4074 normal=[100.35,104.13,89.75] hard=[127.54,118.46,89.73] score=[127.41,121.11,88.13]
style=27 prompt=kp1 ratio=0.4447 normal=[135.45,151.34,159.73] hard=[150.51,157.76,154.80] score=[152.15,157.67,154.23]
style=27 prompt=kp3 ratio=0.4140 normal=[166.97,134.19,86.54] hard=[174.62,131.35,76.36] score=[178.25,125.39,67.93]
```

视觉结论：

- 9 个 style/prompt 组合都保住了主要 content。
- 没有出现 `inject_beta=0.65` 那种明显复制 style reference 房子/背景的失败。
- score_student 相比 hard 主要是色调/纹理细调，构图变化很小。
- 当前默认可用：`inject_beta=0.5, score_student_scale=1.0`。
- 如果要做更强风格化，优先升级 student/teacher，不建议简单提高 `inject_beta`。

## Recommended setting batch seed48

同一默认设置换 seed 复跑：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 48 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_recommended_s48 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_batch_recommended_s48_summary.json
results/student_samples/score_student_batch_recommended_s48_contact.png
```

统计：

```text
style=00 prompt=kp0 ratio_mean=0.3171 ratio_min=0.2347 ratio_max=0.5787
style=00 prompt=kp1 ratio_mean=0.3522 ratio_min=0.2764 ratio_max=0.4805
style=00 prompt=kp3 ratio_mean=0.3540 ratio_min=0.2819 ratio_max=0.6028
style=13 prompt=kp0 ratio_mean=0.3510 ratio_min=0.2625 ratio_max=0.6250
style=13 prompt=kp1 ratio_mean=0.3797 ratio_min=0.2862 ratio_max=0.5320
style=13 prompt=kp3 ratio_mean=0.3523 ratio_min=0.2643 ratio_max=0.6017
style=27 prompt=kp0 ratio_mean=0.3957 ratio_min=0.2803 ratio_max=0.7359
style=27 prompt=kp1 ratio_mean=0.4502 ratio_min=0.3431 ratio_max=0.6516
style=27 prompt=kp3 ratio_mean=0.4171 ratio_min=0.3164 ratio_max=0.7339
```

视觉结论：

- seed48 与 seed47 一致，`inject_beta=0.5` 下 9 个组合均保住 prompt 主体。
- score_student 仍是 hard CFG 之上的二阶修正，主要改变色调、对比、材质纹理。
- style27 的 correction ratio 最大，也对应更强的风格化趋势。
- 这轮支持把 `beta=0.5 / scale=1.0` 作为当前 qualitative 默认，但还不足以宣称 learned correction 单独完成风格迁移。

## Token-mixer score student

新增结构：

- `ScoreResidualTokenMixer`
- `build_score_residual_student`
- `scripts/train_score_residual_student.py --model-type token_mixer`

目的：让 direct CFG correction 不只做 token-wise MLP，而是在 latent
`32x32` 网格上做局部空间混合，测试纹理/布局 residual 是否更可学。

兼容性检查：

```text
score_residual_student_latent45_lam1_hard_delta (1, 1024, 128) True
score_residual_student_latent45_lam1_hard_delta_token_mixer (1, 1024, 128) True
```

说明旧 `token_mlp` checkpoint 和新 `token_mixer` checkpoint 都能通过同一个
runtime 加载。

主训练：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/train_score_residual_student.py \
  --manifest results/score_residual_manifest_latent45_lam1.jsonl \
  --out-dir results/score_residual_student_latent45_lam1_hard_delta_token_mixer \
  --epochs 120 \
  --batch-size 3 \
  --hidden-dim 384 \
  --cond-dim 128 \
  --time-embed-dim 128 \
  --model-type token_mixer \
  --mixer-layers 4 \
  --mixer-kernel 3 \
  --dropout 0.05 \
  --lr 0.0015 \
  --weight-decay 0.0001 \
  --val-fraction 0.2 \
  --prediction-mode hard_delta \
  --device cuda \
  --log-every 10
```

结果：

```text
token_mlp random:
  best_epoch = 12
  best_mse = 0.390818
  hard_base = 0.418083
  improvement = 0.027265

token_mixer random:
  best_epoch = 10
  best_mse = 0.389370
  hard_base = 0.418083
  improvement = 0.028714

token_mixer_small random:
  best_epoch = 51
  best_mse = 0.391142
  hard_base = 0.418083
  improvement = 0.026941
```

group holdout:

```text
token_mixer_prompt_kp4:
  best_epoch = 5
  best_mse = 0.398062
  hard_base = 0.420769
  improvement = 0.022708

token_mixer_style_27:
  best_epoch = 4
  best_mse = 0.412534
  hard_base = 0.436566
  improvement = 0.024032

token_mixer_timestep_363:
  best_epoch = 7
  best_mse = 0.305963
  hard_base = 0.334272
  improvement = 0.028309
```

图像级 smoke：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sample_student_images.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 48 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_token_mixer_image_style27_kp0_s48_beta0p5_steps12 \
  --results-dir results \
  --score-student-checkpoint results/score_residual_student_latent45_lam1_hard_delta_token_mixer/student_best.pt \
  --score-student-device cuda \
  --score-student-scale 1.0 \
  --inject-beta 0.5 \
  --modes normal,hard,score_student
```

输出：

```text
results/student_samples/score_student_token_mixer_image_style27_kp0_s48_beta0p5_steps12_normal.png
results/student_samples/score_student_token_mixer_image_style27_kp0_s48_beta0p5_steps12_hard.png
results/student_samples/score_student_token_mixer_image_style27_kp0_s48_beta0p5_steps12_score_student.png
results/score_student_token_mixer_image_style27_kp0_s48_beta0p5_steps12_meta.json
results/student_samples/score_student_mlp_vs_token_mixer_style27_kp0_s48_contact.png
```

结论：

- token-mixer 是有效结构 ablation：random/timestep 指标略优，prompt/style
  holdout 略差。
- 训练在 5-10 epoch 后快速过拟合；当前 45-sample 数据量不足以稳定吃下更高容量。
- 图像级结果与 token MLP 很接近，没有明显 qualitative 升级。
- 当前默认 checkpoint 仍用 `results/score_residual_student_latent45_lam1_hard_delta/student_best.pt`。
- 下一步要提高视觉结果，优先扩 latent manifest，再重训 token-mixer 或做 gated mixer/LoRA residual head。

## 120-sample style expansion

扩数据目标：从 3 个 style 扩到 8 个 style，验证 direct score correction
是否能更稳定泛化，而不是继续在 45 samples 上调结构。

新增 styles：

```text
02 anime_cel
05 blueprint
08 chinese_ink
22 manga_bw
34 stained_glass
```

采集命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_style_sweep.py \
  --probe-mode run-model \
  --style-ids 02,05,08,22,34 \
  --prompt-ids kp0,kp1,kp2,kp3,kp4 \
  --seed 45 \
  --selected-timesteps 700,500,300 \
  --out-prefix score_residual_latent120_extra_styles_lam1 \
  --results-dir results \
  --teacher-lambdas 1.0 \
  --teacher-highpass-kernel 5 \
  --sinkhorn-iters 3 \
  --runtime-height 512 \
  --runtime-width 512 \
  --min-free-gb 40 \
  --dump-latents
```

采集结果：

```text
new field dumps = 75
old field dumps = 45
global best teacher = teacher_hp_lam1
global teacher score = 0.726073
ranking = results/score_residual_latent120_extra_styles_lam1_global_teacher_ranking.json
```

manifest：

```text
results/score_residual_manifest_latent120_lam1.jsonl
results/score_residual_manifest_latent120_lam1_summary.json
num_samples = 120
styles = 00,02,05,08,13,22,27,34 each 15
prompts = kp0,kp1,kp2,kp3,kp4 each 24
timesteps = 363.147003,472.075500,718.322083 each 40
field_shape = [1,1024,128]
latent_dtype = float16
verified_teacher = true
```

random split 结果：

```text
latent45_mlp:
  best_epoch = 12
  val_mse = 0.390818
  hard = 0.418083
  improvement = 0.027265
  val_cos = 0.854317
  hard_cos = 0.838770

latent120_mlp:
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_mlp
  best_epoch = 35
  val_mse = 0.368352
  hard = 0.406864
  improvement = 0.038513
  val_cos = 0.861675
  hard_cos = 0.841386

latent120_token_mixer:
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_token_mixer
  best_epoch = 24
  val_mse = 0.369588
  hard = 0.406864
  improvement = 0.037276
  val_cos = 0.861949
  hard_cos = 0.841386
```

holdout style34：

```text
latent120_mlp_style34:
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_mlp_holdout_style34
  best_epoch = 19
  val_mse = 0.399988
  hard = 0.430052
  improvement = 0.030065
  val_cos = 0.861231
  hard_cos = 0.844364

latent120_token_mixer_style34:
  out_dir = results/score_residual_student_latent120_lam1_hard_delta_token_mixer_holdout_style34
  best_epoch = 16
  val_mse = 0.401241
  hard = 0.430052
  improvement = 0.028811
  val_cos = 0.862225
  hard_cos = 0.844364
```

图像级检查：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 48 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_latent120_mlp_recommended_s48 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_batch_latent120_mlp_recommended_s48_summary.json
results/student_samples/score_student_batch_latent120_mlp_recommended_s48_contact.png
```

ratio：

```text
style=00 prompt=kp0 ratio_mean=0.2649
style=00 prompt=kp1 ratio_mean=0.3058
style=00 prompt=kp3 ratio_mean=0.2639
style=13 prompt=kp0 ratio_mean=0.3132
style=13 prompt=kp1 ratio_mean=0.3817
style=13 prompt=kp3 ratio_mean=0.3156
style=27 prompt=kp0 ratio_mean=0.4054
style=27 prompt=kp1 ratio_mean=0.4803
style=27 prompt=kp3 ratio_mean=0.4061
```

结论：

- 扩 style 数据是目前最有效改造，优于继续调 token-mixer。
- 当前默认 checkpoint 升级为：

```text
results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt
```

- token-mixer 保留为结构 ablation；它 cosine 略高但 MSE 和 style-holdout
  不如 MLP。
- 图像级没有出现新 content leak，但视觉提升仍是细调；下一步如果要明显更强，
  应该做更多 style/prompt 数据或 LoRA residual head，而不是单纯放大
  `score_student_scale`。

## Score student scale sweep

新增：

- `scripts/evaluate_score_residual_student.py`

用途：对 direct score student 做 residual-space scale sweep：

```text
pred = hard_scfg + score_student_scale * f_theta(z_t,t,prompt,style)
```

random split：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/evaluate_score_residual_student.py \
  --manifest results/score_residual_manifest_latent120_lam1.jsonl \
  --checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --out-prefix score_residual_student_latent120_mlp_eval_random \
  --results-dir results \
  --student-scales 0,0.25,0.5,0.75,1,1.25,1.5,1.75,2 \
  --batch-size 6 \
  --device cuda \
  --seed 43 \
  --val-fraction 0.2
```

结果：

```text
scale=0    val_mse=0.406864 improvement=0.000000 delta_norm_ratio=0.000
scale=0.25 val_mse=0.387557 improvement=0.019308 delta_norm_ratio=0.088
scale=0.5  val_mse=0.374702 improvement=0.032162 delta_norm_ratio=0.176
scale=0.75 val_mse=0.368300 improvement=0.038564 delta_norm_ratio=0.264
scale=1    val_mse=0.368352 improvement=0.038513 delta_norm_ratio=0.352
scale=1.25 val_mse=0.374856 improvement=0.032008 delta_norm_ratio=0.439
scale=1.5  val_mse=0.387813 improvement=0.019051 delta_norm_ratio=0.527
scale=1.75 val_mse=0.407223 improvement=-0.000359 delta_norm_ratio=0.615
scale=2    val_mse=0.433086 improvement=-0.026222 delta_norm_ratio=0.703
best scale = 0.75
```

style34 slice：

```bash
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/evaluate_score_residual_student.py \
  --manifest results/score_residual_manifest_latent120_lam1.jsonl \
  --checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --out-prefix score_residual_student_latent120_mlp_eval_holdout_style34 \
  --results-dir results \
  --student-scales 0,0.25,0.5,0.75,1,1.25,1.5,1.75,2 \
  --batch-size 6 \
  --device cuda \
  --split-mode style \
  --holdout-value 34
```

结果：

```text
scale=0    val_mse=0.430052 improvement=0.000000 delta_norm_ratio=0.000
scale=0.25 val_mse=0.404278 improvement=0.025774 delta_norm_ratio=0.088
scale=0.5  val_mse=0.385284 improvement=0.044768 delta_norm_ratio=0.175
scale=0.75 val_mse=0.373071 improvement=0.056982 delta_norm_ratio=0.263
scale=1    val_mse=0.367637 improvement=0.062415 delta_norm_ratio=0.351
scale=1.25 val_mse=0.368983 improvement=0.061069 delta_norm_ratio=0.439
scale=1.5  val_mse=0.377109 improvement=0.052943 delta_norm_ratio=0.526
scale=1.75 val_mse=0.392015 improvement=0.038037 delta_norm_ratio=0.614
scale=2    val_mse=0.413701 improvement=0.016351 delta_norm_ratio=0.702
best scale = 1.0
```

输出：

```text
results/score_residual_student_latent120_mlp_eval_random_scale_sweep.csv
results/score_residual_student_latent120_mlp_eval_random_scale_sweep_summary.json
results/score_residual_student_latent120_mlp_eval_holdout_style34_scale_sweep.csv
results/score_residual_student_latent120_mlp_eval_holdout_style34_scale_sweep_summary.json
```

图像级 scale=0.75 batch：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 48 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_latent120_mlp_s0p75_s48 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 0.75 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_batch_latent120_mlp_s0p75_s48_summary.json
results/student_samples/score_student_batch_latent120_mlp_s0p75_s48_contact.png
```

结论：

- `scale=0.75` 与 `scale=1.0` 图像级都安全，没有新增 content leak。
- random split 更偏好 `0.75`，style34 slice 更偏好 `1.0`。
- 当前默认仍设为 `score_student_scale=1.0`；保守复现实验可用 `0.75`。
- 不建议 `scale >= 1.75`，这时 residual-space 已经回到或低于 hard CFG baseline。

## New-style image batch

目的：验证 latent120 默认 checkpoint 不只在旧 styles `00,13,27` 上稳定，也能在
新增 styles `02,05,08,22,34` 上保持 content。

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/run_score_image_batch.py \
  --style-ids 02,05,08,22,34 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 49 \
  --height 512 \
  --width 512 \
  --num-steps 12 \
  --out-prefix score_student_batch_latent120_mlp_newstyles_s49 \
  --results-dir results \
  --inject-beta 0.5 \
  --score-student-scale 1.0 \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --force
```

输出：

```text
results/score_student_batch_latent120_mlp_newstyles_s49_summary.json
results/student_samples/score_student_batch_latent120_mlp_newstyles_s49_contact.png
```

ratio：

```text
style=02 prompt=kp0 ratio_mean=0.3276
style=02 prompt=kp1 ratio_mean=0.3854
style=02 prompt=kp3 ratio_mean=0.3403
style=05 prompt=kp0 ratio_mean=0.2672
style=05 prompt=kp1 ratio_mean=0.3333
style=05 prompt=kp3 ratio_mean=0.2704
style=08 prompt=kp0 ratio_mean=0.3971
style=08 prompt=kp1 ratio_mean=0.4772
style=08 prompt=kp3 ratio_mean=0.3931
style=22 prompt=kp0 ratio_mean=0.2383
style=22 prompt=kp1 ratio_mean=0.3046
style=22 prompt=kp3 ratio_mean=0.2498
style=34 prompt=kp0 ratio_mean=0.2932
style=34 prompt=kp1 ratio_mean=0.3578
style=34 prompt=kp3 ratio_mean=0.3189
ratio_range = 0.2383 to 0.4772
ratio_average = 0.3303
```

视觉结论：

- 15 个新增 style/prompt 组合都保住了主要 prompt content。
- 没有出现 `inject_beta=0.65` 那类 style-reference content leak。
- style08 的 correction 最强，style22 最保守；二者都没有破坏主体。
- 这支持当前默认：

```text
checkpoint = results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt
inject_beta = 0.5
score_student_scale = 1.0
```

## Per-style scale support

新增可选参数：

```text
scripts/sample_student_images.py --score-student-style-scales
scripts/run_score_image_batch.py --score-student-style-scales
```

用法：

```bash
--score-student-scale 1.0 --score-student-style-scales 08:0.75,22:1.25
```

匹配到 style id 时，style map 覆盖全局 `--score-student-scale`。

smoke：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sample_student_images.py \
  --style-id 02 \
  --prompt-id kp0 \
  --seed 50 \
  --height 512 \
  --width 512 \
  --num-steps 4 \
  --out-prefix score_student_style_scale_smoke_style02_kp0_s50 \
  --results-dir results \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --score-student-scale 1.0 \
  --score-student-style-scales 02:0.5 \
  --inject-beta 0.5 \
  --modes normal,hard,score_student
```

meta 验证：

```text
effective = 0.5
map = {"02": 0.5}
step_scale = [0.5, 0.5, 0.5, ...]
```

per-style residual scale sweep：

```text
style=00 best_scale=1.00 best_mse=0.378285 hard=0.429302 improvement=0.051017
style=02 best_scale=1.00 best_mse=0.350537 hard=0.411211 improvement=0.060674
style=05 best_scale=1.00 best_mse=0.317695 hard=0.365963 improvement=0.048268
style=08 best_scale=1.00 best_mse=0.303077 hard=0.367472 improvement=0.064395
style=13 best_scale=1.00 best_mse=0.343779 hard=0.401224 improvement=0.057446
style=22 best_scale=1.00 best_mse=0.327990 hard=0.367699 improvement=0.039709
style=27 best_scale=1.00 best_mse=0.358062 hard=0.436376 improvement=0.078314
style=34 best_scale=1.00 best_mse=0.367637 hard=0.430052 improvement=0.062415
```

结论：

- per-style scale map 已可用于后续实验。
- 当前 8 个 style 的最优 scale 全是 `1.0`，所以默认不需要非平凡 style map。
- 这进一步支持当前全局默认 `score_student_scale=1.0`。

## Score student timestep window

动机：

- latent120 student 的 teacher 监督只来自三个 timestep：
  `718.322083`, `472.075500`, `363.147003`。
- 12-step sampler 实际会在 `1000.0 -> 395.470` 的完整区间调用 student。
- 因此加一个 timestep window 是合理的安全/诊断开关，用来测试“只在训练
  timestep 附近注入 correction”是否更稳。

新增参数：

```text
scripts/sample_student_images.py --score-student-timestep-min
scripts/sample_student_images.py --score-student-timestep-max
scripts/run_score_image_batch.py --score-student-timestep-min
scripts/run_score_image_batch.py --score-student-timestep-max
```

smoke：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sample_student_images.py \
  --style-id 27 \
  --prompt-id kp0 \
  --seed 51 \
  --height 512 \
  --width 512 \
  --num-steps 4 \
  --out-prefix score_student_timestep_window_smoke_style27_kp0_s51 \
  --results-dir results \
  --score-student-checkpoint results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt \
  --score-student-device cuda \
  --score-student-scale 1.0 \
  --score-student-timestep-max 718.4 \
  --inject-beta 0.5 \
  --modes normal,hard,score_student
```

meta 验证：

```text
score_student_timestep_min = null
score_student_timestep_max = 718.4
step_scale = [
  (1000.0, 0.0),
  (958.085, 0.0),
  (883.982, 0.0),
  (717.497, 1.0)
]
```

同 seed 图像对照：

```text
styles = 00,13,27
prompts = kp0,kp1,kp3
seed = 52
inject_beta = 0.5
score_student_scale = 1.0
checkpoint = results/score_residual_student_latent120_lam1_hard_delta_mlp/student_best.pt
```

full runtime：

```text
results/score_student_batch_latent120_mlp_full_s52_summary.json
results/student_samples/score_student_batch_latent120_mlp_full_s52_contact.png

active_step_count = 12
step_scale_mean = 1.0
scaled_ratio_mean = 0.348995
scaled_ratio_min/max over rows = 0.262272 / 0.470553
```

`--score-student-timestep-max 718.4`：

```text
results/score_student_batch_latent120_mlp_twmax718_s52_summary.json
results/student_samples/score_student_batch_latent120_mlp_twmax718_s52_contact.png

active_step_count = 3
step_scale_mean = 0.25
scaled_ratio_mean = 0.137189
scaled_ratio_min/max over rows = 0.113911 / 0.162451
```

视觉观察：

- full 和 window 两组都保住了主要 prompt content。
- window 主要效果是把 learned correction 降到 full 的约 39%，图像更接近
  hard CFG。
- 这组没有看到 full runtime 的明显 content leak，因此 window 暂时没有提供
  足够强的默认收益。

当前结论：

```text
默认：不设置 score_student_timestep_min/max
保守/诊断：--score-student-timestep-max 718.4
```

timestep window 应作为 ablation 与风险控制开关保留，但第二篇主线仍应报告
full-runtime score correction，因为它保留了更强的 learned residual signal。
## Native Base9B With-Style Teacher Audit (2026-07-13)

The previous hard/OT teachers were built from manually replaced reference K/V.
The new V0 teacher instead uses `/mnt/local/klein9b-base` through FLUX.2's
native edit-image conditioning path. The with-reference and without-reference
velocities share the exact target latent, timestep, prompt, model, and schedule.
Only clean edit-image tokens differ, and no attention hook is installed.

Reproduction:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/miniconda3/envs/diffsynth/bin/python \
  scripts/probe_native_edit_teacher.py \
  --model-root /mnt/local/klein9b-base \
  --style-ids 00,13,27 \
  --prompt-ids kp0,kp1,kp3 \
  --seed 65 \
  --height 512 --width 512 --num-steps 12 \
  --output-prefix results/native_edit_teacher_field_audit_base9b_s65
```

At `t=1000`, the old dynamically noised reference K/V branch was exactly
reference-blind. The native teacher is strongly reference-discriminative:

```text
average between-reference energy fraction  40.4%
average pairwise reference cosine            0.403
average pairwise reference relative L2       1.103
average between-prompt energy fraction      21.8%
average delta / no-reference norm            20.7%
```

Across the measured trajectory, reference energy falls into the `15-34%`
range, while prompt energy grows and reaches `60.6%` at `t=395.47`. Thus native
conditioning resolves the old `t=1000` reference-information failure, but its
large early-time field must be audited for reference layout/object leakage
before it is accepted as a pure style teacher.

Artifacts:

- `results/native_edit_teacher_field_audit_base9b_s65_summary.json`
- `results/native_edit_teacher_field_audit_base9b_s65.csv`

## TPAMI Native-Teacher Smoke Status (2026-07-13)

The native Base9B teacher is now wired into the TPAMI latent loop through:

```text
scripts/run_tpami_with_style_teacher.py
```

and validated by:

```text
scripts/test_tpami_with_style_teacher.py
```

Current regression status:

```text
15 tests passing locally and on ubuntu-107
```

The most important runtime fix was not in the teacher field itself but in the
teacher condition graph:

```text
teacher prompt embeds were initially created with autograd history
-> runner moved teacher text encoder back to CPU to save VRAM
-> Plan A backward hit a cuda/cpu mixed graph
```

The fix is now locked in code:

```text
encode_prompt_for_pipe       -> torch.no_grad() + detach()
prepare_edit_condition       -> torch.no_grad() + detach()
runner static-condition gate -> grad_fn must be None / requires_grad False
```

Accepted smoke commands:

```bash
# CA-only
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --main-model-root /mnt/local/klein9b \
  --teacher-model-root /mnt/local/klein9b-base \
  --style-mode ca_only \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --height 512 --width 512 \
  --num-steps 12 --iters 1 --disable-dynamic-iters \
  --main-vram-limit-gb 24 --teacher-vram-limit-gb 24 \
  --out-prefix tpami_native_teacher_smoke_dualvram24 \
  --results-dir results

# score_match
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/chengyanli/flux/bin/python3 \
  scripts/run_tpami_with_style_teacher.py \
  --main-model-root /mnt/local/klein9b \
  --teacher-model-root /mnt/local/klein9b-base \
  --style-mode score_match \
  --style-id 27 --prompt-id kp0 --seed 64 \
  --height 512 --width 512 \
  --num-steps 12 --iters 1 --disable-dynamic-iters \
  --main-vram-limit-gb 16 \
  --out-prefix tpami_native_teacher_smoke_main16 \
  --results-dir results
```

Smoke results:

```text
ca_only:
  scheduler_step_calls_total = 12
  peak cuda memory           = 66.9902 GB

score_match:
  scheduler_step_calls_total = 12
  peak cuda memory           = 45.5204 GB
  teacher loading mode       = full CUDA
```

Artifacts:

```text
results/tpami_native_teacher_smoke_dualvram24_ca_only_style27_kp0_seed64.{png,csv,summary.json}
results/tpami_native_teacher_smoke_main16_score_match_style27_kp0_seed64.{png,csv,summary.json}
```

Current interpretation:

```text
the native teacher path is now execution-stable for both CA-only and
score-match smoke settings
```

But these are still only smoke checks:

```text
12 steps / 1 inner iter / one style / one seed
```

Formal visual or quality claims must wait for the full `25-step / 5-iter`
comparison and multi-style / multi-seed evaluation.

Plan B remains incomplete in the DMD sense:

```text
ca_only is real
ca_dmd still needs a dynamic Base9B-compatible fake scorer
```

The existing DMD2 implementation is not a correctness oracle. In particular,
the clean flow target must remain:

```text
noise - fake_latents_clean
```

and not the incorrect:

```text
noise - noisy_latents
```
