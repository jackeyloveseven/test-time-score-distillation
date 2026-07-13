# Environment Notes

## 推荐运行环境

- 首选（Klein9B 运行链路）：`/home/ubuntu/miniconda3/envs/diffsynth`
  - 验证项：`torch==2.11.0+cu130`, `diffsynth/einops/PIL/numpy/tqdm` 全部可用
  - 已执行：`/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sanity_import.py`，输出 `OK`

- 备选（同样适合 Klein9B runtime）：`/home/ubuntu/miniconda3/envs/yzh-dmd`
  - 验证项：`torch==2.10.0+cu128`, `diffsynth/einops/PIL/numpy/tqdm` 全部可用
  - 已执行：`/home/ubuntu/miniconda3/envs/yzh-dmd/bin/python scripts/sanity_import.py`，输出 `OK`

- 纯 tensor 测试环境（不含 diffsynth）：`/home/ubuntu/miniconda3/envs/sensenova`
  - 可用于纯张量 smoke test，但不建议承接后续 Klein9B runtime。

## 验收顺序（当前版本）

1. `python3 -m py_compile src/*.py scripts/sanity_import.py`
2. `/home/ubuntu/miniconda3/envs/diffsynth/bin/python scripts/sanity_import.py`
3. `/home/ubuntu/miniconda3/envs/yzh-dmd/bin/python scripts/sanity_import.py`
