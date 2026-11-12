# 本地验证

验证日期：2026-11-12。使用 Python 3.10.14、NVIDIA A100-SXM4-80GB。
测试使用小模型、本地 tokenizer 和合成数据，不下载预训练权重或启动完整规模训练。

| 项目 | 结果 |
|---|---|
| 全量回归 | **130 passed, 2 skipped, 4 deselected**，130.69 秒 |
| 跳过项目 | 老版 PyTorch 不支持的 CPU FP16 autocast 组合 |
| 默认排除项目 | 4 个需下载模型的 `slow` 测试 |
| 包构建 | wheel 与 source distribution 构建成功 |
| wheel 独立安装 | 临时目录安装后，6 项检查通过 |
| CUDA 采样 | LangFlow / RePlaid 均产生 `[2,8]` token tensor；固定种子可重复；CPU/CUDA 全局 RNG 均保持 |
| LangFlow 训练入口 | 打包的 Hydra 配置完成 2 步离线 CPU 训练及验证，导出 Lightning 与 HF 权重 |
| LangFlow 导出恢复 | Lightning EMA checkpoint 与导出的 safetensors 逐 tensor 一致 |
| RePlaid 训练入口 | 小模型训练、精确恢复、NELBO、采样及双进程 CPU 路径通过回归 |
| 静态检查 | 全部源码通过 `E9,F63,F7,F82`；新增入口/API/测试与示例通过 Ruff 检查 |
| 原工程隔离 | 94 个来源文件仍与迁移时记录的哈希一致；原目录无需加入 Python 搜索路径 |

主要依赖版本：

| 依赖 | 本次测试版本 |
|---|---|
| torch | 2.3.1 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| transformers | 4.48.3 |
| safetensors | 0.8.0 |
| lightning / pytorch-lightning | 2.5.0.post0 |
| datasets | 3.6.0 |
| pyarrow | 19.0.1 |
| hydra-core | 1.3.8 |
| pytest | 9.2.0 |
| setuptools | 80.10.2 |

复跑：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest -q
.venv/bin/python -m build --no-isolation
.venv/bin/python scripts/verify_wheel.py dist/condila-0.1.0-py3-none-any.whl
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python examples/tiny_models.py
```

wheel 检查在新的临时目录中安装构建物，确认实际导入位置属于该安装目录，
然后检查指标兼容别名、包内 YAML、两个模型的采样、控制台脚本、
LangFlow Hydra 配置解析和 RePlaid 无数据 dry-run。
具体过程见 [verify_wheel.py](../scripts/verify_wheel.py)。

复用源工程的指标与模型回归测试时只调整了命名空间、打包资源路径和独立测试所需的入口。
依赖旧实验队列目录的 `test_ppl_pipeline.py` 不属于库接口，未迁入；
对应实验和后台任务保持在原 CDLM_Benchmarks 工程中。

本次没有重跑论文规模训练、预训练 checkpoint 的完整评测或跨依赖版本矩阵。
测试结果验证本次库整合的功能和数值回归，不代表达到论文报告指标。
