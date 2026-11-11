# <img src="docs/assets/condila-logo.svg" alt="" width="58" height="36" align="absmiddle"> Condila

Condila 是面向连续扩散语言模型的 Python 工具库，提供模型配置、训练、
checkpoint 管理、采样和生成评测接口，支持 LangFlow 与 RePlaid。

```python
from condila.dlm_metrics import MMD, GenerativePerplexity, evaluate
from condila.cdlm import LangFlow, LangFlowConfig, RePlaid, RePlaidConfig
```

## 安装

要求 Python 3.10+、PyTorch 2.3+。GPU 环境需安装与 CUDA 版本匹配的 PyTorch。
在项目根目录执行：

```bash
python -m pip install -e .
```

根据使用场景安装可选依赖：

| Extra | 功能 |
|---|---|
| `train` | Lightning / Hydra 训练与数据加载 |
| `data` | 数据集处理与 Parquet 支持 |
| `mauve` | MAUVE 评测 |
| `topics` | 主题覆盖统计 |
| `tracking` | W&B 实验跟踪 |
| `test` | 测试、静态检查与构建工具 |
| `all` | 全部运行时可选功能 |

```bash
python -m pip install -e '.[train,data]'
python -m pip install -e '.[mauve,topics,tracking]'
```

## 命名空间

| 入口 | 内容 |
|---|---|
| `condila.dlm_metrics` | `Metric`、`MetricCollection`、MMD、MAUVE、PRDC、GenPPL、熵、重复与覆盖诊断 |
| `condila.dlm_metrics.functional` | 指标的函数式接口 |
| `condila.dlm_metrics.likelihood` | NELBO / ODE 似然估计与模型适配器 |
| `condila.cdlm` | 模型发现、注册、创建、加载、统一 token 采样 |
| `condila.cdlm.langflow` | LangFlow 配置、HF 模型、采样、LightningModule、DataModule |
| `condila.cdlm.replaid` | RePlaid 配置、模型、训练目标、数据处理、训练、原生评估与采样 |

`condia.dlm_metrics` 为兼容命名空间，与 `condila.dlm_metrics` 共享模块和类。

## 模型与统一采样

```python
from condila.cdlm import create_model, list_models, from_pretrained, generate

print(list_models())  # ('langflow', 'replaid')

# 从配置创建随机初始化模型
model = create_model(
    'langflow', vocab_size=32, hidden_size=16, cond_dim=8,
    n_blocks=1, n_heads=2, dropout=0.0, model_length=8,
)
ids = generate(model, num_samples=2, length=8, steps=2, seed=42)
assert ids.shape == (2, 8)

# 加载 LangFlow checkpoint
model = from_pretrained('langflow', '/path/to/langflow-checkpoint', device='cuda')
ids = generate(model, num_samples=8, length=1024, steps=128, seed=42)

# 加载 RePlaid checkpoint，默认使用 EMA 权重
model = from_pretrained('replaid', '/path/to/checkpoint-1000000', device='cuda')
ids = generate(model, num_samples=8, steps=128, sampler='ddpm', seed=42)
```

`generate` 返回形状为 `[num_samples, length]` 的 token-ID tensor。
采样期间模型处于 eval 模式，完成后恢复各模块的训练状态。
指定 `seed` 时，第 i 条输出使用 `seed+i` 作为独立随机种子，
不修改全局随机数状态。

统一接口的 `steps` 表示求解区间数。LangFlow 对应原生参数
`num_steps=steps+1`，RePlaid 对应原生参数 `steps=steps`。

| 模型 | 采样器 | 每区间去噪调用次数 |
|---|---|---|
| LangFlow | `euler_edm`，`temperature=1` | 1 |
| RePlaid | `ddpm`、`ddim` | 1 |
| RePlaid | `heun` | 2 |

所有采样器在最后增加一次 decoder 调用。

也可使用模型原生接口：

```python
from condila.cdlm.langflow import LangFlow, LangFlowConfig
from condila.cdlm.replaid import RePlaid, RePlaidConfig, load_checkpoint

langflow = LangFlow(LangFlowConfig())
replaid = RePlaid(RePlaidConfig())
model, tokenizer, experiment = load_checkpoint('/path/to/checkpoint', device='cpu')
```

LangFlow 支持 Hugging Face 模型 ID、本地模型目录、safetensors 文件、
分片权重和 Lightning checkpoint。独立权重文件需要同目录的 `config.json`
或显式传入 `config=`；Lightning checkpoint 可使用其中记录的架构信息。

RePlaid checkpoint 目录包含 `experiment_config.json`、`model.safetensors`、
`ema.safetensors` 和 `tokenizer/`。通过 `load_checkpoint` 可同时获取模型、
分词器和实验配置；通过 `weights='raw'` 可选择非 EMA 权重。

## 指标与模型联用

```python
import numpy as np
from condila.dlm_metrics import MMD

# 使用预计算特征评估分布差异
rng = np.random.default_rng(42)
metric = MMD(bandwidth=1.0)
metric.update_features(rng.normal(size=(64, 16)), real=True)
metric.update_features(rng.normal(size=(64, 16)) + 0.3)
print(metric.compute())
```

```python
from condila.dlm_metrics import evaluate

report = evaluate(
    {'langflow': '/path/to/samples.pt'}, reference='/path/to/owt_reference.pt',
    tokenizer='gpt2', n=512, device='cuda',
)
print(report.table())
report.save('quality.json')
```

生成样本需至少包含 `n` 条，`reference` 需至少包含 `2*n` 条，
分别用于 scoring reference 和 real–real 对照。
`evaluate` 默认使用 GPT-2-large 作为评估模型，首次使用时下载权重。
分词器应与生成样本的 token-ID 编码一致。

MMD 同时返回 U-MMD² 和 signed-root 展示值。`null_sd` 表示 permutation null
分布的标准差，不作为模型比较的置信区间。
完整参数与统计口径见 [指标接口说明](docs/metrics.md)。

LangFlow 与指标似然模块可直接联用：

```python
from condila.cdlm import from_pretrained
from condila.dlm_metrics.likelihood import LangFlowAdapter, NLLBound

langflow_model = from_pretrained('langflow', '/path/to/langflow-checkpoint', device='cuda')
bound = NLLBound(LangFlowAdapter(langflow_model), estimator='ode', n_steps=128)
# real_ids 为使用相同分词器编码的留出数据
bound.update(real_ids)
print(bound.compute())
```

RePlaid 的留出集 NELBO 使用
`condila.cdlm.replaid.objective.validation_terms` 或 `condila-replaid-evaluate`。

## 训练与命令行

```bash
condila models
condila metrics --help
condila-metrics eval samples.pt --reference owt_ref.pt --preset owt

# LangFlow
condila-langflow-train --cfg job
condila-langflow-train model=langflow-owt data=openwebtext
condila-langflow-sample --checkpoint /path/to/langflow --output samples.pt \
  --num-samples 512 --batch-size 8 --steps 128

# RePlaid
condila-replaid-train --dry-run --device cpu
condila-replaid-prepare --help
condila-replaid-train --data-dir /path/to/owt_gpt2 --output-dir outputs/replaid
condila-replaid-evaluate --checkpoint /path/to/checkpoint --output nelbo.json
condila-replaid-sample --checkpoint /path/to/checkpoint --output samples.jsonl
condila-replaid-score --samples samples.jsonl --output quality.json
condila-replaid-reference --help
```

模型命令支持 `python -m condila.cdlm.<model>.<module>`。
LangFlow 的训练入口为 `condila.cdlm.langflow.cli`，采样入口为
`condila.cdlm.langflow.inference`。RePlaid 支持使用
`torchrun -m condila.cdlm.replaid.train ...` 启动多进程训练。

LangFlow 使用 Hydra 配置，支持参数覆盖、`resume` 恢复训练与
`init_from` 权重初始化。验证结束后保留最佳 checkpoint，训练过程中定期保存
最近状态和 `last.ckpt`；训练结束时导出 `final_model.ckpt` 和 HF 格式权重。

实验日志默认写入本地 CSV。W&B 跟踪通过 `tracking` extra、环境变量和训练配置启用。

## 开发

```bash
python -m pip install -e '.[train,data,test]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m build
```

默认测试集不包含需要下载模型的 `slow` 测试。使用 `python -m pytest -m slow`
运行模型下载相关的集成测试。

## 文档

- [指标接口](docs/metrics.md)
- [扩展模型与指标](docs/extending.md)
- [使用示例](examples)

## 许可

[MIT License](LICENSE)
