> 原复现工程的说明快照；其中相对路径和脚本属于原工程。新包入口以根目录 README 为准。

# RePlaid 原版论文对齐复现：OpenWebText

基于旁边的 Open CDLM 源码整理的独立工程，目标是 RePlaid 论文第 4 节的
OpenWebText、1024 token、0.1B 规模实验。提供训练自条件开启和关闭两套配置。
这是依据论文重新实现的代码，不是作者官方仓库，也尚未完成 100 万步的论文指标验证。

依据：[RePlaid 论文 v3](https://arxiv.org/html/2605.18530v3)，重点为
[算法 1 和自条件](https://arxiv.org/html/2605.18530v3#A4)、
[架构及精度表 5、6](https://arxiv.org/html/2605.18530v3#A11)、
[OWT 优化器表 9](https://arxiv.org/html/2605.18530v3#A12)。
源文件和参考实现的版本记录在 [PROVENANCE.json](PROVENANCE.json)。

## 相对原目录还原的内容

| 项目 | 本工程 |
|---|---|
| tokenizer | GPT-2，50257 词；仅使用分词器，不加载预训练语言模型权重 |
| 扩散状态 | 16 维、可学习的单位球面词嵌入，VP 高斯加噪 |
| Transformer | 768 宽、12 层、12 个普通注意力头；LayerNorm，无 QK RMSNorm |
| MLP / 条件 | 带偏置的 GELU(tanh) MLP；block 和最终输出层均使用 adaLN-Zero |
| 时间输入 | 恢复输入端 Fourier gamma 投影，以及 block / 输出层的条件路径 |
| 自条件 | 训练时固定 25%；概率加权嵌入，无置信度门控；验证时 100% 同时刻 bootstrap |
| 训练损失 | 重建 CE + SNR 导数加权嵌入 MSE + 高斯先验 KL |
| 噪声调度 | FP64 单调 MLP；端点优化平均损失，内部形状优化损失二阶矩 |
| 优化器 | AdamW；主干 LR 3e-4，嵌入/调度/端点 LR 1e-2；warmup 2500 后恒定 |
| 正则与 EMA | 主干 weight decay 0，端点 0.1，dropout 0.1，EMA 0.9999 |
| 精度 | 输入投影/残差 FP32，主干/最终线性层 BF16，调度/SNR 加权/聚合 FP64 |
| 批次 | 全局 512；在整个每卡优化器批次上分配重建/扩散样本，再分成微批次 |

修复了原代码保存 checkpoint 缺少 json 导入的问题。新包名为 condila.cdlm.replaid，
不依赖原目录的 Python 包。checkpoint 同时保存 raw / EMA、优化器、调度器、
各 rank 的 RNG、已完成步数和数据清单。

## 环境

已在 Python 3.10.14、PyTorch 2.3.1、CUDA、A100 上验证。
当前目录已有使用宿主 PyTorch 的 .venv。直接使用 .venv/bin/python 即可。
重新安装时：

~~~bash
# 先确保当前 Python 环境中的 PyTorch 与 CUDA 匹配
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[data,dev]'
~~~

NumPy 限制为 1.x，以兼容现有 PyTorch 2.3 的 NumPy 接口。
所有以下命令均从本目录执行。运行单元测试：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
~~~

## 1. 准备 OpenWebText

输入为具有 text 列的原始 OpenWebText parquet / JSONL；txt 则每行视为一篇文档。
传入的文件和行顺序必须与要比较的原始语料一致。默认将最后 100000 篇文档
划为留出集，分割发生在分词之前，不会从训练 token 中抽取验证样本。

~~~bash
.venv/bin/python -m condila.cdlm.replaid.prepare \
  --input '/path/to/openwebtext/*.parquet' \
  --output data/owt_gpt2
~~~

只会下载 GPT-2 tokenizer。也可以用 --tokenizer 指向本地 GPT-2 tokenizer。
预处理遵循 [MDLM 的公开 OWT 数据流程](https://github.com/kuleshov-group/mdlm/blob/main/dataloader.py)：
每篇文档末尾附加 EOS；每 1000 篇文档的 token 拼接后切成 1022 token，
再在每个块两端加入 BOS / EOS，得到 1024 token；该组剩余 token 丢弃。
GPT-2 的 BOS / EOS 均为 50256。这里关闭了原目录的随机裁剪和逐文档额外 BOS。

输出为：

~~~
data/owt_gpt2/
  manifest.json
  tokenizer/
  train.bin
  validation.bin
~~~

二进制为固定长度 uint16 行。清单记录划分、打包参数、来源顺序、tokenizer 指纹、
样本数与数据 SHA256。训练使用内存映射，按块均匀有放回抽样；
批次由 seed、rank 和微批次序号决定，恢复无需遍历或跳过之前的数据。

## 2. 训练

默认配置是 1000000 次更新、全局 batch 512。先检查结构和拓扑，不需要数据：

~~~bash
PYTHON=.venv/bin/python bash scripts/train.sh --dry-run
~~~

本机为 4 卡，脚本默认使用 4 进程；每卡微批次 8，自动梯度累积 16 次。
8 卡时自动改为累积 8 次，全局 batch 保持 512。

~~~bash
PYTHON=.venv/bin/python bash scripts/train.sh

NPROC_PER_NODE=8 PYTHON=.venv/bin/python bash scripts/train.sh

# 显存不足时缩小微批次，累积次数自动增加
PYTHON=.venv/bin/python bash scripts/train.sh \
  --batch-size 4 --gradient-checkpointing
~~~

模型、训练日志和验证记录默认写入 outputs/owt_sc。每 10000 步保存 checkpoint；
默认保留最近 5 份，并额外保留论文对照所需的 250000 / 1000000 步；
save_total_limit=0 可保留全部，checkpoint_keep_steps 可调整额外保留的步数。
训练中的短验证默认只覆盖 64 个批次，其 PPL 是监控值，正式评测应单独运行下一节命令。
可以先做少量更新，再恢复，warmup 与学习率定义不变：

~~~bash
PYTHON=.venv/bin/python bash scripts/train.sh --stop-after-steps 20
PYTHON=.venv/bin/python bash scripts/train.sh --resume latest
~~~

精确恢复要求 GPU 数、微批次、全局批次、训练配方和数据清单一致。
可使用 --data-dir、--output-dir 覆盖数据及输出目录。
多机时设置 NNODES、NODE_RANK、MASTER_ADDR、MASTER_PORT；NPROC_PER_NODE 是每机进程数。

原论文无自条件对照应从头训练：

~~~bash
CONFIG=configs/owt_no_sc.yaml PYTHON=.venv/bin/python bash scripts/train.sh
~~~

采样时给有自条件的 checkpoint 加 --no-self-conditioning 仅代表推理消融，
与单独训练的无自条件模型不同。

## 3. 留出集 NELBO / PPL

默认使用 EMA。每条留出序列都计算重建、扩散和先验三项；
有自条件模型对全部验证样本执行同一噪声状态下的 bootstrap，
不会将生成文本的 GPT-2 困惑度当作模型的似然上界。

~~~bash
.venv/bin/python -m condila.cdlm.replaid.evaluate \
  --checkpoint outputs/owt_sc/checkpoints/checkpoint-1000000 \
  --batch-size 8 --mc-samples 1 --output outputs/owt_sc/nelbo.json
~~~

去掉 --max-batches 时遍历整个留出集；可增加 --mc-samples 降低随机估计误差。
支持用 torch.distributed.run 启动多进程，留出行不重复填充。
输出明确记录 PPL 上界、三项 nats/token、样本数、随机种子和权重选择。

## 4. 生成、GenPPL 与 MAUVE

~~~bash
.venv/bin/python -m condila.cdlm.replaid.sample \
  --checkpoint outputs/owt_sc/checkpoints/checkpoint-1000000 \
  --output outputs/owt_sc/samples_128.jsonl \
  --num-samples 5120 --batch-size 8 --length 1024 --steps 128 --sampler ddpm
~~~

默认温度 1，DDPM，均匀时间网格。还保留 DDIM 和 Heun。
--steps 表示积分区间数；DDPM/DDIM 每个区间一次去噪，Heun 两次，
最后的 decoder 调用额外计算一次。实际 NFE 写入 metadata，避免将 Heun
区间数直接与 DDPM 的去噪次数比较。每个 sample ID 使用独立种子，支持多进程分片。

导出固定的留出文本参考并计算 GPT-2 Large GenPPL：

~~~bash
.venv/bin/python -m condila.cdlm.replaid.reference \
  --data-dir data/owt_gpt2 --num-samples 5120 \
  --output outputs/reference_5120.jsonl

.venv/bin/python -m condila.cdlm.replaid.score \
  --samples outputs/owt_sc/samples_128.jsonl \
  --output outputs/owt_sc/quality_128.json
~~~

GenPPL 需要另外下载 GPT-2 Large 权重。MAUVE 是可选依赖：

~~~bash
.venv/bin/python -m pip install -e '.[eval]'
.venv/bin/python -m condila.cdlm.replaid.score \
  --samples outputs/owt_sc/samples_128.jsonl \
  --reference outputs/reference_5120.jsonl --mauve \
  --output outputs/owt_sc/quality_with_mauve.json
~~~

MAUVE 默认使用论文指定的 answerdotai/ModernBERT-large，自行提取末尾有效 token 的
特征后传给 MAUVE。特征池化、特殊 token 的处理会影响指标；报告中明确记录这些选择。
默认对含特殊 token 的 text 评分，--text-field clean_text 可切换为移除特殊 token 的文本。
没有下载评估模型时，可用 --skip-gen-ppl 只检查本地多样性指标。

## 复现边界与核对事项

- 目前交付的是代码和运行验证，没有进行 OWT 的 100 万步训练，也没有声称达到论文 PPL 22.1。
  合成数据的 smoke test 数值不是论文结果。
- 论文 v3 的可复现性清单仍写着计划公开代码；本次未取得作者 RePlaid 源码。
  架构以论文和其所指的 MDLM small 配置为依据，扩散细节沿用 Open CDLM / Plaid。
- 当前 GPT-2 配置实际共有 131909219 个参数。论文表 11 的 116M 与这一
  GPT-2 / DDiT 定义的直接计数不一致，该差异尚不能通过作者代码核实；
  实际计数保存在 environment.json，没有将其标成“精确 116M”。
- cond_dim=128 来自 MDLM small；gamma 初始化、Fourier 特征、输出先验 warmup
  来自 Plaid / 原本地核心；seed=42、梯度范数阈值 1 沿用本地默认。
  论文未充分规定的初始化、数据顺序、抽样顺序与 MAUVE 池化选择都可能影响数值。
- 留出末尾 100000 篇文档和按 1000 篇打包遵循公开 MDLM 流程。
  如果本地 parquet 的排列不同于该流程的原语料排列，划分和结果也会不同。
- 这里实现的是 OWT 基准的训练、NELBO 和无条件生成流程；SlimPajama scaling-law
  sweep、TinyGSM 条件生成、PFODE 似然和 DPM-Solver++ 并未纳入这一工程。

验证记录见 [VERIFICATION.md](VERIFICATION.md)。
