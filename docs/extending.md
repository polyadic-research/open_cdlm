# 扩展 Condila

源码布局采用 `src/`：

```text
src/condila/
  cdlm/
    registry.py      # 模型发现、创建、加载和统一采样
    langflow/        # 模型、加载、Lightning 训练与数据工具
    replaid/         # 模型、原生目标、训练、数据、评估、采样
  dlm_metrics/       # 原有指标、functional 与 likelihood
```

新增 CDLM 可放入 `condila.cdlm.<model>`，然后注册原生模型、配置、checkpoint loader
和返回 token IDs 的 sampler：

```python
from condila.cdlm import register_model

register_model(
    'my_model',
    model_class='my_package.model:MyModel',
    config_class='my_package.config:MyConfig',
    loader='my_package.api:from_pretrained',
    sampler='my_package.api:sample_tokens',
    samplers=('my_sampler',),
)
```

字符串入口按需导入。也可以直接传类或函数。loader 签名是
`loader(checkpoint, *, device='cpu', **kwargs) -> model`；sampler 签名是
`sampler(model, *, num_samples, length, steps, seed, sampler, temperature, **kwargs)`，
返回整数 tensor `[num_samples, length]`。注册名忽略大小写，默认拒绝重名覆盖。
注册只对当前 Python 进程有效。

配置实例保持原生类型；配置 dict 与 `create_model` 的关键字参数可合并。
创建返回随机初始化模型，加载权重必须使用 `from_pretrained` 或模型原生接口。
公共 `generate` 在执行前设为 eval，并在结束或异常时恢复各层 training 标记。
原生的训练目标、self-conditioning 策略和 checkpoint 格式由模型自身定义。

新指标可以继承 `condila.dlm_metrics.Metric`，实现 `update` 和 `_compute`，
并通过 `MetricCollection` 组合。为多个 feature metric 共享一个
`TextFeatureExtractor`，可复用特征提取缓存。
