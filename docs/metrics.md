# Metrics API

原有指标接口已迁入 `condila.dlm_metrics`。以下为原接口参数摘要。



All entropies are in nats. "n-dependent" means the value changes with the number of rows even for a perfect
generator: compare such metrics only at equal `n` (and equal row length `L`).

**Token level (no model).**

| Class | Returns | Hyperparameters (defaults) | Notes |
|---|---|---|---|
| `EntropyDecomposition` | `H_row`, `H_pool`, `JSD_rows` (+ `*_sd_half`) | `ignore_ids=None`, `split_half=0`, `seed=0` | `H_row` = LangFlow / FLM "sample entropy" (within-row). `JSD_rows = H_pool - sum_r w_r H_r` (between-row). `H_pool`, `JSD_rows` n- and L-dependent |
| `SampleEntropy`, `PooledEntropy`, `JSDRows` | one of the above | as above | |
| `RepetitionRate` | `rep_{n}` | `n=4` | mean within-row share of repeated n-grams |
| `DistinctN` | `distinct_{n}` | `n=4` | corpus-level distinct n-grams / n-grams; n-dependent |
| `NGramDiversity` | `VS_{n}`, `xD_{n}`, `NDR_{n}`, `MPJ_{n}` | `n=4`, `near_dup_threshold=0.5` | n-gram-set Vendi / n, cross-row novelty, near-duplicate rate, mean pairwise Jaccard; O(n²) |
| `ExactDuplicateRate` | `exact_dup_rate` | | rows identical to another row |
| `CompressionRate` | `lzma_bits_per_token` | | |

**Judge LM.**

| Class | Returns | Hyperparameters (defaults) | Notes |
|---|---|---|---|
| `GenerativePerplexity` | `gen_ppl`, `nll_per_token`, `n_tokens` (+ `gen_ppl_sd_half`, `row_nll_per_token`) | `judge="gpt2-large"`, `tokenizer`, `protocol="langflow"`, `max_length=None`, `batch_size=None`, `split_half=0`, `return_rows=False` | `protocol="langflow"`: LangFlow `gen_ppl.py` (special tokens kept as text, judge truncation 1024, all judge tokens but the first). `protocol="flm"`: MDLM / DUO / FLM `metrics.py` (truncation at the generator row length, EOS excluded except the first). The two protocols give different numbers: never mix them |

**Feature space** (two-sample, `update(x, real=...)`; common arguments `extractor`, `device`, `normalize=True`,
`reset_real_features=True`, `tokenizer`).

| Class | Returns | Hyperparameters (defaults) | Notes |
|---|---|---|---|
| `MMD` | `mmd`, `mmd2`, `bandwidth` (+ `null_sd`, `null_mean`) | `bandwidth="median"`, `bandwidth_multiplier=1.0`, `null_repeats=0`, `block_size=2048` | RBF kernel, bandwidth = median distance among the real rows × multiplier; unbiased MMD², reported as `sign · sqrt`; 0 = the reference distribution, slightly negative values are noise. Fixed-kernel U-MMD² is unbiased under iid sampling; the signed root is not |
| `MultiScaleMMD` | `mmd_s{scale}`, `mmd_multiscale` | `scales=(0.25, 0.5, 1, 2, 4)` | `mmd_multiscale = signed_sqrt(mean_s MMD²_s)` |
| `KID` | `kid` (+ `kid_std`) | `degree=3`, `gamma=1/d`, `coef0=1`, `subsets=None`, `subset_size=1000`, `normalize=False` | polynomial-kernel MMD². Nearly constant on L2-normalized gpt2-large features, hence `normalize=False`; not validated for text |
| `FrechetDistance` | `fd`, `fd_mean_part` | | the FID formula on text features; n-biased and blind to uniform duplication |
| `PRDC` | `precision`, `recall`, `density`, `coverage`, `feature_dup_rate` | `k=5`, `duplicate_tol=1e-6` | Kynkäänniemi et al. 2019, Naeem et al. 2020; n-dependent |
| `FeatureVendi` | `vendi_per_n` | | single-sample |
| `TopicOccupancy` | `tv`, `min_ratio`, `n_under_half`, `n_over_double`, `share`, `share_real`, `ratio` | `n_clusters=16`, `seed=0`, `n_init=10` | k-means topics of the real rows; needs scikit-learn |
| `MAUVE` | `mauve`, `mauve_sd`, `per_seed`, `num_buckets` | `num_buckets="auto"` (= n/10), `seeds=(25, …, 29)`, `pca_max_data=-1`, `kmeans_explained_var=0.9`, `kmeans_num_redo=5`, `kmeans_max_iter=500`, `divergence_curve_discretization_size=25`, `mauve_scaling_factor=5` | features = mauve-text's (gpt2-large last token, fp32, special tokens removed) via `TextFeatureExtractor(pooling="last")`; needs `mauve-text`. Depends on n and k |

`TextFeatureExtractor(model="gpt2-large", device=None, pooling="mean", layer=-1, max_length=1024, batch_size=8,
autocast=<bf16 for mean, fp32 for last>, keep_special_tokens=<True for mean, False for last>, cache_dir=None,
memory_cache=16)`. Mean pooling averages the last hidden layer over all non-padding positions (texts keep special
tokens, as gen ppl scores them); last pooling reproduces mauve-text's featurization.

**Diagnostics** (localize a loss).

| Class | Returns | Hyperparameters (defaults) | Notes |
|---|---|---|---|
| `AttributePanel` | `retention`, `mean_ratio`, `min_retention`, `min_retention_attribute`, `detail` | `attributes=DEFAULT_ATTRIBUTES`, `percentile=90`, `count_documents=True` | per-row digits, non-ASCII, code characters, URLs, quotes, line breaks, capitalized words, word length, documents per row. Retention = share of generated rows above the real 90th percentile / share of real rows above it. Pass `{name: f(text) -> float}` for task-specific attributes |
| `ReverseNGramCE` | `reverse_ce_excess`, `ce_real`, `ce_generated`, `row_excess` | `order=4`, `discount=0.75`, `vocab_size=None` | Kneser-Ney LM fit on the samples vs on as many real rows, scored on held-out real rows (coverage in token space); pure Python, slow |

**Uncertainty** ([`condila.dlm_metrics.stats`](../src/condila/dlm_metrics/stats.py)): `split_half_sd`, `mmd_permutation_null`,
`bootstrap_se`, `real_floor`.


## 统计口径

固定核与独立 iid 样本下，无偏的是 U-MMD²。展示的 signed square root 不继承无偏性。
默认带宽由 scoring reference 的 pairwise median 选取；正式推断宜独立校准并冻结。
`null_sd` 是 permutation null 的离散程度，不是模型得分标准误或模型差异置信区间。
复制样本文件改变 U-statistic 的有限样本修正，不能视作新增独立生成。

现有数值算法保持原库实现，本次整合主要变更命名空间和模型接口。
