"""dlm-metrics: evaluation metrics for diffusion language models (and any text generator).

Modular (torchmetrics-style) metrics::

    from open_cdlm.dlm_metrics import KTD, MTD, GenerativePerplexity, EntropyDecomposition, TextFeatureExtractor

Functional versions live in :mod:`open_cdlm.dlm_metrics.functional`; likelihood bounds for continuous DLMs in
:mod:`open_cdlm.dlm_metrics.likelihood`; the one-call recommended protocol is :func:`open_cdlm.dlm_metrics.evaluate`.
"""
from .base import Metric, MetricCollection
from .features import TextFeatureExtractor
from .inputs import PRESETS, Samples, load_samples
from .metrics import (KID, KTD, MAUVE, MMD, MTD, PRDC, AttributePanel, CompressionRate, DistinctN,
                      EntropyDecomposition, ExactDuplicateRate, FeatureVendi, FrechetDistance,
                      GenerativePerplexity, JSDRows, MultiScaleMMD, MultiScaleMTD, NGramDiversity,
                      PooledEntropy, RepetitionRate, ReverseNGramCE, SampleEntropy, TopicOccupancy)
from .models import JudgeLM, get_judge
from .suite import Report, evaluate

__version__ = "0.1.0"

__all__ = [
    "Metric", "MetricCollection", "TextFeatureExtractor", "PRESETS", "Samples", "load_samples",
    "KID", "KTD", "MAUVE", "MTD", "MMD", "PRDC", "AttributePanel", "CompressionRate", "DistinctN",
    "EntropyDecomposition", "ExactDuplicateRate", "FeatureVendi", "FrechetDistance",
    "GenerativePerplexity", "JSDRows", "MultiScaleMTD", "MultiScaleMMD", "NGramDiversity",
    "PooledEntropy", "RepetitionRate", "ReverseNGramCE", "SampleEntropy", "TopicOccupancy", "JudgeLM", "get_judge",
    "Report", "evaluate", "__version__",
]
