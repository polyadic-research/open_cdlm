"""dlm-metrics: evaluation metrics for diffusion language models (and any text generator).

Modular (torchmetrics-style) metrics::

    from condila.dlm_metrics import MMD, GenerativePerplexity, EntropyDecomposition, TextFeatureExtractor

Functional versions live in :mod:`condila.dlm_metrics.functional`; likelihood bounds for continuous DLMs in
:mod:`condila.dlm_metrics.likelihood`; the one-call recommended protocol is :func:`condila.dlm_metrics.evaluate`.
"""
from .base import Metric, MetricCollection
from .features import TextFeatureExtractor
from .inputs import PRESETS, Samples, load_samples
from .metrics import (KID, MAUVE, MMD, PRDC, AttributePanel, CompressionRate, DistinctN, EntropyDecomposition,
                      ExactDuplicateRate, FeatureVendi, FrechetDistance, GenerativePerplexity, JSDRows, MultiScaleMMD,
                      NGramDiversity, PooledEntropy, RepetitionRate, ReverseNGramCE, SampleEntropy, TopicOccupancy)
from .models import JudgeLM, get_judge
from .suite import Report, evaluate

__version__ = "0.1.0"

__all__ = [
    "Metric", "MetricCollection", "TextFeatureExtractor", "PRESETS", "Samples", "load_samples", "KID", "MAUVE", "MMD",
    "PRDC", "AttributePanel", "CompressionRate", "DistinctN", "EntropyDecomposition", "ExactDuplicateRate",
    "FeatureVendi", "FrechetDistance", "GenerativePerplexity", "JSDRows", "MultiScaleMMD", "NGramDiversity",
    "PooledEntropy", "RepetitionRate", "ReverseNGramCE", "SampleEntropy", "TopicOccupancy", "JudgeLM", "get_judge",
    "Report", "evaluate", "__version__",
]
