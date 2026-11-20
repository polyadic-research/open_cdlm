"""Kernel Text Discrepancy with independently calibrated projected token features."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..features import KTD_QUANTILE_DIM
from ..functional.kernel import median_bandwidth, mmd2_rbf
from .feature import FeatureMetric


class KTD(FeatureMetric):
    """Kernel Text Discrepancy on projected token quantiles and the final token state.

    Call ``fit(calibration)`` on independent real texts, then use the standard
    ``update(..., real=True/False)``, ``compute()``, and ``reset()`` methods. Reset
    keeps the calibration. The default encoder is frozen GPT-2-large.

    Precomputed features for ``fit_features()`` and ``update_features()`` have
    8192 raw quantile columns followed by the raw final-token hidden state.
    The score in ``ktd`` is the unbiased U-statistic on the squared MMD scale.
    """

    _pooling = "ktd"
    higher_is_better = False

    def __init__(self, block_size: int = 512, **kwargs):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        super().__init__(normalize=False, **kwargs)
        self.block_size = block_size
        self.calibration: dict[str, float | int] | None = None

    @staticmethod
    def _split(features):
        rows = np.asarray(features, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] <= KTD_QUANTILE_DIM or len(rows) == 0:
            raise ValueError("KTD features need 8192 quantile columns followed by final-token states")
        if not np.isfinite(rows).all():
            raise ValueError("KTD features must be finite")
        quantiles, last = rows[:, :KTD_QUANTILE_DIM], rows[:, KTD_QUANTILE_DIM:]
        norms = np.linalg.norm(last, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 1e-12).any():
            raise ValueError("KTD final-token states must have a finite nonzero norm")
        return quantiles, last / norms

    @staticmethod
    def _compose(quantiles, last, calibration):
        return np.concatenate([
            (quantiles / calibration["quantile_scale"]) * np.sqrt(0.75),
            (last / calibration["last_scale"]) * np.sqrt(0.25),
        ], axis=1).astype(np.float32)

    @staticmethod
    def _median_scale(features):
        scale = median_bandwidth(features, lower=True)
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("KTD calibration needs real samples with nonzero median distances")
        return scale

    def fit(self, calibration: Any) -> KTD:
        """Fit feature scales and bandwidth on a separate real calibration corpus."""
        return self.fit_features(self.extractor(self._wrap(calibration)))

    def fit_features(self, features: np.ndarray) -> KTD:
        """Calibrate using precomputed raw features from the same encoder and preprocessing."""
        quantiles, last = self._split(features)
        if len(quantiles) < 2:
            raise ValueError("KTD needs at least two calibration samples")
        calibration = {
            "quantile_scale": self._median_scale(quantiles),
            "last_scale": self._median_scale(last),
            "last_dimension": last.shape[1],
            "calibration_rows": len(last),
        }
        calibration["bandwidth"] = 0.5 * self._median_scale(self._compose(quantiles, last, calibration))
        self.calibration = calibration
        return self

    def state_dict(self) -> dict[str, Any]:
        calibration = self.calibration
        return {**super().state_dict(), "calibration": None if calibration is None else dict(calibration)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        calibration = state.get("calibration")
        self.calibration = None if calibration is None else dict(calibration)

    def _compute(self) -> dict[str, Any]:
        calibration = self.calibration
        if calibration is None:
            raise RuntimeError("Fit KTD on independent real calibration texts before computing a score")
        reference, generated = self._xy()
        ref_quantiles, ref_last = self._split(reference)
        gen_quantiles, gen_last = self._split(generated)
        if ref_last.shape[1] != calibration["last_dimension"] or gen_last.shape[1] != calibration["last_dimension"]:
            raise ValueError("KTD score features must match the calibration feature dimension")
        reference = self._compose(ref_quantiles, ref_last, calibration)
        generated = self._compose(gen_quantiles, gen_last, calibration)
        bandwidth = float(calibration["bandwidth"])
        value = float(mmd2_rbf(reference, generated, [bandwidth], self.block_size)[0])
        return {"ktd": value, "bandwidth": bandwidth, "n_real": len(reference), "n_fake": len(generated)}
