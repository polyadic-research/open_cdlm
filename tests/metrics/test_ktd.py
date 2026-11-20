import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_cdlm.dlm_metrics import KTD, evaluate, suite
from open_cdlm.dlm_metrics.cli import cmd_eval
from open_cdlm.dlm_metrics.features import KTD_QUANTILE_DIM, _quantile_bin_means


def test_integrated_quantiles_with_fractional_bin_boundaries():
    states = torch.tensor([[1.0, 4.0], [5.0, 12.0], [3.0, 8.0]])
    expected = torch.tensor([[5 / 3, 13 / 3], [16 / 3, 32 / 3]])
    torch.testing.assert_close(_quantile_bin_means(states, 2), expected)
    torch.testing.assert_close(_quantile_bin_means(states[:1], 64), states[:1].T.expand(-1, 64))


def test_ktd_calibration_and_raw_score_survive_state_round_trip():
    calibration = np.zeros((4, KTD_QUANTILE_DIM + 2), dtype=np.float32)
    calibration[:, 0] = [0, 1, 4, 9]
    calibration[:, -2:] = [[2, 0], [0, 3], [-4, 0], [0, -5]]
    with pytest.raises(RuntimeError, match="Fit KTD"):
        KTD().compute()
    ktd = KTD(reset_real_features=False).fit_features(calibration)
    assert ktd.calibration["quantile_scale"] == 4.0
    assert ktd.calibration["last_scale"] == pytest.approx(np.sqrt(2))

    reference = calibration[[0, 3]].copy()
    reference[:, 0] += 0.5
    ktd.update_features(reference, real=True)
    restored = KTD(reset_real_features=False)
    restored.load_state_dict(ktd.state_dict())
    restored.reset()
    restored.update_features(reference[::-1])

    # With two documents and an identical generated multiset, the U-statistic is k(x1, x2) - 1.
    squared_distance = 0.75 * (9.0 / 4.0) ** 2 + 0.25
    bandwidth = restored.calibration["bandwidth"]
    expected = np.exp(-squared_distance / (2 * bandwidth**2)) - 1.0
    assert restored.compute()["ktd"] == pytest.approx(expected, abs=1e-7)


def test_cli_reuses_calibration_across_sample_sizes(tmp_path, monkeypatch, capsys):
    rows_by_file = {
        "reference": ["11", "12", "14", "15", "18", "19"],
        "calibration": ["1", "2", "4", "9"],
        "short": ["21", "22"],
        "long": ["31", "33", "37"],
    }
    for filename, rows in rows_by_file.items():
        (tmp_path / f"{filename}.json").write_text(json.dumps(rows))
    calibration_calls = []

    class StubExtractor:
        def __init__(self, judge, *, pooling, **kwargs):
            self.pooling = pooling

        def __call__(self, samples):
            texts = samples.texts()
            values = np.asarray([float(text) for text in texts], dtype=np.float32)
            last = np.column_stack((values, np.ones_like(values)))
            if self.pooling == "mean":
                return last
            if texts == rows_by_file["calibration"]:
                calibration_calls.append(texts)
            quantiles = np.zeros((len(values), KTD_QUANTILE_DIM), dtype=np.float32)
            quantiles[:, 0] = values
            return np.concatenate((quantiles, last), axis=1)

    monkeypatch.setattr(suite, "TextFeatureExtractor", StubExtractor)
    monkeypatch.setattr(suite, "get_judge", lambda *args: SimpleNamespace(name="stub"))
    with pytest.raises(ValueError, match="separate real calibration corpus"):
        evaluate(rows_by_file["short"], rows_by_file["reference"], metrics="ktd")

    path = tmp_path / "report.json"
    cmd_eval([
        str(tmp_path / "short.json"), str(tmp_path / "long.json"),
        "--reference", str(tmp_path / "reference.json"), "--calibration", str(tmp_path / "calibration.json"),
        "--metrics", "mtd,ktd", "--mtd-null-repeats", "0", "--out", str(path),
    ])
    report = json.loads(path.read_text())
    bandwidth = report["settings"]["ktd_calibration"]["bandwidth"]
    assert calibration_calls == [rows_by_file["calibration"]]
    for name, size in (("short", 2), ("long", 3)):
        results = report["results"][name]
        assert results["mtd"]["n_fake"] == size
        assert results["ktd"]["n_real"] == results["ktd"]["n_fake"] == size
        assert results["ktd"]["bandwidth"] == report["floors"][str(size)]["ktd"]["bandwidth"] == bandwidth
    assert "| MTD | KTD |" in capsys.readouterr().out
