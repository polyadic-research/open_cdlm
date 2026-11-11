import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from condila.cdlm import create_model, from_pretrained, generate
from condila.cdlm.langflow import LangFlowLightningModule
from tests.test_cdlm_api import tiny_config


def assert_same_state(first, second):
    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("sharded", [False, True])
def test_hf_round_trip_with_bundled_class(tmp_path, sharded):
    model = create_model("langflow", tiny_config("langflow")).eval()
    model.save_pretrained(tmp_path, max_shard_size="2KB" if sharded else "5GB")
    loaded = from_pretrained("langflow", tmp_path, local_files_only=True)
    assert type(loaded) is type(model) and not loaded.training
    assert_same_state(model, loaded)
    torch.testing.assert_close(generate(model, steps=2), generate(loaded, steps=2))


def test_legacy_safetensors_and_missing_architecture(tmp_path):
    model = create_model("langflow", tiny_config("langflow"))
    state = dict(model.state_dict())
    state["proposal.scale"] = model.proposal.scale.detach().clone()
    del state["proposal.raw_scale"]
    checkpoint = tmp_path / "legacy.safetensors"
    save_file(state, str(checkpoint))
    with pytest.raises(ValueError, match="architecture"):
        from_pretrained("langflow", checkpoint)
    loaded = from_pretrained("langflow", checkpoint, config=model.config)
    torch.testing.assert_close(model.proposal.scale, loaded.proposal.scale)
    model.config.save_pretrained(tmp_path)
    loaded = from_pretrained("langflow", checkpoint)
    torch.testing.assert_close(model.proposal.scale, loaded.proposal.scale)


@pytest.mark.parametrize("ema_format", ["separate", "prefixed"])
def test_lightning_checkpoint_selects_raw_or_ema_and_records_architecture(tmp_path, ema_format):
    raw = create_model("langflow", tiny_config("langflow"))
    ema = create_model("langflow", tiny_config("langflow"))
    state = {"model." + key: value for key, value in raw.state_dict().items()}
    payload = {"state_dict": state, "condila_model_config": raw.config.to_dict()}
    if ema_format == "separate":
        payload["ema_state_dict"] = ema.state_dict()
    else:
        state.update({"ema_model." + key: value for key, value in ema.state_dict().items()})
    path = tmp_path / "model.ckpt"
    torch.save(payload, path)
    assert_same_state(ema, from_pretrained("langflow", path))
    assert_same_state(raw, from_pretrained("langflow", path, weights="raw"))
    with pytest.raises(ValueError, match="weights"):
        from_pretrained("langflow", path, weights="other")


def test_lightning_saves_its_actual_explicit_config():
    module = LangFlowLightningModule(config=tiny_config("langflow"))
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["condila_model_config"]["hidden_size"] == 16
    assert checkpoint["condila_model_config"]["vocab_size"] == 18


def test_missing_local_checkpoint_is_not_interpreted_as_hub_id(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_pretrained("langflow", tmp_path / "missing.safetensors")


def test_langflow_token_cli_integrates_with_metric_sample_reader(tmp_path):
    from condila.dlm_metrics import load_samples

    model = create_model("langflow", tiny_config("langflow"))
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    paths = []
    for batch_size in (1, 2):
        output = tmp_path / f"samples-{batch_size}.pt"
        result = subprocess.run(
            [sys.executable, "-m", "condila.cdlm.langflow.inference", "--checkpoint", str(checkpoint),
             "--output", str(output), "--num-samples", "3", "--batch-size", str(batch_size),
             "--steps", "2", "--seed", "8", "--device", "cpu"],
            check=False, cwd=tmp_path, text=True, capture_output=True, timeout=60,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        samples = load_samples(str(output))
        assert len(samples) == 3 and samples.lengths() == [8, 8, 8]
        metadata = json.loads(Path(str(output) + ".meta.json").read_text())
        assert metadata["steps"] == 2
        paths.append(torch.load(output, weights_only=True)["ids"])
    torch.testing.assert_close(*paths, rtol=0, atol=0)
