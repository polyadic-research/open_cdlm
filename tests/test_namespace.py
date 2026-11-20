import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_root_and_model_discovery_do_not_import_heavy_dependencies(tmp_path):
    code = """
import sys
import open_cdlm
from open_cdlm.cdlm import list_models
assert list_models() == ('langflow', 'replaid')
assert open_cdlm.__version__ == '0.1.0'
assert not {'torch', 'transformers', 'lightning', 'datasets'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_metrics_use_the_open_cdlm_namespace():
    from open_cdlm.dlm_metrics import MMD

    assert MMD.__module__ == "open_cdlm.dlm_metrics.metrics.feature"


@pytest.mark.parametrize("module", [
    "open_cdlm", "open_cdlm.cdlm.langflow", "open_cdlm.cdlm.replaid",
])
def test_unknown_attributes_fail_normally(module):
    from importlib import import_module

    with pytest.raises(AttributeError):
        _ = import_module(module).nonexistent_attribute


def test_public_namespaces_do_not_import_original_projects():
    from open_cdlm.cdlm import LangFlow, LangFlowConfig, RePlaid, RePlaidConfig

    assert LangFlow.__module__.startswith("open_cdlm.cdlm.langflow")
    assert RePlaid.__module__.startswith("open_cdlm.cdlm.replaid")
    assert LangFlowConfig.__module__.startswith("open_cdlm.")
    assert RePlaidConfig.__module__.startswith("open_cdlm.")
    assert not {"langflow", "duo", "replaid_original", "dlm_metrics"} & sys.modules.keys()


@pytest.mark.parametrize("command", [
    ["open_cdlm", "--help"],
    ["open_cdlm", "metrics", "--help"],
    ["open_cdlm", "train", "--help"],
    ["open_cdlm.cdlm.langflow.cli", "--dry-run", "--device", "cpu"],
    ["open_cdlm.cdlm.langflow.inference", "--help"],
    ["open_cdlm.cdlm.replaid.prepare", "--help"],
    ["open_cdlm.cdlm.replaid.sample", "--help"],
    ["open_cdlm.cdlm.replaid.evaluate", "--help"],
    ["open_cdlm.cdlm.replaid.score", "--help"],
    ["open_cdlm.cdlm.replaid.reference", "--help"],
])
def test_cli_entrypoints_work_outside_repository(command, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", *command], check=False, cwd=tmp_path, text=True, capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if "--dry-run" in command:
        assert '"experiment_id": "LANGFLOW_OWT"' in result.stdout


def test_default_replaid_config_is_packaged_and_independent_of_cwd(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "open_cdlm.cdlm.replaid.train", "--dry-run", "--device", "cpu"],
        check=False, cwd=tmp_path, text=True, capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"parameters": 131909219' in result.stdout


def test_no_original_project_paths_or_runtime_path_injection():
    import open_cdlm

    for path in Path(open_cdlm.__file__).parent.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in source, path
        for prefix in ("from langflow ", "from replaid_original ", "from dlm_metrics ", "from duo "):
            assert prefix not in source, path

