import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_root_and_model_discovery_do_not_import_heavy_dependencies(tmp_path):
    code = """
import sys
import condila
from condila.cdlm import list_models
assert list_models() == ('langflow', 'replaid')
assert condila.__version__ == '0.1.0'
assert not {'torch', 'transformers', 'lightning', 'datasets'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_compatibility_spelling_shares_all_module_and_class_objects():
    from condia.dlm_metrics.base import Metric as alias_metric
    from condia.dlm_metrics.functional.kernel import mmd2_rbf as alias_mmd
    from condia.dlm_metrics.likelihood import LangFlowAdapter as alias_adapter

    import condia.dlm_metrics as alias
    import condila.dlm_metrics as canonical
    from condila.dlm_metrics.base import Metric
    from condila.dlm_metrics.functional.kernel import mmd2_rbf
    from condila.dlm_metrics.likelihood import LangFlowAdapter

    assert alias is canonical
    assert alias_metric is Metric
    assert alias_mmd is mmd2_rbf
    assert alias_adapter is LangFlowAdapter
    assert canonical.MMD.__module__ == "condila.dlm_metrics.metrics.feature"


@pytest.mark.parametrize("module", [
    "condila", "condila.cdlm.langflow", "condila.cdlm.replaid", "condia",
])
def test_unknown_attributes_fail_normally(module):
    from importlib import import_module

    with pytest.raises(AttributeError):
        _ = import_module(module).nonexistent_attribute


def test_public_namespaces_do_not_import_original_projects():
    from condila.cdlm import LangFlow, LangFlowConfig, RePlaid, RePlaidConfig

    assert LangFlow.__module__.startswith("condila.cdlm.langflow")
    assert RePlaid.__module__.startswith("condila.cdlm.replaid")
    assert LangFlowConfig.__module__.startswith("condila.")
    assert RePlaidConfig.__module__.startswith("condila.")
    assert not {"langflow", "duo", "replaid_original", "dlm_metrics"} & sys.modules.keys()


@pytest.mark.parametrize("command", [
    ["condila", "--help"],
    ["condila", "metrics", "--help"],
    ["condila.cdlm.langflow.cli", "--cfg", "job"],
    ["condila.cdlm.langflow.inference", "--help"],
    ["condila.cdlm.replaid.prepare", "--help"],
    ["condila.cdlm.replaid.sample", "--help"],
    ["condila.cdlm.replaid.evaluate", "--help"],
    ["condila.cdlm.replaid.score", "--help"],
    ["condila.cdlm.replaid.reference", "--help"],
])
def test_cli_entrypoints_work_outside_repository(command, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", *command], check=False, cwd=tmp_path, text=True, capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if "--cfg" in command:
        assert "use_wandb: false" in result.stdout


def test_default_replaid_config_is_packaged_and_independent_of_cwd(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "condila.cdlm.replaid.train", "--dry-run", "--device", "cpu"],
        check=False, cwd=tmp_path, text=True, capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"parameters": 131909219' in result.stdout


def test_no_original_project_paths_or_runtime_path_injection():
    import condila

    for path in Path(condila.__file__).parent.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in source, path
        for prefix in ("from langflow ", "from replaid_original ", "from dlm_metrics ", "from duo "):
            assert prefix not in source, path

