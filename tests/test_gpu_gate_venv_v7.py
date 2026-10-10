from pathlib import Path
import hashlib
import json

import pytest

from scripts.gpu_gate_venv_v7 import (
    EXPECTED_TARGET_PIP_VERSION,
    EXTERNAL_MODULES,
    validate_isolated_environment,
    validate_pip_bootstrap_identity,
)


def _module_paths(root: Path) -> dict[str, str]:
    return {name: str(root / "lib" / "python3.12" / "site-packages" / name / "__init__.py")
            for name in EXTERNAL_MODULES}


def test_v7_environment_provenance_accepts_only_venv_package_paths(tmp_path):
    root = tmp_path / "venv"
    modules = _module_paths(root)
    record = validate_isolated_environment(
        prefix=str(root),
        base_prefix="/usr/local",
        sys_path=[str(root / "lib/python3.12/site-packages"), "/usr/local/lib/python3.12"],
        module_paths=modules,
    )
    assert record["sys_prefix"] == str(root.resolve())
    assert record["sys_base_prefix"] == "/usr/local"
    assert record["system_site_packages"] is False
    assert set(record["module_files"]) == set(EXTERNAL_MODULES)


def test_v7_isolation_rejects_host_packages_and_host_imports(tmp_path):
    root = tmp_path / "venv"
    with pytest.raises(SystemExit, match="global package path"):
        validate_isolated_environment(
            prefix=str(root), base_prefix="/usr/local",
            sys_path=[str(root / "lib/python3.12/site-packages"),
                      "/usr/local/lib/python3.12/site-packages"],
            module_paths=_module_paths(root),
        )
    modules = _module_paths(root)
    modules["pip"] = "/usr/local/lib/python3.12/site-packages/pip/__init__.py"
    with pytest.raises(SystemExit, match="imported outside"):
        validate_isolated_environment(
            prefix=str(root), base_prefix="/usr/local",
            sys_path=[str(root / "lib/python3.12/site-packages")], module_paths=modules,
        )


def test_v7_isolation_rejects_base_interpreter(tmp_path):
    with pytest.raises(SystemExit, match="not running inside"):
        validate_isolated_environment(
            prefix="/usr/local", base_prefix="/usr/local", sys_path=[],
            module_paths=_module_paths(tmp_path / "venv"),
        )


def test_v7_bootstrap_binds_host_pip_option_and_exact_target_version(tmp_path):
    root = tmp_path / "venv"
    module = root / "lib/python3.12/site-packages/pip/__init__.py"
    record = validate_pip_bootstrap_identity(
        prefix=str(root), host_pip_version="24.1.2",
        target_pip_version=EXPECTED_TARGET_PIP_VERSION, target_pip_module=str(module),
    )
    assert record["method"] == "python -m pip --python <venv> install pip"
    assert record["host_pip_version"] == "24.1.2"
    assert record["target_pip_version"] == EXPECTED_TARGET_PIP_VERSION


@pytest.mark.parametrize(
    ("host", "target", "module", "message"),
    [
        ("22.2", "26.2.1", "/tmp/venv/lib/python3.12/site-packages/pip/__init__.py", "lacks"),
        ("24.1.2", "26.2.2", "/tmp/venv/lib/python3.12/site-packages/pip/__init__.py", "must be"),
        ("24.1.2", "26.2.1", "/usr/local/lib/python3.12/site-packages/pip/__init__.py", "outside"),
    ],
)
def test_v7_bootstrap_fails_closed_on_version_or_path_mismatch(tmp_path, host, target, module, message):
    root = tmp_path / "venv"
    module_path = module.replace("/tmp/venv", str(root)) if module.startswith("/tmp/venv") else module
    with pytest.raises(SystemExit, match=message):
        validate_pip_bootstrap_identity(
            prefix=str(root), host_pip_version=host, target_pip_version=target,
            target_pip_module=module_path,
        )


def test_v7_lock_preserves_all_v6_scientific_conditions_and_hashes():
    root = Path(__file__).resolve().parents[1]
    v6 = json.loads((root / "protocols/qwen25_deepmath_grpo_math500_v6.lock.json").read_text())
    v7 = json.loads((root / "protocols/qwen25_deepmath_grpo_math500_v7.lock.json").read_text())
    assert v7["protocol_id"] == "qwen25_deepmath_grpo_math500_v7"
    for field in ("model", "training_data", "evaluation_data", "reward_and_independent_metric",
                  "first_gpu_gate", "conditional_followup_if_gate_passes"):
        assert v7[field] == v6[field]
    assert (root / "requirements/math500-study-gpu-gate-v7.txt").read_bytes() == (
        root / "requirements/math500-study-gpu-gate-v6.txt").read_bytes()
    assert v7["environment_isolation"]["pip_bootstrap"]["target_pip_version_required"] == "26.2.1"
    for relative_path, expected_hash in v7["execution_code"]["files_sha256"].items():
        observed = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        assert observed == expected_hash, relative_path
