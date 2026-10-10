from pathlib import Path
import hashlib
import json

import pytest

from scripts.gpu_gate_venv_v6 import EXTERNAL_MODULES, validate_isolated_environment


def _module_paths(root: Path) -> dict[str, str]:
    return {name: str(root / "lib" / "python3.12" / "site-packages" / name / "__init__.py")
            for name in EXTERNAL_MODULES}


def test_isolated_venv_provenance_accepts_venv_modules_and_stdlib_paths(tmp_path):
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


def test_isolated_venv_rejects_system_site_packages_on_sys_path(tmp_path):
    root = tmp_path / "venv"
    with pytest.raises(SystemExit, match="global package path"):
        validate_isolated_environment(
            prefix=str(root),
            base_prefix="/usr/local",
            sys_path=[str(root / "lib/python3.12/site-packages"),
                      "/usr/local/lib/python3.12/site-packages"],
            module_paths=_module_paths(root),
        )


def test_isolated_venv_rejects_imported_module_from_global_site_packages(tmp_path):
    root = tmp_path / "venv"
    modules = _module_paths(root)
    modules["trl"] = "/usr/local/lib/python3.12/site-packages/trl/__init__.py"
    with pytest.raises(SystemExit, match="imported outside"):
        validate_isolated_environment(
            prefix=str(root),
            base_prefix="/usr/local",
            sys_path=[str(root / "lib/python3.12/site-packages")],
            module_paths=modules,
        )


def test_isolated_venv_rejects_base_interpreter(tmp_path):
    with pytest.raises(SystemExit, match="not running inside"):
        validate_isolated_environment(
            prefix="/usr/local",
            base_prefix="/usr/local",
            sys_path=[],
            module_paths=_module_paths(tmp_path / "venv"),
        )


def test_v6_lock_preserves_frozen_v5_scientific_conditions_and_hashes():
    root = Path(__file__).resolve().parents[1]
    v5 = json.loads((root / "protocols/qwen25_deepmath_grpo_math500_v5.lock.json").read_text())
    v6 = json.loads((root / "protocols/qwen25_deepmath_grpo_math500_v6.lock.json").read_text())
    assert v6["protocol_id"] == "qwen25_deepmath_grpo_math500_v6"
    for field in ("model", "training_data", "evaluation_data", "reward_and_independent_metric"):
        assert v6[field] == v5[field]
    for field in ("generation", "resource_budget", "pass_criteria", "budget"):
        assert v6["first_gpu_gate"][field] == v5["first_gpu_gate"][field]
    for relative_path, expected_hash in v6["execution_code"]["files_sha256"].items():
        observed = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        assert observed == expected_hash, relative_path
