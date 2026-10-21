from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.grade_transformers_fresh_dev_v7 import TASK_IDS, _verify
from scripts.grade_transformers_phase_a_v1 import _scan_candidate


def _entry(content: bytes) -> dict[str, object]:
    return {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content), "mode": 0o644}


def _workspace(tmp_path: Path) -> tuple[dict[str, object], str]:
    target = "src/transformers/target.py"
    dependency = "src/transformers/dependency.py"
    target_bytes = b"value = 1\n"
    dependency_bytes = b"from typing import Any\n"
    public = b"assert True\n"
    root = tmp_path / "candidate"
    (root / "src/transformers").mkdir(parents=True)
    (root / target).write_bytes(target_bytes)
    (root / dependency).write_bytes(dependency_bytes)
    (root / "test_public.py").write_bytes(public)
    manifest: dict[str, object] = {
        "files": {
            target: _entry(target_bytes), dependency: _entry(dependency_bytes),
            "test_public.py": _entry(public),
        },
        "directories": {"src": 0o755, "src/transformers": 0o755},
    }
    return manifest, target


def test_v6_accepts_exact_baseline_inventory_and_target_edit(tmp_path: Path) -> None:
    manifest, target = _workspace(tmp_path)
    root = tmp_path / "candidate"
    (root / target).write_bytes(b"value = 2\n")
    files, file_count, directory_count = _scan_candidate(root, manifest, target)
    assert files[target] == b"value = 2\n"
    assert file_count == 3
    assert directory_count == 2


@pytest.mark.parametrize("attack", ["extra_module", "imported_dependency", "public_test_edit"])
def test_v6_rejects_unpermitted_workspace_changes(tmp_path: Path, attack: str) -> None:
    manifest, target = _workspace(tmp_path)
    root = tmp_path / "candidate"
    if attack == "extra_module":
        (root / "src/transformers/sitecustomize.py").write_text("pass\n")
    elif attack == "imported_dependency":
        (root / "src/transformers/dependency.py").write_text("from os import environ\n")
    else:
        (root / "test_public.py").write_text("print('forged')\n")
    with pytest.raises(ValueError):
        _scan_candidate(root, manifest, target)


def test_v6_rejects_symlink_and_path_escape(tmp_path: Path) -> None:
    manifest, target = _workspace(tmp_path)
    root = tmp_path / "candidate"
    (root / "src/transformers/escape.py").symlink_to("../../../../etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        _scan_candidate(root, manifest, target)


@pytest.mark.parametrize("task_index", range(len(TASK_IDS)))
def test_v6_host_verifier_accepts_only_calibrated_semantics(task_index: int) -> None:
    task_id = TASK_IDS[task_index]
    values = [
        {"num_tokens_scored": [1.0], "num_green_tokens": [1.0]},
        {"rope_type": "yarn", "factor": 4.0, "future_warning": True,
         "none_future_warning": True, "none_preserved": True},
        {"output_type": "BaseModelOutput", "hidden_states_type": "tuple",
         "last_hidden_state_shape": [1, 2, 4], "nested_values_preserved": True},
        {"default_mean": [0.48145466, 0.4578275, 0.40821073],
         "default_background": [122, 116, 104], "custom_background": [127, 63, 0]},
    ]
    assert _verify(task_id, values[task_index])[0]
    assert not _verify(task_id, {})[0]
