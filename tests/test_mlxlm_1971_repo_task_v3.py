from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vare.mlxlm_1971_repo_task_v3 import (
    EXPECTED_OUTPUT,
    MlxLm1971TaskError,
    grade_workspace_v3,
    verify_protocol_v3,
)


def test_v3_protocol_covers_multiple_scales_and_cached_values():
    protocol = verify_protocol_v3()
    expected = json.loads(EXPECTED_OUTPUT.read_text(encoding="utf-8"))
    assert protocol["value_scales"] == [0.25, 0.5, 0.8]
    assert len(set(protocol["value_scales"])) == 3
    assert len(expected["cases"]) == 3
    for case in expected["cases"]:
        assert set(case) == {
            "full_reference",
            "full_configured",
            "incremental_reference",
            "incremental_configured",
        }
        assert "cache_state" in case["incremental_reference"]
        assert "cache_state" in case["incremental_configured"]


def test_v3_protocol_and_implementation_hashes_are_pinned():
    protocol = verify_protocol_v3()
    assert protocol["protocol_id"] == "vare.mlsys-repository-task.mlxlm-1971.v3"
    assert protocol["expected_output_sha256"]
    assert protocol["implementation_sha256"]
    task_root = Path(__file__).resolve().parents[1] / (
        "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
    )
    v1_protocol = task_root / "protocol-v1.json"
    v1_expected = task_root / "expected-output-v1.json"
    assert hashlib.sha256(v1_protocol.read_bytes()).hexdigest() == protocol[
        "v1_preservation"
    ]["protocol_sha256"]
    assert hashlib.sha256(v1_expected.read_bytes()).hexdigest() == protocol[
        "v1_preservation"
    ]["expected_output_sha256"]


@pytest.mark.parametrize(
    ("image", "timeout_s", "memory_mb"),
    [
        ("other-image", 30, 1536),
        (None, 31, 1536),
        (None, 30, 2048),
    ],
)
def test_v3_rejects_altered_runtime_limits(tmp_path, image, timeout_s, memory_mb):
    protocol = verify_protocol_v3()
    with pytest.raises(MlxLm1971TaskError):
        grade_workspace_v3(
            tmp_path,
            image=image if image is not None else protocol["runtime"]["image"],
            timeout_s=timeout_s,
            memory_mb=memory_mb,
        )


def test_v3_rejects_symlinked_candidate_target(tmp_path):
    workspace = tmp_path / "candidate"
    target = workspace / "mlx_lm/models/mimo_v2_flash.py"
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    target.symlink_to(outside)
    with pytest.raises(MlxLm1971TaskError, match="unsafe"):
        grade_workspace_v3(workspace)


def test_v3_rejects_mutated_protocol_bytes(monkeypatch):
    import vare.mlxlm_1971_repo_task_v3 as task

    original = task.sha256_file

    def changed(path: Path) -> str:
        if path.name == "protocol-v3.json":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(task, "sha256_file", changed)
    with pytest.raises(MlxLm1971TaskError, match="protocol hash lock"):
        task.verify_protocol_v3()
