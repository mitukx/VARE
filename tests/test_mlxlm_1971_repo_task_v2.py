from __future__ import annotations

import json
from pathlib import Path

import pytest

from vare.mlxlm_1971_repo_task import MlxLm1971TaskError
from vare.mlxlm_1971_repo_task_v2 import verify_protocol_v2


def test_v2_is_historical_and_fails_closed_after_shared_runtime_changed():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v2.json").read_text())
    assert protocol["protocol_id"] == "vare.mlsys-repository-task.mlxlm-1971.v2"
    with pytest.raises(MlxLm1971TaskError, match="implementation hash mismatch"):
        verify_protocol_v2()
