from __future__ import annotations

import pytest

from vare.mlxlm_1971_repo_task import MlxLm1971TaskError
from vare.mlxlm_1971_repo_task_v3 import verify_protocol_v3


def test_v3_is_historical_and_fails_closed_after_public_entrypoint_switch():
    with pytest.raises(MlxLm1971TaskError, match="implementation hash mismatch"):
        verify_protocol_v3()
