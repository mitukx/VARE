"""Historical v4 is immutable and must fail closed after the v9 dispatch."""

from __future__ import annotations

import pytest

from vare.mlxlm_1971_repo_task import MlxLm1971TaskError
from vare.mlxlm_1971_repo_task_v4 import verify_protocol_v4


def test_frozen_v4_protocol_rejects_current_dispatcher_bytes():
    with pytest.raises(MlxLm1971TaskError, match="v4 implementation hash mismatch"):
        verify_protocol_v4()
