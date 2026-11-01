"""Parent-owned function oracle for one exposed Suor/funcy Train task.

The model receives the Train issue and repository source only. This grader
keeps its four expected outputs in the parent and runs the candidate module in
the isolated function-call harness. It does not load repository tests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from vare.isolated_candidate import FunctionCall
from scripts.long_horizon_eval import FunctionOracleGrader

TASK_ID = "Suor__funcy.207a7810.func_basic__jtj6rhfx"
TARGET_FILE = "funcy/tree.py"
CONTAINER_IMAGE = "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"
CALL_MODULE = "vare_oracle_target"
WRAPPER_SOURCE = "from funcy.tree import ltree_leaves\n"


def _support_source(env: Any, relative_path: str) -> str:
    commit = str(env.commit)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("the repository environment must pin a full commit SHA")
    result = subprocess.run(
        ["git", "-C", str(env.path), "show", f"{commit}:{relative_path}"],
        check=True, capture_output=True, timeout=20,
    )
    return result.stdout.decode("utf-8")


def grader_factory(record: Mapping[str, Any], env: Any) -> FunctionOracleGrader:
    """Construct the fixed oracle only for the preregistered Train task."""
    if (record.get("instance_id") != TASK_ID or record.get("upstream_repo") != "Suor/funcy"
            or record.get("split") != "train" or record.get("target_file") != TARGET_FILE
            or getattr(env, "upstream", None) != "Suor/funcy"):
        raise ValueError("the funcy oracle is scoped to its one frozen Train task")
    calls = [
        FunctionCall("scalar_root", CALL_MODULE, "ltree_leaves", [7], [7]),
        FunctionCall("nested_lists", CALL_MODULE, "ltree_leaves", [[[1, 2], 3]], [1, 2, 3]),
        FunctionCall("false_and_none_leaves", CALL_MODULE, "ltree_leaves",
                     [[[], [0, [False, None]]]], [0, False, None]),
        FunctionCall("empty_tree", CALL_MODULE, "ltree_leaves", [[]], []),
    ]
    support_files = {
        "funcy/__init__.py": "",
        "funcy/types.py": _support_source(env, "funcy/types.py"),
    }
    source_path = Path(__file__)
    return FunctionOracleGrader(
        task_id=TASK_ID, target_file=TARGET_FILE, calls=calls,
        image=CONTAINER_IMAGE, timeout_s=5, memory_mb=512,
        support_files=support_files,
        module=CALL_MODULE, wrapper_source=WRAPPER_SOURCE,
        additional_source_hashes={
            "scripts/funcy_tree_oracle.py": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
    )
