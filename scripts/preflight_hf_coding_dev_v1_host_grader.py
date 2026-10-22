"""Fail fast on the host-side PyTorch oracle dependency for the HF v1 grader."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 12)
EXPECTED_TORCH = "2.9.1"
TASK_ID = "trl-6625-entropy-gradient-v1"
sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight() -> dict[str, Any]:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"host grader requires Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}; "
            f"found {platform.python_version()} at {sys.executable}"
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "host grader requires torch==2.9.1; install/use the isolated host-grader environment"
        ) from exc
    actual_torch = importlib.metadata.version("torch")
    if actual_torch != EXPECTED_TORCH:
        raise RuntimeError(f"host grader requires torch=={EXPECTED_TORCH}; found {actual_torch}")
    if not hasattr(torch, "nn") or not hasattr(torch.nn, "functional"):
        raise RuntimeError("PyTorch functional API is unavailable")

    from scripts.grade_hf_coding_dev_v1 import _cases

    cases = _cases(TASK_ID)
    if len(cases) != 3 or any(not case.expected for case in cases):
        raise RuntimeError("the pinned TRL entropy oracle did not produce all three expected cases")
    return {
        "schema_version": 1,
        "decision": "GO",
        "purpose": "host-side independent grader dependency check; no model weights loaded and no inference",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": actual_torch,
        "torch_path": torch.__file__,
        "torch_device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "task_id": TASK_ID,
        "oracle_cases_computed": len(cases),
        "grader_sha256": _sha256(ROOT / "scripts/grade_hf_coding_dev_v1.py"),
        "expected_grader_sha256": "0fc3e7902bb94cda6df6032f19eb225d77f3e4c8c0004a697c836aa3afa6c9f3",
        "grader_hash_matches_frozen_v1": _sha256(ROOT / "scripts/grade_hf_coding_dev_v1.py")
        == "0fc3e7902bb94cda6df6032f19eb225d77f3e4c8c0004a697c836aa3afa6c9f3",
    }


def main() -> int:
    try:
        result = preflight()
    except Exception as exc:
        print(json.dumps({"decision": "NO-GO", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["grader_hash_matches_frozen_v1"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
