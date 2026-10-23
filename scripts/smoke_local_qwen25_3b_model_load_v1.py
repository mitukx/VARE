#!/usr/bin/env python3
"""Versioned entry point for the bounded local Qwen2.5-3B MLX load smoke."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import smoke_local_mlx_model_load_v1 as implementation  # noqa: E402

BUNDLE = ROOT / "results/mlsys-repo-task-mlx-lm-1912-qwen25-3b-load-smoke-v1"
PROTOCOL_PATH = BUNDLE / "protocol.json"
PROTOCOL_SHA_PATH = BUNDLE / "protocol.sha256"
DEFAULT_RUN = BUNDLE / "run-1"
SHARED_IMPLEMENTATION_PATH = Path(implementation.__file__).resolve()

implementation.BUNDLE = BUNDLE
implementation.PROTOCOL_PATH = PROTOCOL_PATH
implementation.PROTOCOL_SHA_PATH = PROTOCOL_SHA_PATH
implementation.DEFAULT_RUN = DEFAULT_RUN
implementation.__file__ = __file__


def _load_protocol() -> dict[str, Any]:
    protocol = implementation._read_json(PROTOCOL_PATH)
    expected = PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    if implementation.sha256_file(PROTOCOL_PATH) != expected:
        raise implementation.SmokeError("frozen protocol SHA-256 mismatch")
    if protocol.get("protocol_id") != "vare.local-mlx-model-load-smoke.qwen25-3b-v1":
        raise implementation.SmokeError("unexpected Qwen2.5-3B load-smoke protocol")
    if protocol.get("runner_sha256") != implementation.sha256_file(Path(__file__)):
        raise implementation.SmokeError("versioned Qwen2.5-3B launcher differs from protocol")
    if protocol.get("shared_implementation_sha256") != implementation.sha256_file(
        SHARED_IMPLEMENTATION_PATH
    ):
        raise implementation.SmokeError("shared model-load monitor differs from protocol")
    return protocol


implementation._load_protocol = _load_protocol


if __name__ == "__main__":
    raise SystemExit(implementation.run())
