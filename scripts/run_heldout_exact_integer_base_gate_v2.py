#!/usr/bin/env python3
"""Run held-out exact-integer base gate v2.

The v1 pack and final-line parser stay as they were. This script does not
read or rescore the retained v1 samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_heldout_exact_integer_base_gate_v1 as v1  # noqa: E402
from vare.heldout_exact_integer import FINAL_LINE, PROTOCOL_ID_V2  # noqa: E402

PROTOCOL_PATH = ROOT / "benchmarks/heldout/exact_integer_v2/protocol.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="held-out exact-integer base gate v2")
    parser.add_argument("--backend", choices=("mlx", "transformers"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != PROTOCOL_ID_V2:
        raise SystemExit("v2 protocol id does not match the runner")
    if protocol["answer_extraction"]["regex"] != FINAL_LINE.pattern:
        raise SystemExit("v2 parser regex does not match the frozen grader")
    if protocol.get("frozen_before_model_output") is not True:
        raise SystemExit("v2 protocol is not marked frozen before model output")
    summary = v1.execute(PROTOCOL_PATH, args.backend, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
