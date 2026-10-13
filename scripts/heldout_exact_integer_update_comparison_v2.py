#!/usr/bin/env python3
"""Refuse a policy-update comparison on the held-out exact-integer v2 gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.heldout_exact_integer import PROTOCOL_ID_V2, comparison_refusal  # noqa: E402


def main() -> int:
    print(json.dumps(comparison_refusal(PROTOCOL_ID_V2), indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
