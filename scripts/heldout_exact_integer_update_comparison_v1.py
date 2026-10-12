#!/usr/bin/env python3
"""Refuse a policy-update comparison on the held-out exact-integer pack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.heldout_exact_integer import comparison_refusal  # noqa: E402


def main() -> int:
    print(json.dumps(comparison_refusal(), indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
