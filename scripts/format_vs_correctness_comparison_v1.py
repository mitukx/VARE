#!/usr/bin/env python3
"""Refuse the five-arm format-versus-correctness comparison.

The process always exits 2. A smoke result on disk does not change that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.format_vs_correctness import comparison_refusal  # noqa: E402


def main() -> int:
    print(json.dumps(comparison_refusal(), indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
