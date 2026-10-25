#!/usr/bin/env python3
"""Run one pinned upstream unit test with MLX explicitly bound to CPU."""
from __future__ import annotations

import argparse
import unittest

import mlx.core as mx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selector")
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    suite = unittest.defaultTestLoader.loadTestsFromName(args.selector)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
