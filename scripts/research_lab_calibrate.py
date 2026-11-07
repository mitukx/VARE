"""Calibrate ResearchGym graders on baseline, known-good, and incorrect controls.

Example:
  python scripts/research_lab_calibrate.py calibrate --work-root artifacts/research_lab/calibration
"""

from __future__ import annotations

import sys

from vare.research_lab.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = ["calibrate", "--help"]
    raise SystemExit(main(argv))
