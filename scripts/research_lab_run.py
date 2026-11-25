"""Run a ResearchGym command with the repository CLI.

Example:
  python scripts/research_lab_run.py run --task grad_accum_scale_v1 --run-id demo --proposer diagnostic
"""

from __future__ import annotations

import sys

from vare.research_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
