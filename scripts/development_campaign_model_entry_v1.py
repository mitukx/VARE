#!/usr/bin/env python3
"""Apply the prospective campaign's Metal allocation cap, then use its runner."""
from pathlib import Path
import runpy

import mlx.core as mx

mx.set_memory_limit(6 * 1024**3)
mx.set_cache_limit(512 * 1024**2)
runpy.run_path(str(Path(__file__).with_name("run_local_agent.py")), run_name="__main__")
