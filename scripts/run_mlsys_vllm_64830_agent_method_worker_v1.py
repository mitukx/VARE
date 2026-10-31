#!/usr/bin/env python3
"""Execute one extracted VLLM method in a grader-only subprocess."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def evaluate(method_path: Path, request: dict[str, Any]) -> list[str]:
    method_source = method_path.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "vllm_64830_candidate_worker"}
    exec(compile(method_source, str(method_path), "exec"), namespace)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp",
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            enable_batch_sharded_sampling=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        attention_config=SimpleNamespace(
            hisparse_config=object() if request["hisparse"] else None
        ),
        watermark_config=object() if request["watermarking"] else None,
        speculative_config=None,
        model_config=None,
        # Built-in bound methods do not expose Python test-module globals.
        _dflash_needs_multi_kv_group="".isdecimal,
        _is_dflash_candidate_draft="".isdecimal,
    )
    result = namespace["VllmConfig"]._get_v1_model_runner_unsupported_features(
        config
    )
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise TypeError("candidate must return a list of feature-name strings")
    return result


def main() -> int:
    method_path = Path(sys.argv[1]).resolve(strict=True)
    request = json.loads(sys.stdin.read())
    result = evaluate(method_path, request)
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
