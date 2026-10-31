from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


class HiSparseConfig:
    pass


class AttentionConfig:
    def __init__(self, hisparse_config=None):
        self.hisparse_config = hisparse_config


class WatermarkConfig:
    def __init__(self, key):
        self.key = key


class ParallelConfig:
    def __init__(self, distributed_executor_backend=None):
        self.distributed_executor_backend = distributed_executor_backend
        self.prefill_context_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.enable_batch_sharded_sampling = False


class SchedulerConfig:
    @staticmethod
    def default_factory(async_scheduling=False):
        return SimpleNamespace(async_scheduling=async_scheduling)


class VllmConfig:
    @staticmethod
    def _get_v1_model_runner_unsupported_features(config):
        request = {
            "hisparse": config.attention_config.hisparse_config is not None,
            "watermarking": config.watermark_config is not None,
        }
        command = []
        profile = os.environ.get("VARE_METHOD_PROFILE")
        if profile:
            command.extend([os.environ.get("VARE_SANDBOX_EXEC", "/usr/bin/sandbox-exec"), "-f", profile])
        command.extend(
            [
                sys.executable,
                os.environ["VARE_METHOD_WORKER"],
                os.environ["VARE_METHOD_FILE"],
            ]
        )
        completed = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        if completed.returncode != 0:
            detail = completed.stderr[-1000:].strip()
            raise RuntimeError(
                f"isolated candidate method worker failed ({completed.returncode}): {detail}"
            )
        return json.loads(completed.stdout)

@pytest.mark.parametrize(
    ("overrides", "feature"),
    [
        (
            {"attention_config": AttentionConfig(hisparse_config=HiSparseConfig())},
            "HiSparse",
        ),
        ({"watermark_config": WatermarkConfig(key=42)}, "watermarking"),
    ],
)
def test_v1_model_runner_rejects_v2_only_config(overrides, feature):
    config = SimpleNamespace(
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
        attention_config=AttentionConfig(),
        watermark_config=None,
        speculative_config=None,
        model_config=None,
    )
    config._dflash_needs_multi_kv_group = lambda: False
    config._is_dflash_candidate_draft = lambda: False
    assert VllmConfig._get_v1_model_runner_unsupported_features(config) == []

    for name, value in overrides.items():
        setattr(config, name, value)
    assert VllmConfig._get_v1_model_runner_unsupported_features(config) == [feature]
