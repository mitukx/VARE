#!/usr/bin/env python3
"""Launch a frozen VARE-IG run with incremental sample persistence and MLX cache release.

The shim leaves the locked training runner untouched. It adds two execution
behaviors that are themselves hash-bound by the VARE-IG run protocol:

* clear reusable MLX buffers before each existing resource-guard check;
* append exact learner samples, including token IDs, immediately after their
  graded rollout is durably appended and before any behavior-logprob pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vare import grpo_mlx  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_training_sample_record(generated: Mapping[str, Any], rollout: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize the exact generated token sequence alongside trusted reward provenance."""
    text = generated.get("text")
    prompt_ids = generated.get("prompt_ids")
    completion_ids = generated.get("completion_ids")
    if not isinstance(text, str) or not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
        raise RuntimeError("generated rollout omitted exact response or token IDs")
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_digest != rollout.get("response_sha256"):
        raise RuntimeError("captured generation differs from its persisted graded rollout")
    if len(prompt_ids) != int(rollout["prompt_tokens"]) \
            or len(completion_ids) != int(rollout["completion_tokens"]):
        raise RuntimeError("captured token IDs differ from the persisted rollout token counts")
    grader = rollout.get("grader")
    if not isinstance(grader, Mapping) or not grader.get("verifier_id"):
        raise RuntimeError("rollout sample lacks its trusted verifier identity")

    sample = grpo_mlx.Sample(
        instance_id=str(rollout["task_id"]),
        behavior_checkpoint_id=str(rollout["policy_checkpoint_id"]),
        behavior_step=int(rollout["behavior_step"]),
        prompt_ids=[int(token) for token in prompt_ids],
        completion_ids=[int(token) for token in completion_ids],
        reward=float(rollout["reward"]),
        verifier=str(grader["verifier_id"]),
        sample_index=int(rollout["sample_index"]),
        meta={"task_sha256": str(rollout["task_sha256"]), "phase": str(rollout["phase"])},
    )
    reward_detail = {
        "reward": float(rollout["reward"]),
        "grade_status": rollout["grade_status"],
        "trusted_contract": rollout["trusted_contract"],
        "passed": float(rollout["reward"]) == 1.0,
        "cases": grader.get("case_results", []),
        "case_set_sha256": grader.get("case_set_sha256"),
    }
    row = runner.serialize_sample(
        sample,
        response={"text": text, "candidate_source_sha256": rollout.get("candidate_source_sha256"),
                  "finish": rollout.get("finish")},
        task_sha256=str(rollout["task_sha256"]),
        phase=str(rollout["phase"]),
    )
    row.update({
        "response_text": text,
        "sampler_logprobs": [float(value) for value in generated.get("sampler_logprobs", [])],
        "reward_detail": reward_detail,
    })
    return row


def _protocol_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.protocol.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    protocol_path = _protocol_path(argv)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    execution = protocol.get("execution", {})
    if execution.get("durable_training_launcher_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("launcher shim is not the exact version locked by the run protocol")
    if execution.get("cache_release_before_resource_guard") is not True:
        raise RuntimeError("run protocol does not authorize the locked MLX cache-release behavior")
    if execution.get("persist_token_ids_before_behavior_logprobs") is not True:
        raise RuntimeError("run protocol does not require incremental exact-sample persistence")

    import mlx.core as mx
    from vare.mlx_prefix_reuse import CachedGroupGenerator

    original_guard = runner.resource_guard

    def release_cache_then_guard(path: Path, phase_started: float) -> dict[str, Any]:
        mx.clear_cache()
        return original_guard(path, phase_started)

    runner.resource_guard = release_cache_then_guard

    latest_generated: dict[str, Any] | None = None
    original_generate = runner.generate

    def capture_generate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal latest_generated
        latest_generated = original_generate(*args, **kwargs)
        return latest_generated

    runner.generate = capture_generate

    original_cached_call = CachedGroupGenerator.__call__

    def capture_cached_call(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal latest_generated
        latest_generated = original_cached_call(self, *args, **kwargs)
        return latest_generated

    CachedGroupGenerator.__call__ = capture_cached_call

    original_append = runner.append_jsonl

    def persist_rollout_and_sample(path: Path, value: Mapping[str, Any]) -> None:
        nonlocal latest_generated
        original_append(path, value)
        if Path(path).name not in {"rollouts-train-step-0.jsonl", "rollouts-train-step-1.jsonl"}:
            return
        if latest_generated is None:
            raise RuntimeError("a graded rollout was appended without a captured generation")
        sample_row = make_training_sample_record(latest_generated, value)
        sample_path = Path(path).with_name(
            f"training-samples-streamed-{sample_row['phase']}.jsonl")
        original_append(sample_path, sample_row)

    runner.append_jsonl = persist_rollout_and_sample
    return experiment.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
