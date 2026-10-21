"""Grade one Phase A Transformers patch with parent-owned expected values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from vare.isolated_candidate import FunctionCall, grade_calls

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "transformers-49116-tokenizer-routing-v2": (
        "transformers_49116_tokenizer_routing_v2",
        "src/transformers/models/auto/tokenization_auto.py",
        "routes",
        {
            "model_types": ["qwen3_5_moe_text", "qwen4_exp", "llama", "qwen3"]
        },
        {
            "routes": {
                "qwen3_5_moe_text": "Qwen3_5Tokenizer",
                "qwen4_exp": "Qwen3_5Tokenizer",
                "llama": "<unregistered>",
                "qwen3": "Qwen2Tokenizer",
            },
            "override_types": ["qwen3_5_moe_text", "qwen4_exp"],
        },
    ),
    "transformers-49412-fusion-registration-idempotence-v2": (
        "transformers_49412_fusion_idempotence_v2",
        "src/transformers/fusion_mapping.py",
        "repeated_registration",
        {"model_type": "vare_hidden_fusion_probe"},
        {"repeat_accepted": True, "mapping_unchanged": True},
    ),
}


def grade(task_id: str, workspace: Path, image: str) -> dict[str, Any]:
    task_dir_name, target, function, args, expected = TASKS[task_id]
    task_root = ROOT / "benchmarks/mlsys_repo_eval_v4" / task_dir_name
    adapter = task_root / "grader_adapter.py"
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("candidate workspace is missing or unsafe")
    if (workspace / "evaluator").exists() or (workspace / ".git").exists():
        raise ValueError("candidate workspace contains grader material or Git history")
    if not (workspace / target).is_file():
        raise ValueError("candidate target source is missing")

    # Stage only the adapter entry point alongside the candidate source. Hidden
    # expected values remain in this parent process and are never mounted.
    with tempfile.TemporaryDirectory(prefix="vare-phase-a-subject-") as directory:
        subject = Path(directory) / "subject"
        shutil.copytree(workspace, subject)
        shutil.copy2(adapter, subject / "vare_phase_a_adapter.py")
        call = FunctionCall(
            case_id=task_id,
            module="vare_phase_a_adapter",
            function=function,
            args=[args],
            expected=expected,
        )
        report = grade_calls(
            subject,
            [call],
            task_kind="correctness",
            image=image,
            timeout_s=60,
            memory_mb=1900,
            discard_candidate_stdout=True,
        )
    payload = report.to_dict()
    payload.update(
        {
            "task_id": task_id,
            "candidate_target": target,
            "expected_held_by_parent": True,
            "grader_mounted": False,
            "candidate_git_history_present": False,
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image", required=True, help="Digest-pinned candidate image")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite grade evidence")
    result = grade(args.task_id, args.workspace.resolve(), args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
