#!/usr/bin/env python3
"""Perform exactly one offline greedy development attempt for one pinned local model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/urllib3_3352"
TASK_ID = "urllib3-preserve-ambiguous-request-target-v1"
MODEL_ALIASES = {"qwen": 0, "tinyllama": 1}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_ALIASES, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new run directory outside the VARE checkout")
    parser.add_argument("--task-python", default="/opt/homebrew/bin/python3.11")
    parser.add_argument(
        "--defer-grade",
        action="store_true",
        help="save the single inference result and leave grading to a fresh process",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite model attempt output: {output}")
    output.mkdir(parents=True)
    lock = json.loads((TASK_ROOT / "protocol.lock.json").read_text(encoding="utf-8"))
    asset_paths = {
        "task_json_sha256": TASK_ROOT / "task.json",
        "task_brief_sha256": TASK_ROOT / "TASK.md",
        "evaluator_sha256": TASK_ROOT / "evaluator/grade.py",
        "grader_entrypoint_sha256": ROOT / "scripts/grade_urllib3_3352_task.py",
        "prepare_script_sha256": ROOT / "scripts/prepare_urllib3_3352_task.py",
        "calibration_script_sha256": ROOT / "scripts/calibrate_urllib3_3352_task.py",
        "vare_spec_sha256": TASK_ROOT / "vare_spec.json",
        "model_screen_script_sha256": ROOT / "scripts/run_urllib3_3352_model_screen.py",
        "adversarial_test_script_sha256": ROOT / "scripts/adversarial_test_urllib3_3352.py",
        "evaluator_regression_test_sha256": ROOT / "tests/test_urllib3_3352_evaluator.py",
        "prompt_sha256": TASK_ROOT / "prompt.txt",
        "source_context_sha256": TASK_ROOT / "source_context.txt",
    }
    unlocked = [key for key, path in asset_paths.items()
                if not lock.get("locked_hashes", {}).get(key)
                or sha256_file(path) != lock["locked_hashes"][key]]
    if unlocked:
        raise SystemExit(f"task asset lock is missing or stale: {unlocked}")
    model_config = lock["model_screen"]["candidate_models"][MODEL_ALIASES[args.model]]

    # Force offline use before importing Transformers. No snapshot download or
    # remote inference is permitted by this development-only screen.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import try_to_load_from_cache
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; model attempt was not started")
    revision = model_config["revision"]
    repo_id = model_config["repo_id"]
    config_file = try_to_load_from_cache(repo_id, "config.json", revision=revision)
    if not isinstance(config_file, str):
        raise SystemExit(f"exact model snapshot is not cached offline: {repo_id}@{revision}")
    # Keep the cache snapshot symlink path itself: resolving config.json points
    # into the content-addressed blobs directory and loses the revision label.
    snapshot_dir = Path(config_file).absolute().parent
    if snapshot_dir.name != revision:
        raise SystemExit(f"cached model snapshot path does not match revision: {snapshot_dir}")
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    config_commit_metadata = getattr(model.config, "_commit_hash", None)
    tokenizer_commit_metadata = tokenizer.init_kwargs.get("_commit_hash")
    model.to("mps")
    model.eval()
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started
    mps_after_load = {
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }

    task_prompt = (TASK_ROOT / "prompt.txt").read_text(encoding="utf-8")
    context = (TASK_ROOT / "source_context.txt").read_text(encoding="utf-8")
    user_content = task_prompt + "\n\n" + context
    model_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("mps")
    torch.manual_seed(0)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=model_input,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.mps.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    new_tokens = generated[0, model_input.shape[1]:]
    model_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    mps_after_generation = {
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }
    output.joinpath("model_input.txt").write_text(user_content, encoding="utf-8")
    output.joinpath("model_output.txt").write_text(model_output, encoding="utf-8")

    preparation_dir = output / "preparation"
    prepare = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_urllib3_3352_task.py"), "--output", str(preparation_dir)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    (output / "prepare.stdout.json").write_text(prepare.stdout, encoding="utf-8")
    workspace = preparation_dir / "workspace"
    apply = subprocess.run(
        ["git", "-C", str(workspace), "apply", "--check", "--whitespace=error-all", "-"],
        input=model_output,
        capture_output=True,
        text=True,
    )
    patch_valid = apply.returncode == 0
    if patch_valid:
        applied = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--whitespace=error-all", "-"],
            input=model_output,
            capture_output=True,
            text=True,
        )
        patch_valid = applied.returncode == 0
    paths = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowed_paths = lock["source"]["candidate_paths"]
    patch_valid = patch_valid and bool(paths) and set(paths).issubset(set(allowed_paths))
    diff = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--binary", "HEAD", "--", *allowed_paths],
        check=True,
        capture_output=True,
    ).stdout
    (output / "candidate.diff").write_bytes(diff)

    record_path = preparation_dir / "source-record.json"
    if args.defer_grade:
        grade = None
        grading_result = {"passed": False, "deferred": True}
    else:
        grade = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/grade_urllib3_3352_task.py"),
                "--workspace", str(workspace),
                "--source-record", str(record_path),
                "--task-id", TASK_ID,
                "--task-python", args.task_python,
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        (output / "grading.stdout.json").write_text(grade.stdout, encoding="utf-8")
        (output / "grading.stderr.txt").write_text(grade.stderr, encoding="utf-8")
        try:
            grading_result = json.loads(grade.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            grading_result = {"unparsed_stdout": grade.stdout}
    # macOS reports ru_maxrss in bytes. Keep platform identity in the record.
    end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "task_class": "public historical development-only; retired after this screen",
        "model": model_config,
        "runtime": {
            "platform": sys.platform,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": "Apple MPS",
            "dtype": "float16",
            "snapshot_revision_from_verified_cache_path": snapshot_dir.name,
            "config_commit_metadata": config_commit_metadata,
            "tokenizer_commit_metadata": tokenizer_commit_metadata,
        },
        "prompt_sha256": sha256_bytes(user_content.encode()),
        "decoding": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 512, "seed": 0},
        "attempts": 1,
        "generated_tokens": int(new_tokens.numel()),
        "valid_patch": patch_valid,
        "candidate_changed_paths": paths,
        "independent_task_success": (
            bool(grading_result.get("passed", False)) if not args.defer_grade else None
        ),
        "grading_deferred": args.defer_grade,
        "task_grade": grading_result.get("task_grade") if not args.defer_grade else None,
        "grade_process_exit_code": grade.returncode if grade is not None else None,
        "timing_seconds": {"model_load": round(load_seconds, 6), "generation": round(generation_seconds, 6)},
        "memory": {
            "process_peak_rss_bytes": int(end_rss),
            "process_start_peak_rss_bytes": int(start_rss),
            "mps_after_load": mps_after_load,
            "mps_after_generation": mps_after_generation,
        },
        "paid_api_cost_usd": 0,
        "inference_seconds_per_valid_attempt": round(load_seconds + generation_seconds, 6) if patch_valid else None,
        "interpretation": "One descriptive development attempt; not a capability-improvement or generalization result.",
    }
    write_json(output / "result.json", result)
    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
             and "workspace" not in path.parts and "source" not in path.parts]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
