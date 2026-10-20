#!/usr/bin/env python3
"""Run consumed-task calibration controls through the isolated Docker grader."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

from vare.separated_candidate_harness import (
    PROTOCOL_PATH,
    ROOT,
    grade_function,
)


TASK_DIR = ROOT / "benchmarks/mlsys_coding_v1/tasks/causal_attention"
TASK_ID = "mlsys-v1-causal-attention"
ENTRYPOINT = "causal_attention:causal_attention_weights"
ADAPTER_SCRIPT = ROOT / "scripts/mlsys_task_isolated_calls.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def baseline_rejects_empty_result() -> bool:
    path = TASK_DIR / "evaluator/test_hidden.py"
    spec = importlib.util.spec_from_file_location("frozen_causal_attention_hidden", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen hidden checker")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args, kwargs = module._generator(0)
    expected = module._reference(*args, **kwargs)
    try:
        module._checker([], expected, list(args), {}, args, kwargs)
    except AssertionError:
        return True
    return False


def run(output: Path) -> int:
    output = output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing evidence directory: {output}")
    output.mkdir(parents=True)
    temp_root = output / "tmp"
    temp_root.mkdir()
    os.environ["TMPDIR"] = str(temp_root)
    tempfile.tempdir = str(temp_root)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    image_key = (
        "darwin_arm64" if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}
        else "linux_x86_64" if platform.system() == "Linux" and platform.machine() == "x86_64"
        else None
    )
    if image_key is None:
        raise SystemExit("host architecture has no locked grader image")

    candidate_bad = output / "candidate-empty-result"
    candidate_bad.mkdir()
    (candidate_bad / "causal_attention.py").write_text(
        "def causal_attention_weights(scores):\n    return []\n", encoding="utf-8",
    )
    candidate_good = output / "candidate-oracle-control"
    candidate_good.mkdir()
    shutil.copy(
        TASK_DIR / "evaluator/oracle/causal_attention.py",
        candidate_good / "causal_attention.py",
    )

    started = time.monotonic()
    baseline_rejects = baseline_rejects_empty_result()
    bad = grade_function({
        "workspace": str(candidate_bad),
        "entrypoint": ENTRYPOINT,
        "task_type": "correctness",
        "trusted_task_path": str(TASK_DIR / "task.json"),
        "nonce": "31" * 16,
    })
    good = grade_function({
        "workspace": str(candidate_good),
        "entrypoint": ENTRYPOINT,
        "task_type": "correctness",
        "trusted_task_path": str(TASK_DIR / "task.json"),
        "nonce": "42" * 16,
    })
    elapsed = time.monotonic() - started
    version = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if version.returncode != 0:
        raise SystemExit("Docker runtime version query failed")

    run_record = {
        "protocol_id": protocol["protocol_id"],
        "classification": "harness calibration on a previously consumed task; not model or held-out evidence",
        "source_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "task_id": TASK_ID,
        "task_type": "correctness",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "task_descriptor_sha256": sha256_file(TASK_DIR / "task.json"),
        "hidden_evaluator_sha256": sha256_file(TASK_DIR / "evaluator/test_hidden.py"),
        "oracle_sha256": sha256_file(TASK_DIR / "evaluator/oracle/causal_attention.py"),
        "pack_manifest_sha256": sha256_file(TASK_DIR.parents[1] / "manifest.json"),
        "frozen_harness_adapter_sha256": sha256_file(ROOT / "benchmarks/mlsys_coding_v1/harness_adapter.py"),
        "adapter_source_sha256": sha256_file(ADAPTER_SCRIPT),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "auditor_source_sha256": sha256_file(ROOT / "scripts/audit_mlsys_separated_candidate_harness_v1.py"),
        "harness_source_sha256": sha256_file(ROOT / "src/vare/separated_candidate_harness.py"),
        "isolated_grader_source_sha256": sha256_file(ROOT / "src/vare/isolated_candidate.py"),
        "separated_grader_source_sha256": sha256_file(ROOT / "src/vare/separated_grading.py"),
        "host": {"system": platform.platform(), "machine": platform.machine()},
        "container_image": protocol["container_images"][image_key],
        "docker_server": version.stdout.strip(),
        "elapsed_s": elapsed,
        "controls": {
            "frozen_checker_rejects_empty_result": baseline_rejects,
            "isolated_grader_rejects_empty_result": bad["passed"] is False,
            "isolated_grader_rejection_reason": bad["error"],
            "isolated_grader_accepts_oracle_control": good["passed"] is True,
        },
        "raw": {"empty_result": bad, "oracle_control": good},
    }
    write_json(output / "run.json", run_record)
    artifact_files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"evidence_manifest.json", "audit.json"}
    )
    manifest = {
        "schema_version": 1,
        "files": {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_files
        },
    }
    write_json(output / "evidence_manifest.json", manifest)
    print(json.dumps({
        "output": str(output),
        "frozen_checker_rejects_empty": baseline_rejects,
        "isolated_empty_passed": bad["passed"],
        "oracle_control_passed": good["passed"],
        "elapsed_s": elapsed,
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
