#!/usr/bin/env python3
"""Randomized host-owned grader for Transformers PR #49900.

The candidate target is the only mounted task input. The generic probe is sent
on stdin, not placed in Docker argv or mounted from the evaluator directory.
The parent generates fresh numeric challenge values and checks typed results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time


TARGET = "transformers/data/data_collator.py"
CONTAINER_NAME = re.compile(r"vare-grade-[0-9a-f]{24}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_challenge() -> tuple[str, dict, dict]:
    """Create per-invocation values that are absent from command args and files."""
    nonce = secrets.token_hex(16)
    cases = []
    expected = {}

    def add(name: str, scalar_type: str, field: str, backend: str, values: list, dtype: str):
        cases.append({
            "name": name,
            "scalar_type": scalar_type,
            "field": field,
            "backend": backend,
            "values": values,
        })
        expected[name] = {"dtype": dtype, "values": values}

    int32_rows = [[secrets.randbelow(2_000_001) - 1_000_000 for _ in range(3)] for _ in range(4)]
    int64_rows = [[secrets.randbelow(2**50) - 2**49 for _ in range(3)] for _ in range(4)]
    for scalar_type, rows in (("int32", int32_rows), ("int64", int64_rows)):
        values = [row[0] for row in rows]
        vectors = [row[1:] for row in rows]
        for backend, dtype in (("pt", "torch.int64"), ("np", "int64")):
            add(f"{scalar_type}_label_{backend}", scalar_type, "label", backend, values, dtype)
            add(f"{scalar_type}_label_ids_{backend}", scalar_type, "label_ids", backend, vectors, dtype)

    python_ints = [secrets.randbelow(100_001) - 50_000 for _ in range(4)]
    python_int_vectors = [
        [secrets.randbelow(100_001) - 50_000 for _ in range(2)] for _ in range(4)
    ]
    python_floats = [
        round(secrets.randbelow(200_001) / 100.0 - 1000.0, 2) for _ in range(4)
    ]
    numpy_float_vectors = [
        [round(secrets.randbelow(20_001) / 100.0 - 100.0, 2) for _ in range(2)]
        for _ in range(4)
    ]
    for backend, int_dtype, float_dtype in (
        ("pt", "torch.int64", "torch.float32"),
        ("np", "int64", "float32"),
    ):
        add("python_int_label_" + backend, "python_int", "label", backend, python_ints, int_dtype)
        add(
            "python_int_label_ids_" + backend,
            "python_int",
            "label_ids",
            backend,
            python_int_vectors,
            int_dtype,
        )
        add("python_float_label_" + backend, "python_float", "label", backend, python_floats, float_dtype)
        add(
            "numpy_float_label_ids_" + backend,
            "numpy_float32",
            "label_ids",
            backend,
            numpy_float_vectors,
            float_dtype,
        )
    challenge = {"nonce": nonce, "cases": cases}
    return nonce, challenge, expected


def probe_source() -> str:
    # Generic code only: all task values and nonce arrive on stdin for this run.
    return r'''import json
import sys
import numpy as np
import torch
from transformers.data.data_collator import default_data_collator

request = json.loads(sys.stdin.readline())
results = {}
for case in request["cases"]:
    kind = case["scalar_type"]
    values = case["values"]
    field = case["field"]
    features = []
    for index, value in enumerate(values):
        if kind == "int32":
            scalar = np.int32
        elif kind == "int64":
            scalar = np.int64
        elif kind == "numpy_float32":
            scalar = np.float32
        elif kind == "python_float":
            scalar = float
        else:
            scalar = int
        if field == "label":
            label = scalar(value)
        else:
            label = [scalar(item) for item in value]
        features.append({"input_ids": [index, index + 1], field: label})
    output = default_data_collator(features, return_tensors=case["backend"])["labels"]
    results[case["name"]] = {"dtype": str(output.dtype), "values": output.tolist()}
print(json.dumps({"nonce": request["nonce"], "results": results}, sort_keys=True,
                 separators=(",", ":")))
'''


def image_identity(docker: str, context: str, image: str) -> str:
    result = subprocess.run(
        [docker, "--context", context, "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    identity = result.stdout.strip()
    if not IMAGE_ID_RE.fullmatch(identity):
        raise RuntimeError("Docker returned an invalid image id")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--expected-grader-sha256", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--context", default="colima")
    args = parser.parse_args()

    started = time.monotonic()
    record = {
        "schema_version": 2,
        "task_id": "transformers-49900-numpy-label-dtype-v1",
        "status": "infrastructure_failure",
        "passed": False,
        "workspace": str(args.workspace.resolve()),
    }
    try:
        grader_path = Path(__file__).resolve(strict=True)
        grader_hash = sha256_file(grader_path)
        record["grader_sha256"] = grader_hash
        if grader_hash != args.expected_grader_sha256:
            raise RuntimeError("grader source hash mismatch")
        workspace = args.workspace.resolve(strict=True)
        if args.workspace.is_symlink() or not workspace.is_dir():
            raise RuntimeError("candidate workspace must be a regular directory")
        if (workspace / "evaluator").exists():
            raise RuntimeError("candidate workspace contains evaluator files")
        source = workspace / "src" / TARGET
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("candidate source is missing or is a symlink")
        source_hash_before = sha256_file(source)
        image_id = image_identity(args.docker, args.context, args.image)
        record["image_id"] = image_id
        if image_id != args.expected_image_id:
            raise RuntimeError("grader image id mismatch")
        nonce, challenge, expected = build_challenge()
        record["challenge"] = challenge
        record["expected"] = expected
        challenge_bytes = (json.dumps(challenge, sort_keys=True, separators=(",", ":")) + "\n").encode()
        name = "vare-grade-" + secrets.token_hex(12)
        if not CONTAINER_NAME.fullmatch(name):
            raise RuntimeError("invalid grader container name")
        argv = [
            args.docker,
            "--context",
            args.context,
            "run",
            "--rm",
            "--pull=never",
            "--interactive",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m,mode=1777",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--pids-limit",
            "32",
            "--memory",
            "1536m",
            "--cpus",
            "1",
            "--user",
            "65534:65534",
            "--mount",
            f"type=bind,source={workspace / 'src'},target=/repo/src,readonly",
            "--workdir=/",
            "-e",
            "PYTHONPATH=/repo/src",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            args.image,
            "python",
            "-c",
            probe_source(),
        ]
        result = subprocess.run(
            argv,
            input=challenge_bytes,
            capture_output=True,
            timeout=90,
        )
        source_hash_after = sha256_file(source)
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        record.update({
            "container_name": name,
            "container_exit_code": result.returncode,
            "container_stdout": stdout,
            "container_stderr": stderr,
            "source_sha256_before": source_hash_before,
            "source_sha256_after": source_hash_after,
            "source_unchanged": source_hash_before == source_hash_after,
            "challenge_nonce": nonce,
            "challenge_sha256": hashlib.sha256(challenge_bytes).hexdigest(),
            "resource_limits": {
                "network": "none",
                "rootfs": "read-only",
                "uid": 65534,
                "memory_mb": 1536,
                "cpus": 1,
                "pids": 32,
                "tmpfs_mb": 16,
                "timeout_seconds": 90,
            },
        })
        if result.returncode != 0:
            record["status"] = "candidate_execution_failed"
            record["failure"] = "isolated probe returned nonzero"
        elif source_hash_before != source_hash_after:
            record["status"] = "integrity_failure"
            record["failure"] = "candidate source changed during grading"
        else:
            lines = stdout.splitlines()
            if len(lines) != 1:
                record["status"] = "grader_rejected"
                record["failure"] = "candidate did not emit exactly one JSON line"
            else:
                try:
                    observed = json.loads(lines[0])
                except json.JSONDecodeError:
                    record["status"] = "grader_rejected"
                    record["failure"] = "candidate output was not one JSON object"
                else:
                    if not isinstance(observed, dict) or set(observed) != {"nonce", "results"}:
                        record["status"] = "grader_rejected"
                        record["failure"] = "candidate response schema was invalid"
                    elif observed["nonce"] != nonce:
                        record["status"] = "grader_rejected"
                        record["failure"] = "attempt receipt nonce mismatch"
                    elif not isinstance(observed["results"], dict):
                        record["status"] = "grader_rejected"
                        record["failure"] = "candidate results value was not an object"
                    else:
                        results = observed["results"]
                        record["observed"] = observed
                        record["missing_cases"] = sorted(set(expected) - set(results))
                        record["unexpected_cases"] = sorted(set(results) - set(expected))
                        record["case_pass"] = {
                            key: results.get(key) == expected_value
                            for key, expected_value in expected.items()
                        }
                        record["passed"] = (
                            set(results) == set(expected)
                            and all(record["case_pass"].values())
                        )
                        record["status"] = "passed" if record["passed"] else "grader_rejected"
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    except subprocess.TimeoutExpired as exc:
        record["status"] = "candidate_execution_failed"
        record["failure"] = "isolated probe exceeded 90-second timeout"
        record["partial_stdout"] = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)

    args.record.parent.mkdir(parents=True, exist_ok=True)
    if args.record.exists():
        print("refusing to overwrite grader record", file=sys.stderr)
        return 2
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        key: record.get(key)
        for key in (
            "task_id",
            "status",
            "passed",
            "failure",
            "elapsed_seconds",
            "source_sha256_before",
            "grader_sha256",
            "image_id",
        )
    }, sort_keys=True))
    return 0 if record["status"] not in {"infrastructure_failure", "integrity_failure"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

