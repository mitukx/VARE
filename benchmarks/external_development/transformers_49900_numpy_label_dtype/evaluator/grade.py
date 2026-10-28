#!/usr/bin/env python3
"""Host-side independent grader for Transformers PR #49900.

The candidate target is mounted read-only by itself into a network-disabled,
non-root container. The parent owns all pass/fail comparisons and never trusts
the candidate process exit status as a grade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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


def expected_results() -> dict[str, dict]:
    expected: dict[str, dict] = {}

    def add(name: str, dtype: str, values: list) -> None:
        expected[name] = {"dtype": dtype, "values": values}

    scalar_values = [-17, 0, 29]
    vector_values = [[-3, 8], [0, 11], [41, -9]]
    for scalar_type in ("int32", "int64"):
        for backend, dtype in (("pt", "torch.int64"), ("np", "int64")):
            add(f"{scalar_type}_label_{backend}", dtype, scalar_values)
            add(f"{scalar_type}_label_ids_{backend}", dtype, vector_values)
    for backend, int_dtype, float_dtype in (
        ("pt", "torch.int64", "torch.float32"),
        ("np", "int64", "float32"),
    ):
        add(f"python_int_label_{backend}", int_dtype, [-2, 6, 15])
        add(f"python_int_label_ids_{backend}", int_dtype, [[1, -1], [0, 7], [8, 9]])
        add(f"python_float_label_{backend}", float_dtype, [-1.5, 0.0, 2.25])
        add(
            f"numpy_float_label_ids_{backend}",
            float_dtype,
            [[-0.5, 1.25], [0.0, 2.0], [3.75, -4.5]],
        )
    return expected


def probe_source() -> str:
    # The probe runs separately from this host grader. The candidate source is
    # the only mounted input; no evaluator directory or file is mounted.
    return r'''import json
import numpy as np
import torch
from transformers.data.data_collator import default_data_collator

cases = {}

def record(name, features, backend):
    out = default_data_collator(features, return_tensors=backend)["labels"]
    cases[name] = {"dtype": str(out.dtype), "values": out.tolist()}

for scalar_type in (np.int32, np.int64):
    type_name = "int32" if scalar_type is np.int32 else "int64"
    record(f"{type_name}_label_pt", [
        {"input_ids": [i], "label": scalar_type(v)}
        for i, v in enumerate((-17, 0, 29))], "pt")
    record(f"{type_name}_label_np", [
        {"input_ids": [i], "label": scalar_type(v)}
        for i, v in enumerate((-17, 0, 29))], "np")
    record(f"{type_name}_label_ids_pt", [
        {"input_ids": [i], "label_ids": [scalar_type(a), scalar_type(b)]}
        for i, (a, b) in enumerate(((-3, 8), (0, 11), (41, -9)))], "pt")
    record(f"{type_name}_label_ids_np", [
        {"input_ids": [i], "label_ids": [scalar_type(a), scalar_type(b)]}
        for i, (a, b) in enumerate(((-3, 8), (0, 11), (41, -9)))], "np")

record("python_int_label_pt", [
    {"input_ids": [i], "label": v} for i, v in enumerate((-2, 6, 15))], "pt")
record("python_int_label_np", [
    {"input_ids": [i], "label": v} for i, v in enumerate((-2, 6, 15))], "np")
record("python_int_label_ids_pt", [
    {"input_ids": [i], "label_ids": [a, b]}
    for i, (a, b) in enumerate(((1, -1), (0, 7), (8, 9)))], "pt")
record("python_int_label_ids_np", [
    {"input_ids": [i], "label_ids": [a, b]}
    for i, (a, b) in enumerate(((1, -1), (0, 7), (8, 9)))], "np")
record("python_float_label_pt", [
    {"input_ids": [i], "label": v} for i, v in enumerate((-1.5, 0.0, 2.25))], "pt")
record("python_float_label_np", [
    {"input_ids": [i], "label": v} for i, v in enumerate((-1.5, 0.0, 2.25))], "np")
record("numpy_float_label_ids_pt", [
    {"input_ids": [i], "label_ids": [np.float32(a), np.float32(b)]}
    for i, (a, b) in enumerate(((-0.5, 1.25), (0.0, 2.0), (3.75, -4.5)))], "pt")
record("numpy_float_label_ids_np", [
    {"input_ids": [i], "label_ids": [np.float32(a), np.float32(b)]}
    for i, (a, b) in enumerate(((-0.5, 1.25), (0.0, 2.0), (3.75, -4.5)))], "np")

print(json.dumps(cases, sort_keys=True, separators=(",", ":")))
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
        "schema_version": 1,
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
        name = "vare-grade-" + os.urandom(12).hex()
        if not CONTAINER_NAME.fullmatch(name):
            raise RuntimeError("invalid grader container name")
        argv = [
            args.docker,
            "--context",
            args.context,
            "run",
            "--rm",
            "--pull=never",
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
        result = subprocess.run(argv, capture_output=True, text=True, timeout=90)
        source_hash_after = sha256_file(source)
        record.update(
            {
                "container_name": name,
                "container_exit_code": result.returncode,
                "container_stdout": result.stdout,
                "container_stderr": result.stderr,
                "source_sha256_before": source_hash_before,
                "source_sha256_after": source_hash_after,
                "source_unchanged": source_hash_before == source_hash_after,
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
            }
        )
        if result.returncode != 0:
            record["status"] = "candidate_execution_failed"
            record["failure"] = "isolated probe returned nonzero"
        elif source_hash_before != source_hash_after:
            record["status"] = "integrity_failure"
            record["failure"] = "candidate source changed during grading"
        else:
            lines = result.stdout.splitlines()
            if len(lines) != 1:
                record["status"] = "candidate_execution_failed"
                record["failure"] = "probe did not emit exactly one JSON line"
            else:
                observed = json.loads(lines[0])
                expected = expected_results()
                record["observed"] = observed
                record["expected"] = expected
                record["missing_cases"] = sorted(set(expected) - set(observed))
                record["unexpected_cases"] = sorted(set(observed) - set(expected))
                record["case_pass"] = {
                    key: observed.get(key) == expected_value
                    for key, expected_value in expected.items()
                }
                record["passed"] = (
                    set(observed) == set(expected)
                    and all(record["case_pass"].values())
                )
                record["status"] = "passed" if record["passed"] else "grader_rejected"
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)

    args.record.parent.mkdir(parents=True, exist_ok=True)
    if args.record.exists():
        print("refusing to overwrite grader record", file=sys.stderr)
        return 2
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
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
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] not in {"infrastructure_failure", "integrity_failure"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

