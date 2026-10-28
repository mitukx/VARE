#!/usr/bin/env python3
"""Nonce-bound hidden grader for the MiniCPM-V Stage 2 input contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import secrets
import subprocess
import time


TARGET = "src/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py"
CONTAINER_NAME = re.compile(r"vare-minicpm-grade-[0-9a-f]{24}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_challenge() -> tuple[dict, list[float]]:
    rng = random.SystemRandom()
    values = []
    while len(values) < 4:
        candidate = round(rng.uniform(2.0, 40.0), 5)
        if candidate not in values:
            values.append(candidate)
    return {"nonce": secrets.token_hex(16), "frame_values": values}, values


def probe_source() -> str:
    return r'''from __future__ import annotations
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

request = json.loads(sys.stdin.readline())
source_path = Path("/repo/src/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py")
tree = ast.parse(source_path.read_text(encoding="utf-8"))
processor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniCPMV4_6VideoProcessor")
method = next(node for node in processor.body if isinstance(node, ast.FunctionDef) and node.name == "_preprocess")
probe_class = ast.ClassDef(name="Probe", bases=[], keywords=[], decorator_list=[], body=[method])
module = ast.fix_missing_locations(ast.Module(body=[probe_class], type_ignores=[]))
observed = []
class StopAtStageTwo(Exception): pass
def group_videos_by_shape(items):
    observed.extend(items)
    raise StopAtStageTwo()
namespace = {"math": math, "np": np, "torch": torch, "group_videos_by_shape": group_videos_by_shape}
exec(compile(module, "minicpm_probe.py", "exec"), namespace)
instance = namespace["Probe"]()
instance.concat_frames_as_image = lambda frames: torch.full(
    (1, 1, 1, 1), 1000.0 + float(frames[0, 0, 0, 0]), dtype=torch.float32
)
video = torch.tensor(request["frame_values"], dtype=torch.float32).reshape(4, 1, 1, 1)
try:
    instance._preprocess(
        [video], False, None, False, 1.0, False, None, None, 9, 448, 14, False,
        2, 4, [SimpleNamespace(duration=2.0)], None
    )
except StopAtStageTwo:
    pass
else:
    raise RuntimeError("probe did not reach Stage 2")
units = [{"shape": list(value.shape), "value": float(value[0, 0, 0, 0])} for value in observed]
print(json.dumps({"nonce": request["nonce"], "units": units}, sort_keys=True, separators=(",", ":")))
'''


def output_matches(response: object, nonce: str, values: list[float]) -> bool:
    if not isinstance(response, dict) or set(response) != {"nonce", "units"} or response["nonce"] != nonce:
        return False
    units = response["units"]
    expected = [values[0], 1000 + values[2], values[1], 1000 + values[3]]
    if not isinstance(units, list) or len(units) != 4:
        return False
    for item, wanted in zip(units, expected):
        if not isinstance(item, dict) or set(item) != {"shape", "value"}:
            return False
        if item["shape"] != [1, 1, 1, 1] or not isinstance(item["value"], (int, float)):
            return False
        if not math.isclose(float(item["value"]), wanted, rel_tol=1e-6, abs_tol=1e-4):
            return False
    return True


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
    if args.record.exists():
        parser.error(f"refusing to overwrite grade receipt: {args.record}")
    record = {"schema_version": 1, "task_id": "transformers-49752-minicpm-stacked-visual-units-v1",
              "status": "infrastructure_failure", "passed": False}
    started = time.monotonic()
    try:
        grader_sha = sha256_file(Path(__file__).resolve(strict=True))
        record["grader_sha256"] = grader_sha
        if grader_sha != args.expected_grader_sha256:
            raise RuntimeError("grader source hash mismatch")
        workspace = args.workspace.resolve(strict=True)
        if args.workspace.is_symlink() or not workspace.is_dir() or (workspace / "evaluator").exists():
            raise RuntimeError("candidate workspace isolation boundary invalid")
        source = workspace / TARGET
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("candidate source is missing or linked")
        before = sha256_file(source)
        image_id = subprocess.run(
            [args.docker, "--context", args.context, "image", "inspect", args.image, "--format", "{{.Id}}"],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        if not IMAGE_ID_RE.fullmatch(image_id) or image_id != args.expected_image_id:
            raise RuntimeError("grader image identity mismatch")
        challenge, values = build_challenge()
        challenge_bytes = (json.dumps(challenge, sort_keys=True, separators=(",", ":")) + "\n").encode()
        name = "vare-minicpm-grade-" + secrets.token_hex(12)
        if not CONTAINER_NAME.fullmatch(name):
            raise RuntimeError("invalid container name")
        argv = [
            args.docker, "--context", args.context, "run", "--rm", "--pull=never", "--interactive",
            "--name", name, "--network=none", "--read-only", "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m,mode=1777", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true", "--pids-limit", "32", "--memory", "1536m",
            "--cpus", "1", "--user", "65534:65534", "--mount",
            f"type=bind,source={workspace / 'src'},target=/repo/src,readonly", "--workdir=/",
            "-e", "PYTHONDONTWRITEBYTECODE=1", args.image, "python", "-c", probe_source(),
        ]
        result = subprocess.run(argv, input=challenge_bytes, capture_output=True, timeout=60)
        after = sha256_file(source)
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        record.update({"source_sha256_before": before, "source_sha256_after": after,
                       "source_unchanged": before == after, "image_id": image_id,
                       "challenge": challenge, "challenge_sha256": hashlib.sha256(challenge_bytes).hexdigest(),
                       "container_exit_code": result.returncode, "container_stdout": stdout,
                       "container_stderr": stderr})
        lines = stdout.splitlines()
        if before != after:
            record.update(status="grader_rejected", failure="source changed during grading")
        elif result.returncode != 0 or len(lines) != 1:
            record.update(status="grader_rejected", failure="probe did not return one clean result line")
        else:
            try:
                response = json.loads(lines[0])
            except json.JSONDecodeError:
                response = None
            if output_matches(response, challenge["nonce"], values):
                record.update(status="passed", passed=True, failure=None)
            else:
                record.update(status="grader_rejected", failure="stacked visual-unit ordering/shape contract failed")
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ("status", "passed", "failure", "elapsed_seconds")}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
