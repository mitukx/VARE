#!/usr/bin/env python3
"""Randomized hidden grader for the TimesFM fp16 padding repair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import struct
import subprocess
import time


TARGET = "transformers/models/timesfm/modeling_timesfm.py"
CONTAINER_NAME = re.compile(r"vare-timesfm-grade-[0-9a-f]{24}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
PAD_VALUE = 1_123_581_321.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_challenge() -> tuple[str, dict, dict]:
    nonce = secrets.token_hex(16)
    rng = random.Random(secrets.randbits(64))
    values_fp16 = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(4)]
    ordinary_a = round(rng.uniform(-20.0, 20.0), 3)
    ordinary_b = round(rng.uniform(-20.0, 20.0), 3)
    cases = [
        {"name": "strict_fp16", "dtype": "float16", "values": values_fp16},
        {"name": "fp32_pad_preservation", "dtype": "float32", "values": [PAD_VALUE, ordinary_a, PAD_VALUE]},
        {"name": "fp64_pad_preservation", "dtype": "float64", "values": [ordinary_b, PAD_VALUE, ordinary_a]},
    ]
    expected = {case["name"]: {"dtype": case["dtype"], "values": case["values"]} for case in cases}
    return nonce, {"nonce": nonce, "pad_value": PAD_VALUE, "tolerance": 0.001, "cases": cases}, expected


def probe_source() -> str:
    return r'''import ast
import json
import sys
import types
from pathlib import Path
import torch

class StrictFp16Torch:
    def __getattr__(self, name):
        if name != "full":
            return getattr(torch, name)
        def strict_full(size, fill_value, *args, **kwargs):
            if kwargs.get("dtype") == torch.float16 and abs(float(fill_value)) > 65504:
                raise OverflowError("strict fp16 scalar conversion overflow")
            return torch.full(size, fill_value, *args, **kwargs)
        return strict_full

def load_probe(path, pad_value, tolerance):
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    model_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "TimesFmModel")
    transform = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_transform")
    helper = ast.parse(
        "def _timesfm_masked_mean_std(self, inputs, patched_pads):\n"
        "    shape = inputs.shape[:2]\n"
        "    return (torch.zeros(shape, dtype=inputs.dtype, device=inputs.device), "
        "torch.ones(shape, dtype=inputs.dtype, device=inputs.device))\n"
    ).body[0]
    probe_class = ast.ClassDef(name="Probe", bases=[], keywords=[], decorator_list=[], body=[helper, transform])
    module = ast.fix_missing_locations(ast.Module(body=[probe_class], type_ignores=[]))
    namespace = {"torch": StrictFp16Torch()}
    exec(compile(module, "timesfm_probe.py", "exec"), namespace)
    instance = namespace["Probe"]()
    instance.config = types.SimpleNamespace(pad_val=pad_value, tolerance=tolerance)
    return instance

request = json.loads(sys.stdin.readline())
source = Path("/repo/src") / "transformers/models/timesfm/modeling_timesfm.py"
model = load_probe(source, request["pad_value"], request["tolerance"])
results = {}
for case in request["cases"]:
    dtype = getattr(torch, case["dtype"])
    inputs = torch.tensor(case["values"], dtype=dtype).reshape((1, 1, -1))
    pads = torch.ones_like(inputs)
    try:
        outputs, _ = model._forward_transform(inputs, pads)
        results[case["name"]] = {"inputs": inputs.reshape(-1).tolist(), "outputs": outputs.reshape(-1).tolist()}
    except Exception as exc:
        results[case["name"]] = {"error": type(exc).__name__ + ": " + str(exc)}
print(json.dumps({"nonce": request["nonce"], "results": results}, sort_keys=True, separators=(",", ":")))
'''


def image_identity(docker: str, context: str, image: str) -> str:
    result = subprocess.run(
        [docker, "--context", context, "image", "inspect", image, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    identity = result.stdout.strip()
    if not IMAGE_ID_RE.fullmatch(identity):
        raise RuntimeError("Docker returned an invalid image id")
    return identity


def outputs_match(results: object, expected: dict) -> bool:
    if not isinstance(results, dict) or set(results) != set(expected):
        return False
    for name, values in expected.items():
        item = results.get(name)
        if not isinstance(item, dict) or set(item) != {"inputs", "outputs"}:
            return False
        inputs, outputs = item["inputs"], item["outputs"]
        if len(inputs) != len(values["values"]) or len(outputs) != len(values["values"]):
            return False
        format_char = {"float16": "e", "float32": "f", "float64": "d"}[values["dtype"]]
        for got_input, got_output, value in zip(inputs, outputs, values["values"]):
            if not isinstance(got_input, (int, float)) or not isinstance(got_output, (int, float)):
                return False
            want = struct.unpack(format_char, struct.pack(format_char, float(value)))[0]
            if abs(float(got_input) - float(want)) > max(abs(float(want)) * 1e-6, 1e-6):
                return False
            if abs(float(got_output) - float(want)) > max(abs(float(want)) * 1e-6, 1e-6):
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
        parser.error(f"refusing to overwrite existing grade receipt: {args.record}")
    record = {
        "schema_version": 1,
        "task_id": "transformers-50021-timesfm-fp16-pad-value-v1",
        "status": "infrastructure_failure",
        "passed": False,
        "workspace": str(args.workspace.resolve()),
    }
    started = time.monotonic()
    try:
        grader_path = Path(__file__).resolve(strict=True)
        grader_sha = sha256_file(grader_path)
        record["grader_sha256"] = grader_sha
        if grader_sha != args.expected_grader_sha256:
            raise RuntimeError("grader source hash mismatch")
        workspace = args.workspace.resolve(strict=True)
        if args.workspace.is_symlink() or not workspace.is_dir() or (workspace / "evaluator").exists():
            raise RuntimeError("candidate workspace has an invalid isolation boundary")
        source = workspace / "src" / TARGET
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("candidate target is missing or is a symlink")
        before = sha256_file(source)
        record["source_sha256_before"] = before
        image_id = image_identity(args.docker, args.context, args.image)
        record["image_id"] = image_id
        if image_id != args.expected_image_id:
            raise RuntimeError("grader image identity mismatch")
        nonce, challenge, expected = build_challenge()
        record["challenge"] = challenge
        challenge_bytes = (json.dumps(challenge, sort_keys=True, separators=(",", ":")) + "\n").encode()
        name = "vare-timesfm-grade-" + secrets.token_hex(12)
        if not CONTAINER_NAME.fullmatch(name):
            raise RuntimeError("invalid grader container name")
        argv = [
            args.docker, "--context", args.context, "run", "--rm", "--pull=never", "--interactive",
            "--name", name, "--network=none", "--read-only", "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m,mode=1777", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true", "--pids-limit", "32", "--memory", "1536m",
            "--cpus", "1", "--user", "65534:65534", "--mount",
            f"type=bind,source={workspace / 'src'},target=/repo/src,readonly", "--workdir=/",
            "-e", "PYTHONDONTWRITEBYTECODE=1", args.image, "python", "-c", probe_source(),
        ]
        result = subprocess.run(argv, input=challenge_bytes, capture_output=True, timeout=90)
        after = sha256_file(source)
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        record.update({
            "source_sha256_after": after,
            "source_unchanged": before == after,
            "challenge_sha256": hashlib.sha256(challenge_bytes).hexdigest(),
            "container_exit_code": result.returncode,
            "container_stdout": stdout,
            "container_stderr": stderr,
        })
        lines = stdout.splitlines()
        if not before == after:
            record["status"] = "grader_rejected"
            record["failure"] = "candidate source changed during grading"
        elif result.returncode != 0:
            record["status"] = "grader_rejected"
            record["failure"] = "candidate probe exited nonzero"
        elif len(lines) != 1:
            record["status"] = "grader_rejected"
            record["failure"] = "grader output must contain exactly one JSON line"
        else:
            try:
                response = json.loads(lines[0])
            except json.JSONDecodeError:
                response = None
            if not isinstance(response, dict) or set(response) != {"nonce", "results"}:
                record["status"] = "grader_rejected"
                record["failure"] = "invalid grader response schema"
            elif response["nonce"] != nonce:
                record["status"] = "grader_rejected"
                record["failure"] = "grader nonce mismatch"
            elif not outputs_match(response["results"], expected):
                record["status"] = "grader_rejected"
                record["failure"] = "candidate does not satisfy strict fp16 and padding preservation contracts"
            else:
                record["status"] = "passed"
                record["passed"] = True
                record["failure"] = None
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ("status", "passed", "failure", "elapsed_seconds")}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
