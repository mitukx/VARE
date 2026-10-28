#!/usr/bin/env python3
"""Randomized hidden behavior grader for Transformers PR #50044."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import time


TARGET = "transformers/models/janus/processing_janus.py"
CONTAINER_NAME = re.compile(r"vare-janus-grade-[0-9a-f]{24}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
DEFAULT_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.\n\n"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_challenge() -> tuple[str, dict, dict]:
    nonce = secrets.token_hex(16)
    texts = ["q" + secrets.token_hex(5) for _ in range(6)]
    marker = "<m" + secrets.token_hex(4) + ">"
    cases = [
        {"name": "text_string_default", "mode": "text", "default_prompt": True, "text": texts[0], "marker": marker},
        {"name": "text_string_no_default", "mode": "text", "default_prompt": False, "text": texts[1], "marker": marker},
        {"name": "image_string", "mode": "image", "default_prompt": False, "text": texts[2], "marker": marker},
        {"name": "text_list_default", "mode": "text", "default_prompt": True, "text": texts[3:5], "marker": marker},
        {"name": "image_list", "mode": "image", "default_prompt": False, "text": [texts[4], texts[5]], "marker": marker},
    ]

    def transform(case):
        value = case["text"]
        values = [value] if isinstance(value, str) else list(value)
        if case["mode"] == "text" and case["default_prompt"]:
            values = [DEFAULT_PROMPT + item for item in values]
        elif case["mode"] == "image":
            values = [item + case["marker"] for item in values]
        return values

    expected = {case["name"]: transform(case) for case in cases}
    return nonce, {"nonce": nonce, "cases": cases}, expected


def probe_source() -> str:
    return r'''import json
import sys
from types import SimpleNamespace
from transformers.models.janus.processing_janus import JanusProcessor
from transformers.processing_utils import ProcessorMixin

request = json.loads(sys.stdin.readline())
original_call = ProcessorMixin.__call__
observed = {}
def capture(self, *, images=None, text=None, **kwargs):
    return text
ProcessorMixin.__call__ = capture
try:
    for case in request["cases"]:
        processor = JanusProcessor.__new__(JanusProcessor)
        processor.tokenizer = SimpleNamespace(init_kwargs={})
        processor.use_default_system_prompt = case["default_prompt"]
        processor.image_start_token = case["marker"]
        mode = case["mode"]
        processor._merge_kwargs = lambda *_args, **_kwargs: {"text_kwargs": {"generation_mode": mode}}
        observed[case["name"]] = processor(case["text"], text_kwargs={"generation_mode": mode})
finally:
    ProcessorMixin.__call__ = original_call
print(json.dumps({"nonce": request["nonce"], "results": observed}, sort_keys=True, separators=(",", ":")))
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
    record_path = args.record
    if record_path.exists():
        parser.error(f"refusing to overwrite existing grade receipt: {record_path}")
    record = {
        "schema_version": 1,
        "task_id": "transformers-50044-janus-string-prompt-v1",
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
        record["expected"] = expected
        challenge_bytes = (json.dumps(challenge, sort_keys=True, separators=(",", ":")) + "\n").encode()
        name = "vare-janus-grade-" + secrets.token_hex(12)
        if not CONTAINER_NAME.fullmatch(name):
            raise RuntimeError("invalid grader container name")
        argv = [
            args.docker, "--context", args.context, "run", "--rm", "--pull=never", "--interactive",
            "--name", name, "--network=none", "--read-only", "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m,mode=1777", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true", "--pids-limit", "32", "--memory", "1536m",
            "--cpus", "1", "--user", "65534:65534", "--mount",
            f"type=bind,source={workspace / 'src'},target=/repo/src,readonly", "--workdir=/",
            "-e", "PYTHONPATH=/repo/src", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", args.image,
            "python", "-c", probe_source(),
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
        if result.returncode != 0:
            record["failure"] = "candidate grader process exited nonzero"
        elif len(lines) != 1:
            record["failure"] = "grader output must contain exactly one JSON line"
        else:
            try:
                response = json.loads(lines[0])
            except json.JSONDecodeError:
                response = None
            if not isinstance(response, dict) or set(response) != {"nonce", "results"}:
                record["failure"] = "invalid grader response schema"
            elif response["nonce"] != nonce:
                record["failure"] = "grader nonce mismatch"
            elif response["results"] != expected:
                got = response.get("results")
                record["wrong_cases"] = sorted(
                    key for key in expected if not isinstance(got, dict) or got.get(key) != expected[key]
                )
                record["failure"] = "candidate behavior differs from expected task contract"
            elif before != after:
                record["failure"] = "candidate source changed during grading"
            else:
                record["status"] = "passed"
                record["passed"] = True
                record["failure"] = None
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ("status", "passed", "failure", "wrong_cases", "elapsed_seconds")}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
