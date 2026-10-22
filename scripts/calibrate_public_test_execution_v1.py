#!/usr/bin/env python3
"""Calibrate Colima public-test workspace staging without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_local_agent import run_public_tests  # noqa: E402

PROTOCOL = ROOT / "experiments/coding_agent_phase_a_v6/public-test-execution-v1.json"
LOCK = ROOT / "experiments/coding_agent_phase_a_v6/public-test-execution-v1.lock.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_protocol() -> tuple[dict, dict]:
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    lock = json.loads(LOCK.read_bytes())
    digest = sha256_bytes(protocol_bytes)
    if digest != lock.get("protocol_sha256"):
        raise ValueError("public-test calibration protocol hash mismatch")
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("public-test calibration protocol identity mismatch")
    for relative, expected in lock["implementation_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"locked calibration implementation changed: {relative}")
    return protocol, lock


def image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True, text=True, check=True, timeout=20,
        env={**os.environ, "DOCKER_CONTEXT": "colima"},
    )
    return result.stdout.strip()


def run(output: Path) -> dict:
    protocol, lock = load_protocol()
    if os.environ.get("DOCKER_CONTEXT") != "colima":
        raise RuntimeError("DOCKER_CONTEXT=colima is required")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration output: {output}")
    image = protocol["image"]
    actual_image_id = image_id(image)
    if actual_image_id != protocol["image_id"]:
        raise RuntimeError("local Docker image ID differs from the frozen image identity")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vare-public-test-fixture-") as temp:
        source_root = Path(temp)
        source_workspace = source_root / "candidate"
        source_workspace.mkdir()
        (source_workspace / "src").mkdir()
        test_bytes = protocol["public_test"].encode("utf-8")
        if sha256_bytes(test_bytes) != protocol["public_test_sha256"]:
            raise ValueError("frozen public test bytes do not match the protocol")
        (source_workspace / "test_public.py").write_bytes(test_bytes)

        outcomes = {}
        for name, source in protocol["source_variants"].items():
            variant = source_workspace / "src/value.py"
            variant.write_text(source, encoding="utf-8")
            source_hash = sha256_file(variant)
            if source_hash != protocol["source_variant_sha256"][name]:
                raise ValueError(f"frozen source fixture hash mismatch: {name}")
            test_hash = sha256_file(source_workspace / "test_public.py")
            outcomes[name] = run_public_tests(source_workspace, image, test_hash)
            outcomes[name]["source_sha256"] = source_hash
            outcomes[name]["public_test_sha256"] = test_hash

        baseline = outcomes["broken_baseline"]
        fixed = outcomes["known_fix"]
        passed = (
            baseline.get("status") == "assertion_failure"
            and baseline.get("passed") is False
            and baseline.get("gradeable") is True
            and fixed.get("status") == "passed"
            and fixed.get("passed") is True
            and fixed.get("gradeable") is True
        )
        if not (str(source_root).startswith("/var/folders/") or str(source_root).startswith("/private/var/folders/")):
            raise RuntimeError("fixture workspace was not created in the expected unshared macOS temp root")
        record = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": lock["protocol_sha256"],
            "implementation_sha256": lock["implementation_sha256"],
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": sys.version,
            "docker_context": os.environ["DOCKER_CONTEXT"],
            "image": image,
            "image_id": actual_image_id,
            "source_workspace_class": "macOS-system-temp-outside-home-cache",
            "candidate_network": "none",
            "candidate_workspace_mount": "read-only",
            "results": outcomes,
            "calibration_passed": passed,
            "wall_seconds": round(time.monotonic() - started, 3),
            "model_loaded": False,
            "optimizer_updates": 0,
            "limitations": [
                "Synthetic one-file harness fixture; not a repository task or model rollout.",
                "Validates the host-to-Colima mount and test-result classification path only.",
            ],
        }
    if not record["calibration_passed"]:
        raise RuntimeError("baseline/fix public-test calibration failed")
    output.mkdir(parents=True)
    result_path = output / "calibration.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(result_path)}  calibration.json\n", encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = run(args.output)
    except Exception as exc:
        print(f"calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "calibration_passed": record["calibration_passed"],
        "results": {
            name: {key: value.get(key) for key in ("status", "passed", "gradeable", "returncode")}
            for name, value in record["results"].items()
        },
        "wall_seconds": record["wall_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
