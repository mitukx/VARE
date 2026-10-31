#!/usr/bin/env python3
"""Verify a frozen local-model attempt and enforce launch-time host gates."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_memory_from_vm_stat(text: str) -> int:
    match = re.search(r"page size of (\d+) bytes", text)
    if match is None:
        raise ValueError("vm_stat did not report its page size")
    page_size = int(match.group(1))
    pages: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.strip().rstrip(".").replace(",", "")
        if value.isdigit():
            pages[key.strip()] = int(value)
    if "Pages free" not in pages or "Pages inactive" not in pages:
        raise ValueError("vm_stat is missing free or inactive pages")
    return (pages["Pages free"] + pages["Pages inactive"]) * page_size


def memory_pressure_free_percent(text: str) -> int:
    match = re.search(r"free percentage: (\d+)%", text)
    if match is None:
        raise ValueError("memory_pressure did not report a free percentage")
    return int(match.group(1))


def parse_swap_usage(text: str) -> dict[str, int]:
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}
    result = {}
    for name in ("total", "used", "free"):
        match = re.search(rf"{name} = ([0-9.]+)([KMG])", text)
        if match is None:
            raise ValueError(f"swapusage is missing {name}")
        result[f"{name}_bytes"] = int(float(match.group(1)) * scale[match.group(2)])
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _check_artifact_hashes(root: Path, protocol: dict[str, Any]) -> list[str]:
    failures = []
    frozen_inputs = protocol.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict):
        return ["frozen inputs manifest is malformed"]
    if "task" not in frozen_inputs:
        failures.append("frozen task input manifest is missing")
    for name, value in frozen_inputs.items():
        if name == "task":
            if not isinstance(value, dict):
                failures.append("frozen task input manifest is malformed")
                continue
            for path_key, hash_key in (
                ("task_json_path", "task_json_sha256"),
                ("task_prompt_path", "task_prompt_sha256"),
            ):
                raw_path = value.get(path_key)
                expected = value.get(hash_key)
                if not isinstance(raw_path, str) or not raw_path:
                    failures.append(f"frozen task input manifest missing {path_key}")
                    continue
                if not isinstance(expected, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", expected
                ):
                    failures.append(f"frozen task input manifest missing valid {hash_key}")
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    path = root / path
                if not path.is_file() or sha256_file(path) != expected:
                    failures.append(f"task input hash mismatch: {path}")
            continue
        if not isinstance(value, dict) or "path" not in value or "sha256" not in value:
            continue
        if name == "baseline_source":
            path = Path(protocol["assignment"]["workspace_path"]) / value["path"]
        elif name == "preexisting_ledger_root":
            completed = subprocess.run(
                ["git", "show", f"origin/main:{value['path']}"],
                cwd=root, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                failures.append(f"cannot read frozen Git object for {name}")
                continue
            if hashlib.sha256(completed.stdout).hexdigest() != value["sha256"]:
                failures.append(f"frozen Git object hash mismatch for {name}")
            continue
        else:
            candidate = Path(value["path"])
            path = candidate if candidate.is_absolute() else root / candidate
        if not path.is_file() or sha256_file(path) != value["sha256"]:
            failures.append(f"frozen input hash mismatch: {name} ({path})")

    assignment = protocol["assignment"]["prospective_assignment_record"]
    assignment_path = root / assignment["path"]
    if not assignment_path.is_file() or sha256_file(assignment_path) != assignment["sha256"]:
        failures.append("prospective assignment hash mismatch")
    return failures


def _verify_model_snapshot(root: Path, protocol: dict[str, Any]) -> list[str]:
    manifest_info = protocol["frozen_inputs"]["model_file_manifest"]
    manifest_path = Path(manifest_info["path"])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    failures = []
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_info["sha256"]:
        return ["model file manifest hash mismatch"]
    manifest = _load_json(manifest_path)
    snapshot = Path(manifest["snapshot_path"])
    expected_names = set()
    for item in manifest["files"]:
        path = snapshot / item["path"]
        expected_names.add(item["path"])
        if not path.is_file():
            failures.append(f"model snapshot file missing: {path}")
            continue
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            failures.append(f"model snapshot hash/size mismatch: {path}")
    actual_names = {path.name for path in snapshot.iterdir() if path.is_file()}
    if actual_names != expected_names:
        failures.append("model snapshot file inventory changed")
    return failures


def collect_host_resources(root: Path) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("frozen attempt requires Apple Silicon macOS")
    vm_text = subprocess.run(["vm_stat"], text=True, capture_output=True, check=True).stdout
    pressure_text = subprocess.run(["memory_pressure", "-Q"], text=True, capture_output=True, check=True).stdout
    swap_text = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.platform(),
        "physical_memory_bytes": 34359738368,
        "available_memory_bytes": available_memory_from_vm_stat(vm_text),
        "memory_pressure_free_percent": memory_pressure_free_percent(pressure_text),
        **parse_swap_usage(swap_text),
        "disk_free_bytes": shutil.disk_usage(root).free,
    }


def run_preflight(protocol_path: Path, output_path: Path, root: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    output_path = output_path.resolve()
    protocol = _load_json(protocol_path)
    sidecar = protocol_path.with_suffix(".sha256")
    expected_protocol_sha = sidecar.read_text(encoding="utf-8").split()[0]
    protocol_sha = sha256_file(protocol_path)
    failures = []
    if protocol_sha != expected_protocol_sha:
        failures.append("protocol SHA-256 sidecar mismatch")
    failures.extend(_check_artifact_hashes(root, protocol))
    failures.extend(_verify_model_snapshot(root, protocol))

    assignment = protocol["assignment"]
    workspace = Path(assignment["workspace_path"])
    expected_head = assignment["baseline_revision"]
    head = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "-C", str(workspace), "status", "--porcelain"], text=True, capture_output=True, check=False)
    if head.returncode != 0 or head.stdout.strip() != expected_head:
        failures.append("candidate workspace HEAD mismatch")
    if status.returncode != 0 or status.stdout.strip():
        failures.append("candidate workspace is not clean before attempt")

    resources = collect_host_resources(root)
    policy = protocol["resource_policy"]
    gates = policy["gates"]
    model_manifest_path = root / protocol["frozen_inputs"]["model_file_manifest"]["path"]
    model_manifest = _load_json(model_manifest_path)
    model_config_path = Path(model_manifest["snapshot_path"]) / "config.json"
    config = _load_json(model_config_path)
    layers = config["num_hidden_layers"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    context_tokens = protocol["agent_attempt"]["max_context_tokens"]
    kv_bytes = 2 * layers * kv_heads * head_dim * context_tokens * 2
    reserve_bytes = policy["runtime_activation_reserve_bytes"]
    weight_bytes = model_manifest["model_weight_bytes"]
    estimated_bytes = weight_bytes + kv_bytes + reserve_bytes
    checks = {
        "memory_pressure_free_percent": resources["memory_pressure_free_percent"] >= gates["minimum_memory_pressure_free_percent"],
        "available_memory_bytes": resources["available_memory_bytes"] >= gates["minimum_available_memory_bytes"],
        "swap_used_bytes": resources["used_bytes"] <= gates["maximum_swap_used_bytes"],
        "disk_free_bytes": resources["disk_free_bytes"] >= gates["minimum_disk_free_bytes"],
        "estimated_working_set_and_headroom": estimated_bytes + gates["minimum_weight_kv_reserve_headroom_bytes"] <= resources["available_memory_bytes"],
    }
    failures.extend(f"launch resource gate failed: {name}" for name, passed in checks.items() if not passed)
    result = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "status": "PASSED" if not failures else "NO_GO",
        "failures": failures,
        "resource_snapshot": resources,
        "model_weight_bytes": weight_bytes,
        "estimated_fp16_kv_cache_bytes": kv_bytes,
        "runtime_activation_reserve_bytes": reserve_bytes,
        "estimated_weight_kv_reserve_bytes": estimated_bytes,
        "resource_gate_checks": checks,
        "model_snapshot_hashes_verified": not any("model snapshot" in item for item in failures),
        "frozen_inputs_verified": not any("hash mismatch" in item for item in failures),
        "model_loaded": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run_preflight(args.protocol, args.output, args.repo_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
