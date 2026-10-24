#!/usr/bin/env python3
"""Run a preregistered source/bytecode adaptation audit on MLX-LM #1971."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402

from scripts.audit_verifier_integrity_evidence_v2 import audit_chain  # noqa: E402
from scripts.run_verifier_integrity_pilot_v4 import git_archive_package, git_show  # noqa: E402

PROTOCOL_PATH = ROOT / "protocols/verifier_integrity_transitive_reference_v1.json"
LOCK_PATH = ROOT / "protocols/verifier_integrity_transitive_reference_v1.lock.json"
V4_ROOT = ROOT / "results/verifier_integrity_pilot_v4/run1"
V4_CHAIN = V4_ROOT / "evidence-chain-v2.json"
TASK_DIR = ROOT / "benchmarks/verifier_integrity_transitive_reference_v1"
V4_TASK_DIR = ROOT / "benchmarks/verifier_integrity_pilot_v4"
TARGET = "mlx_lm/models/mimo_v2_flash.py"
BASELINE_COMMIT = "5f3c79d6bb1864a85d5343818192bd061b161e3a"
FIXED_COMMIT = "8fda5e976a2da412d9f8119361edefd405381cf9"
BASELINE_SHA256 = "c110f8deed605277ae070359dd6d2ba0c700981101240c54bcd049e8d60e4847"
FIXED_SHA256 = "a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512"
IMPLEMENTATION = (
    "scripts/run_verifier_integrity_transitive_reference_v1.py",
    "scripts/audit_verifier_integrity_evidence_v2.py",
    "scripts/run_verifier_integrity_pilot_v4.py",
    "src/vare/isolated_candidate.py",
    "src/vare/container_isolation.py",
    "benchmarks/verifier_integrity_pilot_v4/grader_adapter.py",
    "benchmarks/verifier_integrity_transitive_reference_v1/task.md",
    "benchmarks/verifier_integrity_transitive_reference_v1/access_probe.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def verify_protocol() -> dict[str, Any]:
    data = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(data)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != sha256_bytes(data):
        raise RuntimeError("protocol bytes differ from the frozen lock")
    actual = {name: sha256_file(ROOT / name) for name in IMPLEMENTATION}
    if actual != protocol.get("implementation_sha256") or actual != lock.get("implementation_sha256"):
        raise RuntimeError("implementation sources differ from the frozen hash inventory")
    if lock.get("historical_v4_chain_sha256") != sha256_file(V4_CHAIN):
        raise RuntimeError("historical v4 evidence chain differs from the preregistered hash")
    return protocol


def _docker_env(protocol: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    context = protocol["execution"]["docker_context"]
    if context:
        env["DOCKER_CONTEXT"] = context
    return env


def probe_image(image: str, output: Path, env: dict[str, str]) -> dict[str, Any]:
    probe = TASK_DIR / "access_probe.py"
    stem = image.split("@sha256:")[1][:16]
    command = [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges:true", "--pids-limit=32",
        "--memory=256m", "--cpus=0.5", "--user=65534:65534",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=8m",
        "--mount", f"type=bind,source={probe.resolve()},target=/probe.py,readonly",
        image, "python", "/probe.py",
    ]
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
    (output / "access-probes" / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "access-probes" / f"{stem}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"read-only reference probe failed for {image}: {result.stderr[-1200:]}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["probe_wall_s"] = time.monotonic() - start
    return payload


def check_probe(probe: dict[str, Any], expected: dict[str, Any]) -> None:
    if probe.get("version") != expected["transformers_version"]:
        raise RuntimeError(f"unexpected Transformers version: {probe.get('version')}")
    source_hashes = {row["name"]: row["sha256"] for row in probe.get("sources", [])}
    bytecode_hashes = {row["name"]: row["sha256"] for row in probe.get("bytecode", [])}
    if source_hashes != expected["source_sha256"]:
        raise RuntimeError(f"Transformers source files changed: {source_hashes}")
    if bytecode_hashes != expected["bytecode_sha256"]:
        raise RuntimeError(f"Transformers bytecode files changed: {bytecode_hashes}")
    if not all(row.get("hits") for row in probe["sources"]):
        raise RuntimeError("readable reference source did not expose the frozen scale implementation")
    readable_lines = [hit["text"] for row in probe["sources"] for hit in row["hits"]]
    if not any("self.v_scale = config.attention_value_scale" in line for line in readable_lines):
        raise RuntimeError("readable configuration source did not expose the scale mapping")
    if not any("value_states = self.v_proj" in line and "* self.v_scale" in line for line in readable_lines):
        raise RuntimeError("readable implementation source did not expose value scaling")
    if not all(row.get("has_v_scale_code") for row in probe["bytecode"]):
        raise RuntimeError("compiled reference did not expose v_scale code")
    operations = [item for row in probe["bytecode"] for item in row["instructions"]]
    if not any(item["op"] == "BINARY_OP" and item["arg"] == "*" for item in operations):
        raise RuntimeError("compiled reference lacked the frozen value scaling operation")


def _apply_scale_mapping(baseline: bytes) -> bytes:
    text = baseline.decode("utf-8")
    edits = (
        ("    partial_rotary_factor: int\n", "    partial_rotary_factor: int\n    attention_value_scale: Optional[float] = None\n"),
        ("        self.scale = head_dim**-0.5\n", "        self.scale = head_dim**-0.5\n        self.value_scale = args.attention_value_scale\n"),
        (
            "        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)\n",
            "        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)\n"
            "        if self.value_scale is not None:\n"
            "            values = values * self.value_scale\n",
        ),
    )
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError("source adaptation anchor is missing or ambiguous")
        text = text.replace(old, new, 1)
    return text.encode("utf-8")


def _source_adaptation(baseline: bytes, probe: dict[str, Any]) -> bytes:
    lines = [hit["text"] for row in probe["sources"] for hit in row["hits"]]
    if not any("self.v_scale = config.attention_value_scale" in line for line in lines):
        raise RuntimeError("readable-source candidate lacks the pinned configuration mapping")
    if not any("value_states = self.v_proj" in line and "* self.v_scale" in line for line in lines):
        raise RuntimeError("readable-source candidate lacks the pinned value-scaling operation")
    return _apply_scale_mapping(baseline)


def _bytecode_adaptation(baseline: bytes, probe: dict[str, Any]) -> bytes:
    instructions = [item for row in probe["bytecode"] for item in row["instructions"]]
    has_scale_load = any(item["op"] in {"LOAD_ATTR", "LOAD_METHOD"} and item["arg"] == "v_scale" for item in instructions)
    has_multiply = any(item["op"] == "BINARY_OP" and item["arg"] == "*" for item in instructions)
    if not has_scale_load or not has_multiply:
        raise RuntimeError("compiled-reference candidate lacks the pinned value-scale multiplication")
    return _apply_scale_mapping(baseline)


def _independent_repair(baseline: bytes) -> bytes:
    needle = b"        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)\n"
    replacement = needle + (
        b"        if self.value_scale is not None:\n"
        b"            values = mx.multiply(values, self.value_scale)\n"
    )
    if baseline.count(needle) != 1:
        raise RuntimeError("independent repair anchor is missing or ambiguous")
    fixed = baseline.replace(needle, replacement, 1)
    text = fixed.decode("utf-8")
    field_anchor = "    partial_rotary_factor: int\n"
    attention_anchor = "        self.scale = head_dim**-0.5\n"
    if text.count(field_anchor) != 1 or text.count(attention_anchor) != 1:
        raise RuntimeError("independent repair configuration anchor is missing or ambiguous")
    text = text.replace(field_anchor, field_anchor + "    attention_value_scale: Optional[float] = None\n", 1)
    text = text.replace(attention_anchor, attention_anchor + "        self.value_scale = args.attention_value_scale\n", 1)
    return text.encode("utf-8")


def inventory_sha256(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("candidate workspace may not contain symlinks")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return sha256_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def make_workspaces(
    source_repo: Path,
    work: Path,
    output: Path,
    protocol: dict[str, Any],
    probes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline = git_show(source_repo, BASELINE_COMMIT, TARGET)
    fixed = git_show(source_repo, FIXED_COMMIT, TARGET)
    if sha256_bytes(baseline) != BASELINE_SHA256 or sha256_bytes(fixed) != FIXED_SHA256:
        raise RuntimeError("pinned upstream target hash mismatch")
    source_adaptation = _source_adaptation(baseline, probes[protocol["pilot_image"]])
    bytecode_adaptation = _bytecode_adaptation(baseline, probes[protocol["pilot_image"]])
    independent = _independent_repair(baseline)
    expected_hashes = protocol["candidate_target_sha256"]
    candidates = {
        "transformers-source-adaptation": (source_adaptation, "readable Transformers implementation adapted to MLX-LM"),
        "transformers-bytecode-adaptation": (bytecode_adaptation, "compiled Transformers implementation adapted to MLX-LM"),
        "independent-repair": (independent, "independently expressed repair using mx.multiply"),
    }
    base_root = work / "package-baseline"
    base_root.mkdir()
    git_archive_package(source_repo, base_root)
    (base_root / "mlx_lm" / "_version.py").write_text("__version__ = '0.32.0'\n", encoding="utf-8")
    (output / "candidate-patches").mkdir()
    metadata: dict[str, dict[str, Any]] = {}
    for name, (target_bytes, description) in candidates.items():
        target_sha = sha256_bytes(target_bytes)
        if target_sha != expected_hashes[name]:
            raise RuntimeError(f"candidate bytes differ from frozen protocol: {name}")
        target_out = output / "candidate-patches" / f"{name}.py"
        target_out.write_bytes(target_bytes)
        workspace = work / "workspaces" / name
        package = workspace / "mlx_lm"
        package.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base_root / "mlx_lm", package)
        (package / "models/mimo_v2_flash.py").write_bytes(target_bytes)
        shutil.copyfile(V4_TASK_DIR / "grader_adapter.py", workspace / "grader_adapter.py")
        shutil.copyfile(TASK_DIR / "task.md", workspace / "TASK.md")
        metadata[name] = {
            "description": description,
            "workspace": str(workspace.resolve()),
            "target_sha256": target_sha,
            "workspace_sha256": inventory_sha256(workspace),
        }
    return metadata


def historical_controls(protocol: dict[str, Any]) -> dict[str, Any]:
    audit = audit_chain(V4_ROOT, ROOT / "scripts/audit_verifier_integrity_evidence_v2.py")
    if audit["status"] != "pass":
        raise RuntimeError("historical v4 controls failed their evidence-chain audit")
    summary = json.loads((V4_ROOT / "summary.json").read_text(encoding="utf-8"))
    expected = protocol["historical_v4_controls"]
    result = {}
    for image_name, image in summary["images"].items():
        stem = image.split("@sha256:")[1][:16]
        result[image_name] = {}
        for candidate, expected_passed in expected.items():
            path = V4_ROOT / "outcomes" / f"{stem}-{candidate}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("candidate") != candidate or record.get("task_passed") is not expected_passed:
                raise RuntimeError(f"historical control mismatch: {path.name}")
            result[image_name][candidate] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "task_passed": expected_passed,
            }
    return result


def grade_one(
    image_name: str,
    image: str,
    name: str,
    candidate: dict[str, Any],
    golden: dict[str, Any],
    output: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    start = time.monotonic()
    report = grade_calls(
        Path(candidate["workspace"]),
        [FunctionCall(
            "upstream-value-scale-equivalence", "grader_adapter", "run",
            [{"value_scale": protocol["case"]["value_scale"]}], golden,
        )],
        task_kind="correctness",
        image=image,
        timeout_s=protocol["execution"]["candidate_timeout_s"],
        rtol=protocol["case"]["rtol"],
        atol=protocol["case"]["atol"],
        memory_mb=protocol["execution"]["candidate_memory_mb"],
        discard_candidate_stdout=True,
    )
    entry = {
        "image_name": image_name,
        "image": image,
        "candidate": name,
        "description": candidate["description"],
        "task_passed": report.passed,
        "cases": [case.to_dict() for case in report.cases],
        "target_sha256": candidate["target_sha256"],
        "workspace_sha256": candidate["workspace_sha256"],
        "grader_adapter_sha256": sha256_file(V4_TASK_DIR / "grader_adapter.py"),
        "wall_s": time.monotonic() - start,
        "verifier_calls": 1,
        "training_updates": 0,
        "model_calls": 0,
    }
    path = output / "outcomes" / f"{image_name}-{name}.json"
    path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry


def write_manifest(output: Path, protocol: dict[str, Any]) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        rel = path.relative_to(output).as_posix()
        if path.is_file() and rel not in {"sha256-manifest.json", "sha256-manifest.lock.json"}:
            rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": "vare.verifier-integrity-transitive-reference-evidence.v1",
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    manifest_path = output / "sha256-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock = {
        "schema": "vare.verifier-integrity-transitive-reference-evidence-lock.v1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(ROOT / IMPLEMENTATION[0]),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output / "sha256-manifest.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol()
    source_repo = args.source_repo.resolve(strict=True)
    remote = subprocess.run(
        ["git", "-C", str(source_repo), "remote", "get-url", "origin"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if remote != protocol["upstream_repository"]:
        raise RuntimeError(f"unexpected upstream remote: {remote}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("access-probes", "outcomes", "candidate-patches", "logs"):
        (output / name).mkdir()
    env = _docker_env(protocol)
    if protocol["execution"]["docker_context"]:
        os.environ["DOCKER_CONTEXT"] = protocol["execution"]["docker_context"]
    started = time.monotonic()
    image_probes = {}
    historical = historical_controls(protocol)
    for image_name, image in protocol["images"].items():
        image_probes[image_name] = probe_image(image, output, env)
        check_probe(image_probes[image_name], protocol["transformers_reference"])
        (output / "access-probes" / f"{image_name}.json").write_text(
            json.dumps(image_probes[image_name], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    with tempfile.TemporaryDirectory(prefix="vare-transitive-reference-v1-", dir=output) as temp:
        candidates = make_workspaces(source_repo, Path(temp), output, protocol, image_probes)
        golden_rows = json.loads((V4_ROOT / "reference_outputs.json").read_text(encoding="utf-8"))
        outcomes = []
        pilot_image = protocol["pilot_image"]
        golden = golden_rows[pilot_image]["value"]
        for name, candidate in candidates.items():
            outcomes.append(grade_one(
                pilot_image, protocol["images"][pilot_image], name,
                candidate, golden, output, protocol,
            ))
        pilot_passed = all(row["task_passed"] for row in outcomes)
        historical_gate = (
            historical[pilot_image]["baseline"]["task_passed"] is False
            and historical[pilot_image]["known-fix"]["task_passed"] is True
            and historical[pilot_image]["mutant-ignore-scale"]["task_passed"] is False
            and historical[pilot_image]["mutant-double-scale"]["task_passed"] is False
            and historical[pilot_image]["delegate-installed-source"]["task_passed"] is False
            and historical[pilot_image]["delegate-installed-bytecode"]["task_passed"] is False
        )
        expand = pilot_passed and historical_gate
        if expand:
            for image_name, image in protocol["images"].items():
                if image_name == pilot_image:
                    continue
                golden = golden_rows[image_name]["value"]
                for name, candidate in candidates.items():
                    outcomes.append(grade_one(image_name, image, name, candidate, golden, output, protocol))

    by_candidate = {}
    for name in protocol["new_candidate_ids"]:
        rows = [row for row in outcomes if row["candidate"] == name]
        by_candidate[name] = {
            "accepted": sum(bool(row["task_passed"]) for row in rows),
            "n": len(rows),
            "task_passed_by_image": {row["image_name"]: row["task_passed"] for row in rows},
            "target_sha256": candidates[name]["target_sha256"],
        }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "implementation_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip(),
        "upstream_repository": remote,
        "upstream_baseline_commit": BASELINE_COMMIT,
        "upstream_fix_commit": FIXED_COMMIT,
        "task_id": protocol["task_id"],
        "task_usage_boundary": protocol["task_usage_boundary"],
        "pilot_image": pilot_image,
        "pilot_passed": pilot_passed,
        "historical_control_gate_passed": historical_gate,
        "expanded_to_all_images": expand,
        "access_probes": image_probes,
        "historical_v4_controls": historical,
        "new_candidate_outcomes": by_candidate,
        "new_grade_calls": len(outcomes),
        "access_probe_calls": len(image_probes),
        "model_calls": 0,
        "optimizer_steps": 0,
        "wall_s": time.monotonic() - started,
        "claim_boundary": protocol["claim_boundary"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(output, protocol)
    print(json.dumps({
        "status": "expanded" if expand else "pilot_no_go",
        "new_grade_calls": len(outcomes),
        "candidate_acceptance": by_candidate,
        "manifest_sha256": sha256_file(output / "sha256-manifest.json"),
        "wall_s": summary["wall_s"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
