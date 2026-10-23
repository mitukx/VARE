"""Re-run v6 public tests with pinned candidate-source provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.grade_transformers_phase_a_v1 import _inspect_frozen_image  # noqa: E402
from scripts.prepare_transformers_dev_cohort_v2 import PACK, TASKS, extract_source  # noqa: E402
from scripts.run_local_agent import run_public_tests  # noqa: E402

LOCK = PACK / "protocol-v2.lock.json"
PROTOCOL = PACK / "protocol-v2.json"
IMAGE = "vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_sha256(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"calibration workspace contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            rows.append(relative.encode() + b"\0" + bytes.fromhex(sha256(path.read_bytes())) + b"\n")
    return sha256(b"".join(rows))


def calibrate(repository: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite public calibration evidence: {output_dir}")
    lock_bytes = LOCK.read_bytes()
    protocol_bytes = PROTOCOL.read_bytes()
    lock = json.loads(lock_bytes)
    protocol = json.loads(protocol_bytes)
    protocol_hash = sha256(protocol_bytes)
    if protocol_hash != lock.get("protocol_sha256") or protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("frozen v6 protocol/lock provenance failed")
    if protocol.get("container_image") != IMAGE:
        raise ValueError("public calibration image differs from the frozen protocol")
    image_id = _inspect_frozen_image(IMAGE)
    output_dir.mkdir(parents=True)
    cache = Path.home() / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for task_id, spec in TASKS.items():
        task_root = PACK / task_id
        task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
        manifest_path = task_root / "baseline-source-manifest-v1.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if sha256(manifest_bytes) != lock["baseline_manifests"][task_id]["sha256"]:
            raise ValueError(f"baseline manifest hash mismatch for {task_id}")
        for kind in ("baseline", "known-fix"):
            with tempfile.TemporaryDirectory(prefix="vare-v6-public-cal-", dir=cache) as temp:
                workspace = Path(temp) / "workspace"
                extract_source(repository, spec["base"], workspace)
                if kind == "known-fix":
                    target_bytes = subprocess.run(
                        ["git", "-C", str(repository), "show", f"{spec['fixed']}:{spec['target']}"],
                        check=True, capture_output=True, timeout=30,
                    ).stdout
                    target = workspace / spec["target"]
                    target.write_bytes(target_bytes)
                    mode = manifest["files"][spec["target"]]["mode"]
                    target.chmod(mode)
                test_bytes = (task_root / "test_public.py").read_bytes()
                (workspace / "test_public.py").write_bytes(test_bytes)
                if kind == "baseline" and tree_sha256(workspace) != manifest["workspace_snapshot_sha256"]:
                    raise ValueError(f"baseline workspace hash mismatch for {task_id}")
                result = run_public_tests(
                    workspace, IMAGE, sha256(test_bytes), memory_mb=int(task["limits"]["memory_mb"]),
                )
                record = {
                    "task_id": task_id,
                    "kind": kind,
                    "passed": result.get("passed") is True,
                    "returncode": result.get("returncode"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "failure": result.get("error"),
                    "task_json_sha256": sha256((task_root / "task.json").read_bytes()),
                    "baseline_manifest_sha256": sha256(manifest_bytes),
                    "baseline_workspace_sha256": manifest["workspace_snapshot_sha256"],
                    "candidate_workspace_sha256": tree_sha256(workspace),
                    "candidate_target_sha256": sha256((workspace / spec["target"]).read_bytes()),
                    "public_test_sha256": sha256(test_bytes),
                    "source_revision": spec["base"] if kind == "baseline" else spec["fixed"],
                    "workspace_derivation": (
                        "git archive at pinned baseline"
                        if kind == "baseline" else
                        "git archive at pinned baseline with only the target bytes overlaid from the pinned fix revision"
                    ),
                    "modified_paths_from_baseline": [] if kind == "baseline" else [spec["target"]],
                    "expected_pass": kind == "known-fix",
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol_hash,
                    "protocol_lock_sha256": sha256(lock_bytes),
                    "calibration_script_sha256": sha256(Path(__file__).read_bytes()),
                    "grading_image": IMAGE,
                    "container_image_id": image_id,
                    "limits": {
                        "timeout_seconds": 45,
                        "memory_mb": int(task["limits"]["memory_mb"]),
                        "cpus": 1.0,
                        "pids": 32,
                        "network": "none",
                        "workspace_mount": "read-only",
                    },
                    "hidden_grader_mounted": False,
                }
                output = output_dir / f"{task_id.rsplit('-', 1)[0]}-{kind}.json"
                output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                records.append(record)
    report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "grading_image": IMAGE,
        "container_image_id": image_id,
        "results": records,
        "calibration_passed": all(item["passed"] == item["expected_pass"] for item in records),
        "limitations": [
            "Public calibration establishes only public regression behavior, not hidden grader correctness.",
            "Known-fix calibration is not a model-generated success.",
        ],
    }
    report_path = output_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            lines.append(f"{sha256(path.read_bytes())}  {path.name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True, help="Local Transformers clone with all frozen commits")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(args.repository.resolve(strict=True), args.output_dir.resolve())
    return 0 if report["calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
