#!/usr/bin/env python3
"""Independently adjudicate the retained MLX-LM #1880 baseline raw log."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.mlsys_heldout_task_pilot_v1 import write_exclusive_atomic  # noqa: E402

TASK_ID = "mlxlm-1880-gptq-group-error-propagation"
V6_PROTOCOL = ROOT / "protocols/mlsys_heldout_extension_v6.json"
RECORD_PATH = ROOT / "results/mlsys-heldout-queue-qualification-v6/run-1/mlxlm-1880-gptq-group-error-propagation.baseline.json"
LOG_PATH = ROOT / "results/mlsys-heldout-queue-qualification-v6/run-1/mlxlm-1880-gptq-group-error-propagation.baseline.log"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record_raw = RECORD_PATH.read_bytes()
    record = json.loads(record_raw)
    protocol_raw = V6_PROTOCOL.read_bytes()
    protocol = json.loads(protocol_raw)
    protocol_task = next(task for task in protocol["tasks"] if task["task_id"] == TASK_ID)
    log_raw = LOG_PATH.read_bytes()
    text = log_raw.decode("utf-8", "replace")
    normalized = text.replace("\r", "\n")
    checks = {
        "task_and_variant_match": record.get("task_id") == TASK_ID and record.get("variant") == "baseline",
        "pinned_protocol_match": record.get("protocol_id") == protocol["protocol_id"] and record.get("protocol_sha256") == sha(protocol_raw),
        "baseline_revision_match": record.get("source_revision") == protocol_task["baseline_revision"],
        "baseline_target_hash_match": record.get("target_sha256") == protocol_task["baseline_target_sha256"],
        "grader_hash_match": record.get("grader_sha256") == protocol_task["grader_sha256"],
        "image_match": record.get("container_image_id") == protocol["environment"]["image_id"],
        "command_match": record.get("command") == protocol_task["test_command"],
        "offline_readonly_unprivileged": record.get("network") == "none" and record.get("container_read_only") is True and record.get("source_and_grader_mounts_read_only") is True and record.get("container_uid") == "65534:65534",
        "raw_log_hash_matches_record": record.get("raw_log_sha256") == sha(log_raw),
        "selected_case_header_present": bool(re.search(r"^test_multiple_groups \(test_gptq\.TestGPTQ\.test_multiple_groups\) \.\.\.", normalized, re.MULTILINE)),
        "selected_case_failure_observed": bool(re.search(r"^FAIL\s*$", normalized, re.MULTILINE)),
        "one_test_ran": bool(re.search(r"^Ran 1 test in [0-9.]+s$", normalized, re.MULTILINE)),
        "unittest_failure_summary": "FAILED (failures=1)" in normalized,
        "numerical_reference_mismatch_present": "max abs diff 0.3834570646286011" in normalized,
        "expected_nonzero_exit": record.get("exit_code") == 1,
        "cgroup_measurement_bounded": record.get("cgroup_memory_peak_bytes", 0) > 0 and record.get("cgroup_memory_max_bytes") == protocol["controls"]["memory_limit_bytes"] and record["cgroup_memory_peak_bytes"] <= record["cgroup_memory_max_bytes"],
    }
    result = {
        "adjudication_id": "mlxlm-1880-baseline-raw-log-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": TASK_ID,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha(protocol_raw),
        "original_runner_sha256": record.get("runner_sha256"),
        "original_record_path": RECORD_PATH.relative_to(ROOT).as_posix(),
        "original_record_sha256": sha(record_raw),
        "original_record_reports_test_body_executed": record.get("test_body_executed"),
        "original_raw_log_path": LOG_PATH.relative_to(ROOT).as_posix(),
        "original_raw_log_sha256": sha(log_raw),
        "adjudicated_outcome": "EXPECTED_BASELINE_FAILURE" if all(checks.values()) else "UNRESOLVED",
        "checks": checks,
        "interpretation": "The pinned multi-group test executed exactly once and failed at the independent column-by-column reference comparison (max absolute difference 0.3834570646286011). The v6 runner incorrectly reported no parsed test outcome because tqdm progress output inserted carriage returns between the unittest case header and its standalone FAIL marker. This adjudication reads retained bytes only; it does not rerun the test.",
        "rerun_performed": False,
        "model_or_agent_execution": False,
        "training": False,
    }
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    write_exclusive_atomic(args.output, raw)
    print(json.dumps({"outcome": result["adjudicated_outcome"], "checks_passed": sum(checks.values()), "checks_total": len(checks), "sha256": sha(raw)}, sort_keys=True))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
