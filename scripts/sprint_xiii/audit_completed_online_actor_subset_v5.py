#!/usr/bin/env python3
"""Audit every completed case in an interrupted Sprint XIII campaign.

This is audit v5 for campaign v4. It reuses the v4 hash-bound independent
token/update auditor but does not require whole-campaign completion receipts.
Missing cases remain explicit and prevent a full-campaign PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import audit_online_actor_v3 as audit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str, findings: list[str]) -> None:
    if not condition:
        findings.append(message)


def expected_case_ids(protocol: dict) -> list[str]:
    return [f"seed-{seed}-{arm}" for seed, arm in protocol["execution_order"]]


def completed_case_roots(root: Path) -> list[Path]:
    cases_root = root / "cases"
    if not cases_root.is_dir():
        return []
    return sorted(
        path
        for path in cases_root.iterdir()
        if path.is_dir() and (path / "case-result.json").is_file()
    )


def audit_subset(protocol_path: Path, root: Path, auditor_path: Path) -> dict:
    started = time.monotonic()
    p = audit.load_json(protocol_path)
    audit.validate_protocol_contract(p)
    findings: list[str] = []
    protocol_sha = digest(protocol_path)
    protocol_sidecar = audit.load_json(Path(str(protocol_path) + ".sha256"))
    check(
        protocol_sha == protocol_sidecar.get("sha256"),
        "protocol sidecar mismatch",
        findings,
    )
    check(
        digest(root / "protocol-v1.json")
        == audit.load_json(root / "protocol-v1.json.sha256").get("sha256"),
        "run protocol copy mismatch",
        findings,
    )
    check(
        digest(Path(p["model"]["weight_file"])) == p["model"]["weight_sha256"],
        "base model weights changed",
        findings,
    )
    for rel, expected in p["source_hashes"].items():
        check(
            digest(Path(p["repository_root"]) / rel) == expected,
            f"frozen VARE source changed: {rel}",
            findings,
        )
    for rel, expected in p["rvl_source_hashes"].items():
        check(
            digest(Path(p["rvl_source_root"]) / rel) == expected,
            f"frozen RVL source changed: {rel}",
            findings,
        )
    for rel, expected in p["model"]["snapshot_files"].items():
        check(
            digest(Path(p["model"]["snapshot_path"]) / rel) == expected,
            f"model snapshot changed: {rel}",
            findings,
        )
    rvl_head = subprocess.run(
        ["git", "-C", p["rvl_source_root"], "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        rvl_head.returncode == 0 and rvl_head.stdout.strip() == p["rvl_commit"],
        "RVL revision changed",
        findings,
    )
    vare_ancestor = subprocess.run(
        [
            "git",
            "-C",
            p["repository_root"],
            "merge-base",
            "--is-ancestor",
            p["source_commit"],
            "HEAD",
        ],
        check=False,
    )
    check(
        vare_ancestor.returncode == 0,
        "protocol source commit is not an ancestor",
        findings,
    )
    check(
        p["limits"]["max_worker_concurrency"] == 1
        and p["limits"]["max_learner_concurrency"] == 1
        and p["limits"]["max_concurrent_children"] == 2,
        "frozen concurrency caps changed",
        findings,
    )
    expected_calibration = audit.calibrate(p)
    actual_calibration = audit.load_json(root / "verifier-calibration.json")
    check(
        expected_calibration
        == {key: actual_calibration.get(key) for key in expected_calibration},
        "enumerated verifier calibration mismatch",
        findings,
    )

    expected = expected_case_ids(p)
    roots = completed_case_roots(root)
    completed_ids = [path.name for path in roots]
    unexpected = sorted(set(completed_ids) - set(expected))
    missing = sorted(set(expected) - set(completed_ids))
    check(not unexpected, f"unexpected completed cases: {unexpected}", findings)

    failure_path = root / "failure.json"
    check(failure_path.is_file(), "missing campaign stopping receipt", findings)
    failure = audit.load_json(failure_path) if failure_path.is_file() else {}
    check(
        failure.get("status") == "RESOURCE_ABORT"
        and "wall-clock cap" in failure.get("error", ""),
        "campaign stop was not the preregistered wall cap",
        findings,
    )
    run_status_complete = (root / "run-status.json").is_file()
    resource_summary_complete = (root / "resource-summary.json").is_file()

    import psutil

    limits = p["limits"]
    vm_before = psutil.virtual_memory()
    swap_before = psutil.swap_memory().used
    check(
        vm_before.available >= limits["min_available_bytes"],
        "subset audit preflight below frozen available-memory minimum",
        findings,
    )
    if findings:
        return {
            "audit_version": "online-actor-independent-audit-v5-completed-subset",
            "status": "FAIL_PRE_MODEL_AUDIT",
            "findings": findings,
            "protocol_sha256": protocol_sha,
            "completed_case_ids": completed_ids,
            "incomplete_case_ids": missing,
            "model_forward_count": 0,
            "completed_unix_ns": time.time_ns(),
        }

    torch, tok, model, action_ids, eos_id, support_ids, support_mask, fixed_bias = (
        audit.load_snapshot(p)
    )
    baseline_eval = {}
    case_reports = []
    max_rss = 0
    min_available = vm_before.available
    max_swap_growth = 0

    def resource_check(label: str) -> None:
        nonlocal max_rss, min_available, max_swap_growth
        rss = psutil.Process().memory_info().rss
        available = psutil.virtual_memory().available
        swap_growth = psutil.swap_memory().used - swap_before
        max_rss = max(max_rss, rss)
        min_available = min(min_available, available)
        max_swap_growth = max(max_swap_growth, swap_growth)
        if rss > limits["max_child_rss_bytes"]:
            raise RuntimeError(f"audit resource abort: RSS cap at {label}")
        if available < limits["hard_min_available_bytes"]:
            raise RuntimeError(f"audit resource abort: hard memory floor at {label}")
        if swap_growth > limits["max_swap_growth_bytes"]:
            raise RuntimeError(f"audit resource abort: swap growth cap at {label}")
        if time.monotonic() - started > limits["audit_wall_seconds"]:
            raise RuntimeError(f"audit resource abort: wall cap at {label}")

    try:
        for seed in p["seeds"]:
            baseline_root = root / "baseline" / f"seed-{seed}"
            baseline_eval[str(seed)] = audit.audit_eval_file(
                p,
                baseline_root,
                torch,
                tok,
                model,
                action_ids,
                eos_id,
                support_ids,
                support_mask,
                fixed_bias,
                findings,
            )
            resource_check(f"baseline-{seed}")
        for case_root in roots:
            case_reports.append(
                audit.audit_case(
                    p,
                    case_root,
                    torch,
                    tok,
                    model,
                    action_ids,
                    eos_id,
                    support_ids,
                    support_mask,
                    fixed_bias,
                    findings,
                )
            )
            resource_check(case_root.name)
    except BaseException as exc:
        failure_report = {
            "audit_version": "online-actor-independent-audit-v5-completed-subset",
            "status": "AUDIT_ABORT",
            "findings": findings,
            "audit_error_type": type(exc).__name__,
            "audit_error": str(exc),
            "protocol_sha256": protocol_sha,
            "completed_case_ids": completed_ids,
            "incomplete_case_ids": missing,
            "completed_subset_cases_audited": [row["case_id"] for row in case_reports],
            "resource_usage": {
                "peak_rss_bytes": max_rss,
                "min_available_bytes": min_available,
                "max_swap_growth_bytes": max_swap_growth,
            },
            "elapsed_seconds": time.monotonic() - started,
            "completed_unix_ns": time.time_ns(),
        }
        raise RuntimeError(json.dumps(failure_report, sort_keys=True)) from exc

    by_case = {row["case_id"]: row for row in case_reports}
    d_e_comparisons = []
    for seed in p["seeds"]:
        d_id, e_id = f"seed-{seed}-D", f"seed-{seed}-E"
        if d_id not in by_case or e_id not in by_case:
            findings.append(f"missing completed D/E pair for seed {seed}")
            continue
        d_root, e_root = root / "cases" / d_id, root / "cases" / e_id
        d_active, e_active = (
            audit.load_json(d_root / "active.json"),
            audit.load_json(e_root / "active.json"),
        )
        _, d_state = audit.checkpoint(
            d_root, d_active["checkpoint"], d_active["checkpoint_sha256"], findings
        )
        _, e_state = audit.checkpoint(
            e_root, e_active["checkpoint"], e_active["checkpoint_sha256"], findings
        )
        same_bias = (
            d_state is not None
            and e_state is not None
            and torch.equal(d_state["bias"], e_state["bias"])
        )
        same_rng = (
            d_state is not None
            and e_state is not None
            and torch.equal(d_state["torch_rng"], e_state["torch_rng"])
        )
        same_optimizer = False
        if d_state is not None and e_state is not None:
            compare_findings: list[str] = []
            audit.compare_optimizer(
                torch,
                d_state["optimizer"],
                e_state["optimizer"],
                p,
                compare_findings,
                f"seed-{seed}-D-vs-E",
            )
            findings.extend(compare_findings)
            same_optimizer = not compare_findings
        d_e_comparisons.append(
            {
                "seed": seed,
                "adapter_equal": same_bias,
                "optimizer_equal": same_optimizer,
                "rng_equal": same_rng,
            }
        )
        check(
            same_bias and same_optimizer and same_rng,
            f"seed {seed}: D/E recovered state mismatch",
            findings,
        )

    all_faults = sorted({fault for item in case_reports for fault in item["faults"]})
    required_faults = set(p["fault_plan"]["required_faults"])
    missing_faults = sorted(required_faults - set(all_faults))
    check(not missing_faults, f"missing fault cutpoints: {missing_faults}", findings)
    paired = []
    for row in case_reports:
        baseline = baseline_eval.get(str(row["seed"]), {})
        deltas = {
            task_id: row["trusted_eval_by_task"][task_id]["mean_trusted_reward"]
            - baseline[task_id]["mean_trusted_reward"]
            for task_id in p["evaluation"]["task_ids"]
            if task_id in row["trusted_eval_by_task"] and task_id in baseline
        }
        paired.append(
            {
                "case_id": row["case_id"],
                "per_task_delta": deltas,
                "mean_task_delta": sum(deltas.values()) / len(deltas)
                if deltas
                else None,
            }
        )
    case_resources = [
        audit.load_json(path / "case-result.json")["resource_peak"] for path in roots
    ]
    check(
        all(
            r["peak_child_rss_bytes"] <= limits["max_child_rss_bytes"]
            and r["minimum_host_available_bytes"] >= limits["hard_min_available_bytes"]
            and r["swap_growth_bytes"] <= limits["max_swap_growth_bytes"]
            for r in case_resources
        ),
        "one or more completed cases violated a resource limit",
        findings,
    )
    report = {
        "schema_version": 1,
        "audit_version": "online-actor-independent-audit-v5-completed-subset",
        "status": "PASS_COMPLETED_SUBSET" if not findings else "FAIL_COMPLETED_SUBSET",
        "campaign_complete": False,
        "campaign_stop": failure,
        "protocol_sha256": protocol_sha,
        "auditor_sha256": digest(auditor_path),
        "case_auditor_sha256": digest(Path(audit.__file__)),
        "source_commit": p["source_commit"],
        "model": p["model"],
        "expected_case_count": len(expected),
        "completed_case_count": len(completed_ids),
        "completed_case_ids": completed_ids,
        "incomplete_case_ids": missing,
        "partial_case_directories": sorted(
            path.name
            for path in (root / "cases").iterdir()
            if path.is_dir() and not (path / "case-result.json").exists()
        ),
        "findings": findings,
        "baseline_trusted_eval": baseline_eval,
        "completed_case_results": case_reports,
        "paired_task_deltas": paired,
        "d_vs_e_recovery_comparisons": d_e_comparisons,
        "fault_cutpoints_seen": all_faults,
        "missing_fault_cutpoints": missing_faults,
        "effective_update_count": sum(row["effective_updates"] for row in case_reports),
        "optimizer_computation_count_including_retries": sum(
            row["optimizer_computations_including_retries"] for row in case_reports
        ),
        "proxy_vs_trusted_training": {
            f"v{version}": {
                "n": sum(
                    row["proxy_vs_trusted_training"][f"v{version}"]["n"]
                    for row in case_reports
                ),
                "false_positive": sum(
                    row["proxy_vs_trusted_training"][f"v{version}"]["false_positive"]
                    for row in case_reports
                ),
                "false_negative": sum(
                    row["proxy_vs_trusted_training"][f"v{version}"]["false_negative"]
                    for row in case_reports
                ),
            }
            for version in (0, 1)
        },
        "resource_usage": {
            "peak_auditor_rss_bytes": max_rss,
            "minimum_host_available_bytes": min_available,
            "maximum_swap_growth_bytes": max_swap_growth,
            "auditor_wall_seconds": time.monotonic() - started,
            "completed_case_peak_child_rss_bytes": max(
                (r["peak_child_rss_bytes"] for r in case_resources), default=0
            ),
            "completed_case_minimum_host_available_bytes": min(
                (r["minimum_host_available_bytes"] for r in case_resources), default=0
            ),
            "completed_case_maximum_swap_growth_bytes": max(
                (r["swap_growth_bytes"] for r in case_resources), default=0
            ),
        },
        "full_campaign_receipts_present": {
            "run_status": run_status_complete,
            "resource_summary": resource_summary_complete,
        },
        "completed_unix_ns": time.time_ns(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--auditor", type=Path, default=Path(__file__))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_subset(
        args.protocol.resolve(), args.output.parent.resolve(), args.auditor.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_cases": report["completed_case_count"],
                "effective_updates": report["effective_update_count"],
                "findings": report["findings"],
            },
            ensure_ascii=False,
        )
    )
    if report["status"] != "PASS_COMPLETED_SUBSET":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
