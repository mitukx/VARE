#!/usr/bin/env python3
"""Export Sprint IX public evidence while keeping hidden grader outputs private."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public-evidence-v1"
ARMS = ("A_autonomous", "B_progress_aware", "C_localized")
RAW_ATTEMPT_FILES = (
    "agent.json", "events.jsonl", "invocation.json", "preflight.json",
    "resource-samples.jsonl", "candidate.patch", "runtime_snapshot",
    "stdout.log", "stderr.log",
)
PUBLIC_FILES = (
    "protocol-v3.json", "protocol-v3.sha256", "protocol-v3-freeze-record.json",
    "protocol-v1-invalidation.json", "protocol-v1-invalidation-complete.json",
    "protocol-v2-invalidation.json", "campaign-summary-v2-preinference-failure.json",
    "model-snapshot-manifest.json", "task-source-qualification-v3.json",
    "prior-exposure-scan.json", "localization-v3.json",
    "reproduction/run-agent-campaign-v3.py", "reproduction/preflight-v3-runner.py",
    "reproduction/freeze-protocol-v3.py",
    "task/TASK.md", "task/task.json", "task/test_public.py",
)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def copy_file(rel: str) -> None:
    src = ROOT / rel
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

def public_attempt(arm: str) -> dict:
    src = ROOT / "attempts-v3" / arm
    dst = OUT / "attempts-v3" / arm
    dst.mkdir(parents=True, exist_ok=True)
    for name in RAW_ATTEMPT_FILES:
        p = src / name
        if p.is_file():
            shutil.copyfile(p, dst / name)
    summary = json.loads((src / "attempt-summary.json").read_text())
    grade = summary.get("independent_grade") or {}
    public_grade = {
        "grader": grade.get("grader"),
        "case_count": grade.get("case_count"),
        "passed": grade.get("passed"),
        "source": grade.get("source"),
        "candidate_source_sha256": summary.get("candidate_source_sha256_after"),
        "raw_receipt_sha256_local_only": sha(src / "independent-grader.json"),
        "case_results_withheld": True,
        "grader_source_withheld": True,
    }
    safe = {k: v for k, v in summary.items() if k != "independent_grade"}
    safe["independent_grade"] = {
        "grader": grade.get("grader"), "case_count": grade.get("case_count"),
        "passed": grade.get("passed"), "source": grade.get("source"),
        "case_results_withheld": True,
    }
    (dst / "attempt-summary-public.json").write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n")
    (dst / "independent-grader-receipt-public.json").write_text(json.dumps(public_grade, indent=2, sort_keys=True) + "\n")
    return {
        "arm": arm, "status": summary.get("status"),
        "stop_reason": summary.get("agent_stop_reason"),
        "candidate_changed": summary.get("candidate_changed"),
        "patch_bytes": summary.get("candidate_patch_bytes"),
        "event_chain_status": summary.get("event_chain_status"),
        "event_count": summary.get("event_count"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "peak_sampled_process_tree_rss_bytes": summary.get("peak_sampled_process_tree_rss_bytes"),
        "swap_growth_bytes": summary.get("swap_growth_bytes"),
        "resource_abort": summary.get("resource_abort"),
        "public_test": "NOT_RUN_NO_EDIT",
        "independent_grade": {"case_count": grade.get("case_count"), "passed": grade.get("passed")},
        "verified_repair": summary.get("verified_repair"),
        "raw_trajectory_files": list(RAW_ATTEMPT_FILES),
        "hidden_grader_source_and_case_results_withheld": True,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if not args.verify:
        if OUT.exists():
            raise SystemExit(f"refusing to overwrite existing export: {OUT}")
        OUT.mkdir(parents=True)
        for rel in PUBLIC_FILES:
            copy_file(rel)
        controls = json.loads((ROOT / "qualification-controls-v2.json").read_text())
        states = controls["states"]
        controls_public = {
            "schema_version": 1,
            "family_id": "mlx-lm-prefill-history-dev-01",
            "baseline": {"public_pass": states["baseline"]["public"]["passed"], "hidden_pass": states["baseline"]["hidden"]["returncode"] == 0, "source_sha256": states["baseline"]["hidden"]["candidate_source_sha256"]},
            "known_fix": {"public_pass": states["fixed"]["public"]["passed"], "hidden_pass": states["fixed"]["hidden"]["returncode"] == 0, "source_sha256": states["fixed"]["hidden"]["candidate_source_sha256"]},
            "incorrect_mutant": {"public_pass": states["mutant"]["public"]["passed"], "hidden_pass": states["mutant"]["hidden"]["returncode"] == 0, "source_sha256": states["mutant"]["hidden"]["candidate_source_sha256"]},
            "controls_pass": all(states[x]["controls_pass"] for x in ("baseline", "fixed", "mutant")),
            "grader_case_outputs_withheld": True,
            "grader_source_withheld": True,
            "known_fix_source_withheld": True,
        }
        (OUT / "qualification-controls-public-summary.json").write_text(json.dumps(controls_public, indent=2, sort_keys=True) + "\n")
        attempts = [public_attempt(arm) for arm in ARMS]
        (OUT / "campaign-summary-v3-public.json").write_text(json.dumps({
            "schema_version": 1, "protocol_id": "vare-sprint-ix-progress-aware-coding-agent-v3",
            "protocol_sha256": json.loads((ROOT / "protocol-v3.json").read_text())["protocol_sha256"],
            "status": "ALL_PREREGISTERED_ARMS_COMPLETED",
            "claim_boundary": "one consumed Development family; exploratory descriptive only",
            "attempts": attempts,
            "independent_correct_repairs_per_family": {"numerator": 0, "denominator": 1},
            "hidden_case_details_withheld": True,
        }, indent=2, sort_keys=True) + "\n")
        files = []
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                files.append({"path": p.relative_to(OUT).as_posix(), "sha256": sha(p), "size_bytes": p.stat().st_size})
        manifest = {"schema_version": 1, "files": files, "self_excluded": True}
        manifest_path = OUT / "SHA256SUMS.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_hash = sha(manifest_path)
        (OUT / "SHA256SUMS.sha256").write_text(f"{manifest_hash}  SHA256SUMS.json\n")
        print(json.dumps({"status": "EXPORTED", "output": str(OUT), "arms": len(ARMS)}))
        return
    if not OUT.is_dir():
        raise SystemExit("public export does not exist")
    forbidden = {"grade_hidden.py", "independent-grader.json", "qualification-controls-v2.json", "known-fix.py", "fixed"}
    bad = [p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file() and (p.name in forbidden or "/fixed/" in p.as_posix() or "/mutant/" in p.as_posix())]
    if bad:
        raise SystemExit(f"forbidden hidden artifacts in public export: {bad}")
    manifest_path = OUT / "SHA256SUMS.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        expected = {item["path"]: item["sha256"] for item in m["files"]}
        actual = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob("*") if p.is_file() and p not in (manifest_path, OUT / "SHA256SUMS.sha256")}
        if expected != actual:
            raise SystemExit("public manifest mismatch")
        sidecar = (OUT / "SHA256SUMS.sha256").read_text().split()[0]
        if sidecar != sha(manifest_path):
            raise SystemExit("public manifest sidecar mismatch")
        print(json.dumps({"status": "PASS", "file_count": len(actual), "manifest_sha256": sha(manifest_path)}))
    else:
        raise SystemExit("SHA256SUMS.json missing")

if __name__ == "__main__":
    main()
