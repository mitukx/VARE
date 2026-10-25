#!/usr/bin/env python3
"""Independently reconstruct the frozen MLX #4748 qualification result."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/run-v4"
PROTOCOL = ROOT / "protocol-v4.json"
LOCK = ROOT / "protocol-v4.lock.json"
REPO = Path("source/mlx.git").resolve()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shafile(path: Path) -> str:
    return sha(path.read_bytes())


def git(*args: str, binary=False):
    import subprocess
    p = subprocess.run(["git", "-C", str(REPO), *args], check=True,
                       capture_output=True, text=not binary)
    return p.stdout


def write_new(path: Path, obj) -> None:
    data=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    with path.open("xb") as f:
        f.write(data);f.flush()


def main() -> int:
    errors=[]; checks={}
    protocol=json.loads(PROTOCOL.read_text())
    lock=json.loads(LOCK.read_text())
    checks["protocol_hash_locked"] = shafile(PROTOCOL)==lock["protocol_sha256"]
    checks["runner_hash_locked"] = shafile(ROOT/"qualification_runner_v4.py")==lock["runner_sha256"]
    checks["lock_points_to_protocol"] = Path(lock["protocol_path"]).resolve()==PROTOCOL.resolve()
    checks["protocol_hash_matches_summary"] = json.loads((RUN/"summary.json").read_text()).get("protocol_sha256")==lock["protocol_sha256"]
    manifest=json.loads((RUN/"manifest.json").read_text())
    manifest_bad=[]
    listed=set()
    for row in manifest["files"]:
        p=RUN/row["path"];listed.add(row["path"])
        if not p.is_file() or p.stat().st_size!=row["size_bytes"] or shafile(p)!=row["sha256"]:
            manifest_bad.append(row["path"])
    actual={p.relative_to(RUN).as_posix() for p in RUN.rglob("*") if p.is_file() and p.name!="manifest.json"}
    checks["manifest_complete_and_valid"]=(not manifest_bad and listed==actual)
    if manifest_bad: errors.append("manifest entries differ: "+repr(manifest_bad))
    summary=json.loads((RUN/"summary.json").read_text())
    task=protocol["task"]
    phases={}
    per_phase={}
    for phase in task["phase_order"]:
        pdir=RUN/phase
        rec=json.loads((pdir/"phase.json").read_text())
        target=git("show",f"{task['baseline_commit'] if phase=='baseline' else task['fixed_commit']}:{task['target_path']}",binary=True)
        if phase=="mutant":
            needle=b"linalg::tri_inv(a, upper_, stream())"
            assert target.count(needle)==1
            target=target.replace(needle,b"linalg::tri_inv(a, !upper_, stream())",1)
        expected={"baseline":task["baseline_sha256"],"mutant":task["mutant_sha256"],"fixed":task["fixed_sha256"]}[phase]
        cases=rec["oracle"]["cases"]
        direct=[v["passed"] for k,v in cases.items() if k.endswith("_direct")]
        mapped=[v["passed"] for k,v in cases.items() if "vmap" in k]
        semantic={"baseline_direct_or_fix_direct":len(direct)==2 and all(direct),
                  "vmapped_controls_detect_bug":len(mapped)==4 and not all(mapped),
                  "fixed_all_six":len(cases)==6 and all(x["passed"] for x in cases.values())}
        test_exit=rec["upstream_test"]["exit_code"]
        control=phase in ("baseline","mutant")
        outcome=(semantic["baseline_direct_or_fix_direct"] and semantic["vmapped_controls_detect_bug"]
                 and test_exit!=0) if control else (semantic["baseline_direct_or_fix_direct"]
                 and semantic["fixed_all_six"] and test_exit==0)
        files_ok=(shafile(pdir/"build.stdout.log")==rec["build"]["stdout_sha256"]
                  and shafile(pdir/"oracle.stdout.log")==rec["oracle_runtime"]["stdout_sha256"]
                  and shafile(pdir/"upstream-test.stdout.log")==rec["upstream_test"]["stdout_sha256"])
        wrappers=[rec["build"],rec["oracle_runtime"],rec["upstream_test"]]
        sandboxed=all(x["argv"][0:2]==["/usr/bin/sandbox-exec","-p"]
                      and x["network_profile"]=="(version 1) (allow default) (deny network*)"
                      and x["abort_reason"] is None for x in wrappers)
        checks[phase+"_target_identity"] = sha(target)==expected==rec["target_sha256"]
        checks[phase+"_build_pass"] = rec["build"]["exit_code"]==0 and rec["build"]["abort_reason"] is None
        checks[phase+"_independent_control"] = outcome and rec["phase_accepted"] is True
        checks[phase+"_network_and_abort_metadata"] = sandboxed
        checks[phase+"_raw_log_hashes"] = files_ok
        samples=[]
        for label in ("build","oracle","upstream-test"):
            sf=pdir/f"{label}.resources.jsonl"
            rows=[json.loads(line) for line in sf.read_text().splitlines()]
            samples.extend(rows)
        limits=protocol["build_environment"]["resource_limits"]
        phase_resource_pass=bool(samples) and all(
            x["available_memory_bytes"]>=limits["minimum_available_memory_bytes"]
            and x["free_swap_bytes"]>=limits["minimum_free_swap_bytes"]
            and x["disk_free_bytes"]>=limits["minimum_disk_free_bytes"]
            for x in samples)
        checks[phase+"_sampled_resource_floors"] = phase_resource_pass
        per_phase[phase]={"phase_status":rec["status"],"source_sha256":rec["target_sha256"],
            "independent_cases":len(cases),"direct_passes":sum(direct),"vmap_passes":sum(mapped),
            "upstream_test_exit_code":test_exit,
            "build_seconds":rec["build"]["elapsed_seconds"],
            "build_peak_process_tree_rss_bytes":rec["build"]["peak_process_tree_rss_bytes"],
            "oracle_semantics":semantic,"resource_samples":len(samples),
            "minimum_available_memory_bytes":min(x["available_memory_bytes"] for x in samples),
            "minimum_free_swap_bytes":min(x["free_swap_bytes"] for x in samples),
            "maximum_swap_used_bytes":max(x["swap_used_bytes"] for x in samples)}
        phases[phase]=rec
    checks["summary_reconstructed"]=(summary["status"]=="PASS_TASK_CERTIFICATION_PILOT"
       and set(phases)=={"baseline","mutant","fixed"}
       and all(phases[p]["phase_accepted"] for p in phases)
       and summary["model_inference_calls"]==0 and summary["training_updates"]==0)
    checks["grader_inputs_pinned"] = all(
       shafile(RUN/"grader"/spec["path"])==spec["sha256"] for spec in protocol["grader_files"])
    starts={}
    for phase in task["phase_order"]:
       rows=[json.loads(x) for x in (RUN/phase/"build.resources.jsonl").read_text().splitlines()]
       starts[phase]=rows[0]["timestamp_utc"]
    observed_order=sorted(starts,key=lambda phase:starts[phase])
    checks["phase_order_pinned"] = observed_order==task["phase_order"]
    errors.extend(k for k,v in checks.items() if not v)
    all_rows=[json.loads(line) for p in RUN.rglob("*.resources.jsonl") for line in p.read_text().splitlines()]
    pstart=json.loads((RUN/"preflight.json").read_text())
    mins={"available_memory_bytes":min(x["available_memory_bytes"] for x in all_rows),
          "free_swap_bytes":min(x["free_swap_bytes"] for x in all_rows),
          "disk_free_bytes":min(x["disk_free_bytes"] for x in all_rows)}
    overall={"minimum_available_memory_bytes":mins["available_memory_bytes"],
             "minimum_free_swap_bytes":mins["free_swap_bytes"],
             "minimum_disk_free_bytes":mins["disk_free_bytes"],
             "maximum_process_tree_rss_bytes":max(x["process_tree_rss_bytes"] for x in all_rows),
             "preflight_swap_used_bytes":pstart["swap_used_bytes"],
             "maximum_sampled_swap_used_bytes":max(x["swap_used_bytes"] for x in all_rows),
             "max_sampled_swap_growth_vs_global_preflight_bytes":max(x["swap_used_bytes"] for x in all_rows)-pstart["swap_used_bytes"],
             "cgroup_memory":"unavailable on macOS; not claimed"}
    report={"audit":"INDEPENDENT_RECONSTRUCTION_MLX_PR4748_V4",
            "phase_order_observed":observed_order,"phase_first_sample_times_utc":starts,
            "run_manifest_sha256":shafile(RUN/"manifest.json"),
            "protocol_sha256":lock["protocol_sha256"],"runner_sha256":lock["runner_sha256"],
            "checks":checks,"phase_reconstruction":per_phase,"resources":overall,
            "status":"PASS_TASK_CERTIFICATION_PILOT" if not errors else "AUDIT_NO_GO",
            "failed_checks":errors,
            "claim_limit":"One source/grader-certified Development family; no model inference, held-out eligibility, learning, or capability claim."}
    out=RUN.parent/"run-v4-independent-audit-v2.json"
    write_new(out,report)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not errors else 2

if __name__=="__main__":raise SystemExit(main())
