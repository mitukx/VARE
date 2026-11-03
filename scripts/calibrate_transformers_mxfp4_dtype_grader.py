#!/usr/bin/env python3
"""Run the frozen CPU calibration and Seatbelt audit for the MXFP4 task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "benchmarks/mlsys_repo_eval_v1/transformers_mxfp4_dtype"
GRADE_PATH = TASK_ROOT / "evaluator/grade.py"
BASELINE_INPUT = TASK_ROOT / "calibration_inputs/mxfp4.py.baseline"
KNOWN_FIX_INPUT = TASK_ROOT / "calibration_inputs/mxfp4.py.known_fix"
FORGERY_INPUT = REPO_ROOT / "tests/fixtures/transformers_mxfp4_dtype/report_forgery.py"
CALIBRATION_SEED = 20261104


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_grader() -> Any:
    spec = importlib.util.spec_from_file_location("vare_transformers_mxfp4_grade", GRADE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the task grader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str, name: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"{name} patch anchor count changed")
    return source.replace(old, new, 1)


def _make_variants(baseline: bytes, known_fix: bytes) -> dict[str, bytes]:
    base_text = baseline.decode("utf-8")
    fixed_text = known_fix.decode("utf-8")
    baseline_block = """        # Here we are dequantizing the weights
        dequantized = dequantize_convertops(param_data[f"{proj}_blocks"], param_data[f"{proj}_scales"])
        return {full_layer_name: dequantized}"""
    alternate_block = """        # Convert first, then cast to the destination parameter dtype.
        module, tensor_name = get_module_from_name(model, full_layer_name)
        dtype = getattr(module, tensor_name).dtype
        dequantized = dequantize_convertops(param_data[f"{proj}_blocks"], param_data[f"{proj}_scales"])
        return {full_layer_name: dequantized.to(dtype)}"""
    alternate = _replace_once(base_text, baseline_block, alternate_block, "alternate-correct")
    wrong_float32 = _replace_once(
        fixed_text,
        "dtype = getattr(module, tensor_name).dtype",
        "dtype = torch.float32",
        "wrong-hardcoded-float32",
    )
    wrong_zero = _replace_once(
        fixed_text,
        "return torch.nn.Parameter(dequantized)",
        "return torch.nn.Parameter(torch.zeros_like(dequantized))",
        "wrong-zero-values",
    )
    return {
        "baseline": baseline,
        "known_fix": known_fix,
        "alternate_correct": alternate.encode(),
        "wrong_hardcoded_float32": wrong_float32.encode(),
        "wrong_zero_values": wrong_zero.encode(),
        "import_time_report_forgery": FORGERY_INPUT.read_bytes(),
    }


def _case_summary(result: dict[str, Any]) -> dict[str, int]:
    rows = result.get("case_results", [])
    return {
        "passed": sum(bool(row.get("passed")) for row in rows),
        "failed": sum(not bool(row.get("passed")) for row in rows),
    }


def _sandbox_probe_script(paths: dict[str, str]) -> str:
    encoded = json.dumps(paths, sort_keys=True)
    return f'''import json, os, socket, subprocess, sys
paths = json.loads({encoded!r})
results = {{}}
def read_probe(name, path):
    try:
        with open(path, "rb") as stream:
            stream.read(1)
        results[name] = {{"result": "ALLOW"}}
    except OSError as exc:
        results[name] = {{"result": "DENY", "errno": exc.errno, "error": type(exc).__name__}}
read_probe("candidate_source_read", __file__)
try:
    with open(os.path.join(os.environ["TMPDIR"], "allowed-write.txt"), "wb") as stream:
        stream.write(b"allowed")
    results["scratch_write"] = {{"result": "ALLOW"}}
except OSError as exc:
    results["scratch_write"] = {{"result": "DENY", "errno": exc.errno, "error": type(exc).__name__}}
for name, path in paths.items():
    if path:
        read_probe(name, path)
outside = "/tmp/vare-mxfp4-seatbelt-outside-%d.txt" % os.getpid()
try:
    with open(outside, "wb") as stream:
        stream.write(b"must-not-exist")
    results["outside_write"] = {{"result": "ALLOW", "path": outside}}
except OSError as exc:
    results["outside_write"] = {{"result": "DENY", "errno": exc.errno, "error": type(exc).__name__, "path": outside}}
try:
    socket.socket().connect(("127.0.0.1", 1))
    results["network_connect"] = {{"result": "ALLOW"}}
except OSError as exc:
    results["network_connect"] = {{"result": "DENY", "errno": exc.errno, "error": type(exc).__name__}}
try:
    subprocess.run(["/usr/bin/true"], check=True, timeout=2)
    results["unexpected_process"] = {{"result": "ALLOW"}}
except OSError as exc:
    results["unexpected_process"] = {{"result": "DENY", "errno": exc.errno, "error": type(exc).__name__}}
except subprocess.SubprocessError as exc:
    results["unexpected_process"] = {{"result": "DENY", "error": type(exc).__name__}}
print(json.dumps(results, sort_keys=True, separators=(",", ":")))
'''


def _run_sandbox_audit(
    grader: Any,
    evidence_root: Path,
    *,
    protected_paths: tuple[Path, ...],
    upstream_repository: Path | None,
    known_fix_source: Path,
) -> dict[str, Any]:
    audit_root = evidence_root / "sandbox-audit"
    candidate_root = audit_root / "candidate"
    scratch = audit_root / "scratch"
    candidate_root.mkdir(parents=True)
    (scratch / "home").mkdir(parents=True)
    (scratch / "tmp").mkdir()
    paths = {
        "task_protocol": str(TASK_ROOT / "protocol.json"),
        "hidden_grader_definition": str(GRADE_PATH),
        "known_fix_source": str(known_fix_source),
        "original_dirty_checkout": str(Path.home() / "VARE/AGENTS.md"),
        "candidate_git_head": str(upstream_repository / ".git/HEAD") if upstream_repository else "",
    }
    installed_transformers = Path(sysconfig.get_paths()["purelib"]) / "transformers"
    if installed_transformers.exists():
        paths["installed_transformers_source"] = str(
            installed_transformers / "integrations/mxfp4.py"
        )
    probe_path = candidate_root / "probe.py"
    probe_path.write_text(_sandbox_probe_script(paths), encoding="utf-8")
    probe_path.chmod(0o444)
    runtime_root = Path(sysconfig.get_paths()["purelib"]).resolve()
    forbidden = set(protected_paths)
    forbidden.update({TASK_ROOT, REPO_ROOT, Path.home() / "VARE"})
    if upstream_repository:
        forbidden.update({upstream_repository, upstream_repository / ".git"})
    if installed_transformers.exists():
        forbidden.add(installed_transformers)
    profile = grader.cpu_sandbox_profile(
        candidate_root=candidate_root,
        scratch=scratch,
        python=Path(sys.executable).resolve(),
        python_prefix=Path(sys.prefix).resolve(),
        python_base_prefix=Path(sys.base_prefix).resolve(),
        runtime_root=runtime_root,
        forbidden_paths=tuple(sorted(forbidden, key=str)),
    )
    profile_path = audit_root / "candidate.sb"
    profile_path.write_text(profile, encoding="utf-8")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(runtime_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "LANG": "C",
        "HOME": str(scratch / "home"),
        "TMPDIR": str(scratch / "tmp"),
    }
    started = time.monotonic()
    process = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile_path), str(Path(sys.executable).resolve()), "-S", str(probe_path)],
        cwd=candidate_root,
        env=environment,
        capture_output=True,
        timeout=20,
        check=False,
    )
    elapsed = round(time.monotonic() - started, 3)
    output = process.stdout
    (audit_root / "probe-output.log").write_bytes(output)
    if process.returncode != 0:
        return {
            "status": "INFRASTRUCTURE_FAIL",
            "returncode": process.returncode,
            "stderr": process.stderr.decode("utf-8", errors="replace")[-4000:],
            "elapsed_seconds": elapsed,
            "sandbox_profile_sha256": _sha256(profile.encode()),
        }
    try:
        results = json.loads(output.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        return {
            "status": "SECURITY_FAIL",
            "reason": f"sandbox probe output invalid:{type(exc).__name__}",
            "returncode": process.returncode,
            "elapsed_seconds": elapsed,
        }
    outside_path = results.get("outside_write", {}).get("path")
    outside_file_exists = bool(outside_path and Path(outside_path).exists())
    expected_allow = {"candidate_source_read", "scratch_write"}
    expected_deny = set(paths) | {"outside_write", "network_connect", "unexpected_process"}
    observed_allow = {key for key, value in results.items() if value.get("result") == "ALLOW"}
    observed_deny = {key for key, value in results.items() if value.get("result") == "DENY"}
    passed = (
        expected_allow <= observed_allow
        and expected_deny <= observed_deny
        and not outside_file_exists
    )
    return {
        "status": "PASS" if passed else "SECURITY_FAIL",
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "sandbox_profile_sha256": _sha256(profile.encode()),
        "probe_source_sha256": _sha256(probe_path.read_bytes()),
        "results": results,
        "unexpected_allow_results": sorted(observed_allow - expected_allow),
        "missing_denials": sorted(expected_deny - observed_deny),
        "outside_file_exists": outside_file_exists,
    }


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output_dir}")
    output_path = args.output_dir.resolve()
    if output_path == REPO_ROOT or REPO_ROOT in output_path.parents:
        raise ValueError("calibration evidence must be outside the protected VARE checkout")
    grader = _load_grader()
    protocol_info = grader._verify_protocol_lock()
    candidate_workspace = args.candidate_workspace
    if candidate_workspace.is_symlink() or not candidate_workspace.is_dir():
        raise ValueError("candidate workspace must be a real directory")
    if REPO_ROOT in candidate_workspace.resolve().parents:
        raise ValueError("candidate workspace must be outside the protected VARE checkout")
    workspace_head = grader._git(candidate_workspace, "rev-parse", "HEAD").decode().strip()
    workspace_shallow = grader._git(
        candidate_workspace, "rev-parse", "--is-shallow-repository"
    ).decode().strip()
    workspace_remotes = grader._git(candidate_workspace, "remote").decode().strip()
    workspace_changes = grader._changed_paths(candidate_workspace)
    if (
        workspace_head != grader.BASELINE_COMMIT
        or workspace_shallow != "true"
        or workspace_remotes
        or workspace_changes
    ):
        raise RuntimeError("candidate workspace is not a clean shallow baseline without remotes")
    try:
        grader._git(candidate_workspace, "cat-file", "-e", f"{grader.KNOWN_FIX_COMMIT}^{{commit}}")
        raise RuntimeError("candidate workspace contains the known-fix commit object")
    except subprocess.CalledProcessError:
        pass
    known_fix_blob = grader._git(candidate_workspace, "hash-object", str(KNOWN_FIX_INPUT)).decode().strip()
    try:
        grader._git(candidate_workspace, "cat-file", "-e", known_fix_blob)
        raise RuntimeError("candidate workspace contains the known-fix source blob")
    except subprocess.CalledProcessError:
        pass
    lock = json.loads((TASK_ROOT / "protocol.lock.json").read_text(encoding="utf-8"))
    baseline = BASELINE_INPUT.read_bytes()
    known_fix = KNOWN_FIX_INPUT.read_bytes()
    if _sha256(baseline) != lock["source"]["baseline_source_sha256"]:
        raise RuntimeError("baseline input differs from the frozen upstream source hash")
    if _sha256(known_fix) != lock["source"]["known_fix_source_sha256"]:
        raise RuntimeError("known-fix input differs from the frozen upstream source hash")
    variants = _make_variants(baseline, known_fix)
    contracts = {
        name: grader.source_contract_bytes(source)
        for name, source in variants.items()
    }
    if any(not row["accepted"] for name, row in contracts.items() if name != "import_time_report_forgery"):
        raise RuntimeError("a frozen calibration arm no longer matches the allowed AST forms")
    if contracts["import_time_report_forgery"]["accepted"]:
        raise RuntimeError("report-forgery fixture unexpectedly matches an allowed source form")

    args.output_dir.mkdir(parents=True)
    source_root = args.output_dir / "generated-sources"
    source_root.mkdir()
    arms_root = args.output_dir / "arms"
    arms_root.mkdir()
    workspaces_root = args.output_dir / "workspaces"
    workspaces_root.mkdir()
    raw: dict[str, Any] = {}
    expected = {
        "baseline": "TEST_FAIL",
        "known_fix": "PASS",
        "alternate_correct": "PASS",
        "wrong_hardcoded_float32": "TEST_FAIL",
        "wrong_zero_values": "TEST_FAIL",
        "import_time_report_forgery": "SECURITY_FAIL",
    }
    protected_paths = tuple(args.protected_path) + (
        REPO_ROOT,
        TASK_ROOT,
        Path.home() / "VARE",
        source_root,
    )
    for name, source_bytes in variants.items():
        source_path = source_root / f"{name}.py"
        source_path.write_bytes(source_bytes)
        workspace = workspaces_root / name
        shutil.copytree(candidate_workspace, workspace)
        if name != "baseline":
            (workspace / grader.TARGET_PATH).write_bytes(source_bytes)
        result = grader.grade_workspace(workspace, arms_root / name)
        raw[name] = {
            "expected_status": expected[name],
            "source_sha256": _sha256(source_bytes),
            "source_contract": contracts[name],
            "result": result,
            "case_summary": _case_summary(result),
        }

    outcomes_match = all(
        raw[name]["result"].get("grader_status") == expected[name]
        for name in expected
    )
    worker_arms_valid = all(
        len(raw[name]["result"].get("candidate_runs", [])) == 1
        and raw[name]["result"]["candidate_runs"][0].get("returncode") == 0
        and raw[name]["result"]["candidate_runs"][0].get("unexpected_process_count") == 0
        and not raw[name]["result"]["candidate_runs"][0].get("timed_out")
        and not raw[name]["result"]["candidate_runs"][0].get("rss_limit_exceeded")
        and not raw[name]["result"]["candidate_runs"][0].get("output_limit_exceeded")
        and not raw[name]["result"]["candidate_runs"][0].get("monitor_error")
        for name in (
            "baseline", "known_fix", "alternate_correct",
            "wrong_hardcoded_float32", "wrong_zero_values",
        )
    )
    expected_counts = {
        "baseline": {"passed": 2, "failed": 6},
        "known_fix": {"passed": 8, "failed": 0},
        "alternate_correct": {"passed": 8, "failed": 0},
        "wrong_hardcoded_float32": {"passed": 2, "failed": 6},
        "wrong_zero_values": {"passed": 0, "failed": 8},
    }
    case_counts_match = all(
        raw[name]["case_summary"] == expected_counts[name]
        for name in expected_counts
    )
    forged_worker_not_launched = not raw["import_time_report_forgery"]["result"].get("candidate_runs")
    sandbox = _run_sandbox_audit(
        grader,
        args.output_dir,
        protected_paths=protected_paths,
        upstream_repository=candidate_workspace,
        known_fix_source=KNOWN_FIX_INPUT,
    )
    runtime = {
        "python": sys.version.split()[0],
        "torch": grader.importlib.metadata.version("torch"),
        "os": platform.system(),
        "architecture": platform.machine(),
        "macos": platform.mac_ver()[0],
        "device": "CPU only; no Metal or CUDA",
    }
    passed = (
        outcomes_match
        and worker_arms_valid
        and case_counts_match
        and forged_worker_not_launched
        and sandbox["status"] == "PASS"
    )
    summary = {
        "schema_version": 1,
        "task_id": grader.TASK_ID,
        "protocol_lock_sha256": protocol_info["protocol_lock_sha256"],
        "frozen_file_hashes": protocol_info["files"],
        "calibration_seed": CALIBRATION_SEED,
        "runtime": runtime,
        "candidate_workspace": {
            "baseline_revision": workspace_head,
            "shallow": workspace_shallow,
            "remotes": workspace_remotes,
            "changed_paths_before_calibration": workspace_changes,
            "baseline_source_sha256": _sha256(
                (candidate_workspace / grader.TARGET_PATH).read_bytes()
            ),
            "known_fix_commit_present": False,
            "known_fix_source_blob_present": False,
            "candidate_visible_files": sorted(
                path.relative_to(candidate_workspace).as_posix()
                for path in candidate_workspace.rglob("*")
                if path.is_file() and ".git" not in path.parts
            ),
        },
        "arms": raw,
        "checks": {
            "outcomes_match": outcomes_match,
            "all_worker_processes_valid": worker_arms_valid,
            "case_counts_match": case_counts_match,
            "report_forgery_rejected_before_worker_launch": forged_worker_not_launched,
            "real_seatbelt_probe_passed": sandbox["status"] == "PASS",
        },
        "sandbox_audit": sandbox,
        "calibration_status": "PASS" if passed else "INDETERMINATE",
        "model_inference_performed": False,
        "optimizer_updates": 0,
    }
    (args.output_dir / "calibration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-workspace", type=Path, required=True)
    parser.add_argument("--protected-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = run_calibration(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["calibration_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
