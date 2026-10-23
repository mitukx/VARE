#!/usr/bin/env python3
"""Run the bounded, retrospective Qwen2.5-3B #1910 diagnostic protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_REL = "benchmarks/mlsys_repo_eval_v3/mlxlm_1910_agent_task"
BASE = "bacd9e8e1f3a5be7dd64bae3a9ad8d74471ccc3a"
FIXED = "a537041adc86d73cfae4cc1375b2c8851c652c60"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observe() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import monitor_local_agent_resources_v1 as monitor

    return monitor.observe()


def git_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/tmp",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], env=git_env(),
        capture_output=True, text=True, check=False, timeout=30,
    )
    if check and result.returncode:
        raise ValueError(f"git {args[0]} failed: {result.stderr[-1000:]}")
    return result


def is_ancestor(repo: Path, commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
        env=git_env(), capture_output=True, check=False, timeout=30,
    ).returncode == 0


def verify_git_store(repo: Path) -> None:
    if git(repo, "for-each-ref", "--format=%(refname)", "refs/replace").stdout.strip():
        raise ValueError("Git replacement refs are present")
    alternates = repo / ".git/objects/info/alternates"
    if alternates.exists() or alternates.is_symlink():
        raise ValueError("Git object alternates are not permitted")
    if any(path.is_symlink() for path in repo.rglob("*")):
        raise ValueError("symlinks are not permitted in task or upstream Git source trees")


def verify_historical_manifest(record: dict[str, Any]) -> None:
    relative = record["path"]
    manifest_path = ROOT / relative
    if manifest_path.is_symlink() or not manifest_path.is_file() or sha256(manifest_path) != record["sha256"]:
        raise ValueError(f"historical evidence manifest identity mismatch: {relative}")
    sidecar_path = ROOT / record["sidecar_path"]
    if (sidecar_path.is_symlink() or not sidecar_path.is_file()
            or sha256(sidecar_path) != record["sidecar_sha256"]
            or sidecar_path.read_text(encoding="ascii").split()[0] != record["sha256"]):
        raise ValueError(f"historical evidence manifest sidecar mismatch: {record['sidecar_path']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if isinstance(files, list):
        entries = {item["path"]: item["sha256"] for item in files}
        base = ROOT
    elif isinstance(files, dict):
        entries = files
        base = manifest_path.parent
    else:
        raise ValueError(f"unsupported historical evidence manifest: {relative}")
    overrides = record.get("historical_file_overrides", {})
    for name, expected in entries.items():
        item = base / name
        actual = not item.is_symlink() and item.is_file() and sha256(item) == expected
        if actual:
            continue
        override = overrides.get(name)
        if not override or override.get("sha256") != expected:
            raise ValueError(f"raw historical evidence does not match {relative}: {name}")
        committed = git(ROOT, "show", f"{override['commit']}:{name}", check=False)
        if (committed.returncode or hashlib.sha256(committed.stdout.encode()).hexdigest() != expected
                or not is_ancestor(ROOT, override["commit"])):
            raise ValueError(f"historical source override cannot be verified: {name}")


def verify(protocol_path: Path, model_path: Path, python_exe: Path,
           git_source: Path, *, check_resources: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if protocol_path.is_symlink():
        raise ValueError("protocol path must not be a symlink")
    protocol_path = protocol_path.resolve(strict=True)
    lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != "mlsys-repo-task-mlx-lm-1910-qwen25-3b-parser-debug-v1":
        raise ValueError("unexpected protocol identity")
    if lock.get("status") != "FROZEN_NOT_RUN":
        raise ValueError("protocol is not frozen and unrun")
    sidecar = protocol_path.with_name("protocol.sha256")
    if sidecar.is_symlink() or sidecar.read_text(encoding="ascii").split()[0] != sha256(protocol_path):
        raise ValueError("protocol SHA-256 sidecar mismatch")

    expected_committed = lock["implementation"]["code_commit"]
    if not is_ancestor(ROOT, expected_committed):
        raise ValueError("locked implementation commit is not an ancestor of this checkout")
    for relative, expected in lock["code_sha256"].items():
        source = ROOT / relative
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"locked source identity mismatch: {relative}")
    for relative, expected in lock["task"]["bundle_sha256"].items():
        source = ROOT / relative
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"locked task identity mismatch: {relative}")
    for relative, expected in lock["provenance"]["files_sha256"].items():
        source = ROOT / relative
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"locked evidence identity mismatch: {relative}")
    for manifest_record in lock["provenance"]["historical_manifests"]:
        verify_historical_manifest(manifest_record)
    prompt_spec_path = ROOT / lock["agent"]["prompt_spec_path"]
    if prompt_spec_path.is_symlink() or sha256(prompt_spec_path) != lock["agent"]["prompt_spec_sha256"]:
        raise ValueError("frozen prompt and tool-schema artifact identity mismatch")
    prompt_spec = json.loads(prompt_spec_path.read_text(encoding="utf-8"))

    # Every locked repository file must be in the committed tree as frozen.
    for relative, expected in {**lock["code_sha256"], **lock["task"]["bundle_sha256"],
                               **lock["provenance"]["files_sha256"]}.items():
        status = git(ROOT, "status", "--porcelain", "--", relative).stdout
        if status:
            raise ValueError(f"locked file has working-tree changes: {relative}")
        committed = git(ROOT, "show", f"HEAD:{relative}", check=False)
        if committed.returncode or hashlib.sha256(committed.stdout.encode()).hexdigest() != expected:
            raise ValueError(f"locked file differs from committed HEAD: {relative}")
    protocol_relative = protocol_path.relative_to(ROOT).as_posix()
    committed_protocol = git(ROOT, "show", f"HEAD:{protocol_relative}", check=False)
    if committed_protocol.returncode or hashlib.sha256(committed_protocol.stdout.encode()).hexdigest() != sha256(protocol_path):
        raise ValueError("frozen protocol is not committed at this revision")

    if model_path.is_symlink():
        raise ValueError("model snapshot path must not be a symlink")
    model_path = model_path.resolve(strict=True)
    if model_path.name != lock["model"]["revision"]:
        raise ValueError("model snapshot directory does not match the pinned revision")
    model_manifest_path = ROOT / lock["model"]["manifest_path"]
    if model_manifest_path.is_symlink() or sha256(model_manifest_path) != lock["model"]["manifest_sha256"]:
        raise ValueError("model manifest SHA-256 mismatch")
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != lock["model"]["revision"] or manifest.get("repository") != lock["model"]["repository"]:
        raise ValueError("model repository or revision does not match manifest")
    for name, identity in manifest["files"].items():
        item = model_path / name
        if (item.is_symlink() or not item.is_file() or item.stat().st_size != identity["bytes"]
                or sha256(item) != identity["sha256"]):
            raise ValueError(f"model checkpoint file identity mismatch: {name}")

    python_exe = python_exe.absolute()
    # Keep venv/bin/python as supplied; resolving its launcher symlink can jump
    # into the base interpreter and lose the pinned environment's site-packages.
    if not python_exe.is_file():
        raise ValueError("pinned Python executable is missing")
    probe = subprocess.run(
        [str(python_exe), "-c",
         "import importlib.metadata as m,json,platform,sys; print(json.dumps({'python':platform.python_version(),'machine':platform.machine(),'packages':{k:m.version(k) for k in ('mlx-lm','mlx','transformers','numpy','safetensors','tokenizers')}}))"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if probe.returncode:
        raise ValueError("could not inspect the pinned MLX runtime")
    runtime_actual = json.loads(probe.stdout)
    runtime_expected = lock["runtime"]
    if (runtime_actual["python"] != runtime_expected["python_version"]
            or runtime_actual["machine"] != runtime_expected["machine"]
            or runtime_actual["packages"] != runtime_expected["packages"]):
        raise ValueError("runtime package versions or architecture differ from the frozen inventory")
    tool_probe = subprocess.run(
        [str(python_exe), "-c",
         "import json,sys; sys.path.insert(0,sys.argv[1]); from run_local_agent import tools_for; print(json.dumps(tools_for(sys.argv[2],public_tests=True),ensure_ascii=False,sort_keys=True,separators=(',',':'))) ",
         str(ROOT / "scripts"), lock["agent"]["edit_format"]],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if tool_probe.returncode or hashlib.sha256(tool_probe.stdout.strip().encode()).hexdigest() != prompt_spec["tool_schema_sha256"]:
        raise ValueError("runtime tool schema differs from the frozen prompt artifact")
    freeze_path = ROOT / runtime_expected["inventory_path"]
    if freeze_path.is_symlink() or sha256(freeze_path) != runtime_expected["inventory_sha256"]:
        raise ValueError("runtime inventory identity mismatch")

    if git_source.is_symlink():
        raise ValueError("upstream source path must not be a symlink")
    git_source = git_source.resolve(strict=True)
    if not git_source.is_dir():
        raise ValueError("pinned upstream source is not a real directory")
    verify_git_store(git_source)
    if git(git_source, "rev-parse", BASE).stdout.strip() != BASE:
        raise ValueError("upstream baseline commit identity mismatch")
    if git(git_source, "rev-parse", FIXED).stdout.strip() != FIXED:
        raise ValueError("upstream known-fix commit identity mismatch")
    if git(git_source, "for-each-ref", "--format=%(refname)", "refs/replace").stdout.strip():
        raise ValueError("upstream source contains replacement refs")
    for commit, target_hash in ((BASE, lock["task"]["base_source_sha256"]),
                                (FIXED, lock["task"]["fixed_source_sha256"])):
        target = git(git_source, "show", f"{commit}:mlx_lm/tool_parsers/qwen3_coder.py").stdout.encode()
        if hashlib.sha256(target).hexdigest() != target_hash:
            raise ValueError(f"upstream source bytes differ at {commit}")
    task_workspace = ROOT / TASK_REL / "workspace"
    verify_git_store(task_workspace)
    git_config = task_workspace / ".git/config"
    if git_config.is_symlink() or sha256(git_config) != lock["task"]["workspace_git_config_sha256"]:
        raise ValueError("candidate workspace Git configuration identity mismatch")
    hooks = task_workspace / ".git/hooks"
    if any(path.name != "README.sample" and not path.name.endswith(".sample") for path in hooks.iterdir()):
        raise ValueError("candidate workspace contains a non-sample Git hook")
    if git(task_workspace, "rev-parse", "HEAD").stdout.strip() != lock["task"]["vare_workspace_base_commit"]:
        raise ValueError("candidate task workspace baseline mismatch")
    if git(task_workspace, "status", "--porcelain").stdout:
        raise ValueError("candidate task workspace must be clean before the run")
    known_fix_probe = git(task_workspace, "cat-file", "-e", f"{FIXED}^{{commit}}", check=False)
    if known_fix_probe.returncode == 0:
        raise ValueError("known-fix commit is present in candidate workspace object database")

    grader = json.loads((ROOT / lock["grader"]["calibration_path"]).read_text(encoding="utf-8"))
    if grader.get("status") != "CALIBRATION_PASS_DEVELOPMENT_ONLY":
        raise ValueError("candidate grader calibration is not passing")
    expected = {"baseline": "FAIL", "known_fix": "PASS",
                "mutant_restore_default_string": "FAIL",
                "mutant_parse_any_json_scalar": "FAIL", "mutant_object_only": "FAIL"}
    if {item["arm"]: item["grader_status"] for item in grader["records"]} != expected:
        raise ValueError("candidate grader calibration pattern differs from the locked baseline/fix/mutant result")
    image = subprocess.run(
        ["docker", "--context", "colima", "image", "inspect", lock["runtime"]["public_test_image"], "--format", "{{.Id}}"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if image.returncode or image.stdout.strip() != lock["runtime"]["public_test_image"]:
        raise ValueError("pinned public-test image is unavailable or has a different identity")

    resources = observe()
    resources["disk_free_bytes"] = shutil.disk_usage(ROOT).free
    budget = lock["resources"]
    if check_resources:
        if (resources["swap_used_mib"] > budget["max_start_swap_mib"]
                or resources["memory_free_percent"] < budget["min_start_free_memory_percent"]
                or resources["disk_free_bytes"] < budget["min_disk_free_bytes"]):
            raise ResourceError("preflight resource gate failed", resources)
    return lock, {"runtime": runtime_actual, "resources": resources,
                  "model_manifest_sha256": sha256(model_manifest_path),
                  "git_source_head": git(git_source, "rev-parse", "HEAD").stdout.strip(),
                  "repository_head": git(ROOT, "rev-parse", "HEAD").stdout.strip()}


class ResourceError(ValueError):
    def __init__(self, message: str, observation: dict[str, Any]):
        super().__init__(message)
        self.observation = observation


def validate_output_path(path: Path, expected: Path) -> Path:
    output = Path(os.path.abspath(path))
    expected = Path(os.path.abspath(expected))
    if output != expected:
        raise ValueError("output directory differs from the frozen prospective destination")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite experiment evidence: {output}")
    parent = output.parent
    if parent.resolve(strict=True) != parent or not parent.is_dir():
        raise ValueError("evidence destination parent is missing or symlinked")
    return output


def write_evidence_manifest(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in evidence output: {path}")
        if path.is_file() and path.name not in {"evidence-manifest.json", "evidence-manifest.sha256"}:
            entries.append({"path": path.relative_to(output).as_posix(),
                            "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest_path = output / "evidence-manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "files": entries},
                                        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "evidence-manifest.sha256").write_text(sha256(manifest_path) + "  evidence-manifest.json\n",
                                                    encoding="ascii")


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ROOT = args.repo_root.resolve(strict=True)
    try:
        lock, preflight = verify(args.protocol, args.model_path, args.python,
                                 args.git_source, check_resources=True)
    except ResourceError as exc:
        lock = json.loads(args.protocol.read_text(encoding="utf-8"))
        if args.check_only:
            print(json.dumps({"status": "BLOCKED_RESOURCE", "checks_passed": False,
                              "observation": exc.observation}))
            return 3
        try:
            output = validate_output_path(args.output, ROOT / lock["resources"]["output_relative"])
        except (ValueError, FileExistsError) as output_error:
            parser.error(str(output_error))
        os.mkdir(output, 0o700)
        (output / "preflight.json").write_text(json.dumps({"status": "BLOCKED_RESOURCE",
                    "reason": str(exc), "observation": exc.observation}, indent=2, sort_keys=True) + "\n")
        (output / "summary.json").write_text(json.dumps({"protocol_id": lock["protocol_id"],
                    "status": "BLOCKED", "model_loaded": False, "attempts": [],
                    "reason": str(exc)}, indent=2, sort_keys=True) + "\n")
        write_evidence_manifest(output)
        return 3
    except Exception as exc:
        parser.error(f"preflight failed closed: {type(exc).__name__}: {exc}")
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "checks_passed": True,
                          "protocol_id": lock["protocol_id"], **preflight}, sort_keys=True))
        return 0
    prompt_spec = json.loads((ROOT / lock["agent"]["prompt_spec_path"]).read_text(encoding="utf-8"))

    try:
        output = validate_output_path(args.output, ROOT / lock["resources"]["output_relative"])
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    os.mkdir(output, 0o700)
    (output / "protocol.lock.json").write_bytes(args.protocol.read_bytes())
    (output / "protocol.sha256").write_text(sha256(args.protocol) + "  protocol.lock.json\n", encoding="ascii")
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    arms: list[dict[str, Any]] = []
    abort_all = False
    task_root = ROOT / TASK_REL
    for arm in lock["arms"]:
        if abort_all:
            arms.append({"arm": arm["arm"], "status": "NOT_RUN", "reason": "earlier resource or integrity failure"})
            continue
        attempt = output / arm["arm"]
        os.mkdir(attempt, 0o700)
        workspace = attempt / "workspace"
        shutil.copytree(task_root / "workspace", workspace, symlinks=False)
        arm_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        resources_before = observe()
        remaining = lock["resources"]["max_total_agent_wall_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            arms.append({"arm": arm["arm"], "status": "NOT_RUN", "reason": "total agent wall budget exhausted"})
            abort_all = True
            continue
        wall = min(lock["agent"]["wall_seconds_per_attempt"], int(remaining))
        swap_limit = min(lock["resources"]["max_absolute_swap_mib"],
                         resources_before["swap_used_mib"] + lock["resources"]["max_swap_growth_mib"])
        command = [str(args.python.absolute()), str(ROOT / "scripts/run_local_agent.py"),
                   "--workspace", str(workspace), "--model-path", str(args.model_path.resolve(strict=True)),
                   "--model-repository", lock["model"]["repository"], "--backend", "mlx",
                   "--public-test-image", lock["runtime"]["public_test_image"],
                   "--task-root", str(task_root), "--output", str(attempt / "agent.json"),
                   "--instruction", lock["agent"]["instruction"], "--seed", str(arm["seed"]),
                   "--max-turns", str(lock["agent"]["turn_budget"]),
                   "--max-new-tokens-per-turn", str(lock["agent"]["max_new_tokens_per_turn"]),
                   "--max-total-new-tokens", str(lock["agent"]["max_total_new_tokens"]),
                   "--max-context-tokens", str(lock["agent"]["max_context_tokens"]),
                   "--wall-seconds", str(wall), "--temperature", str(arm["temperature"]),
                   "--top-p", str(arm["top_p"]), "--top-k", str(arm["top_k"]),
                   "--edit-format", lock["agent"]["edit_format"],
                   "--do-sample" if arm["do_sample"] else "--no-do-sample"]
        monitor = [str(args.python.absolute()), str(ROOT / "scripts/monitor_local_agent_resources_v2.py"),
                   "--output", str(attempt / "process"), "--max-seconds", str(wall),
                   "--swap-abort-mib", str(swap_limit),
                   "--memory-free-abort-percent", str(lock["resources"]["abort_free_memory_percent"]),
                   "--rss-abort-mib", str(lock["resources"]["max_agent_rss_mib"]),
                   "--poll-interval-seconds", str(lock["resources"]["monitor_interval_seconds"]),
                   "--", *command]
        monitor_env = {**os.environ,
                       "VARE_AGENT_REDACT_ROOT": str(ROOT),
                       "VARE_AGENT_REDACT_MODEL": str(args.model_path.resolve(strict=True)),
                       "VARE_AGENT_REDACT_PYTHON": str(args.python.absolute())}
        proc = subprocess.run(monitor, cwd=ROOT, capture_output=True, text=True,
                              check=False, env=monitor_env)
        (attempt / "monitor-launcher.stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (attempt / "monitor-launcher.stderr.txt").write_text(proc.stderr, encoding="utf-8")
        monitor_path = attempt / "process/resource-monitor.json"
        if not monitor_path.is_file():
            entry = {"arm": arm["arm"], "status": "FAILED", "reason": "resource monitor emitted no result",
                     "monitor_launcher_returncode": proc.returncode}
            arms.append(entry)
            abort_all = True
            continue
        monitor_result = json.loads(monitor_path.read_text(encoding="utf-8"))
        resources_after = observe()
        source = workspace / "qwen3_coder.py"
        public = workspace / "test_public.py"
        source_hash = sha256(source) if source.is_file() and not source.is_symlink() else None
        public_hash = sha256(public) if public.is_file() and not public.is_symlink() else None
        expected_public = lock["task"]["bundle_sha256"][f"{TASK_REL}/workspace/test_public.py"]
        grade_record = None
        grade_error = None
        agent_path = attempt / "agent.json"
        agent_record = json.loads(agent_path.read_text(encoding="utf-8")) if agent_path.is_file() else None
        if public_hash != expected_public or source_hash is None:
            grade_error = "candidate workspace integrity check failed"
            abort_all = True
        elif agent_record is None or agent_record.get("initial_messages") != prompt_spec["messages"]:
            grade_error = "agent initial messages differ from the frozen prompt"
            abort_all = True
        else:
            grade_output = attempt / "independent-grade"
            grade_command = [str(args.python.absolute()), str(ROOT / "scripts/grade_mlx_lm_1910_candidate_v1.py"),
                             "--git-source", str(args.git_source.resolve(strict=True)),
                             "--candidate-source", str(source), "--output", str(grade_output)]
            graded = subprocess.run(grade_command, cwd=ROOT, capture_output=True, text=True,
                                    check=False, env={**os.environ,
                                        "GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                                        "GIT_CONFIG_GLOBAL": os.devnull})
            (attempt / "grader-launcher.stdout.txt").write_text(graded.stdout, encoding="utf-8")
            (attempt / "grader-launcher.stderr.txt").write_text(graded.stderr, encoding="utf-8")
            if (grade_output / "grade.json").is_file():
                grade_record = json.loads((grade_output / "grade.json").read_text(encoding="utf-8"))
            else:
                grade_error = f"grader emitted no grade.json (return code {graded.returncode})"
                abort_all = True
        entry = {"arm": arm["arm"], "status": monitor_result["status"], "seed": arm["seed"],
                 "started_at_utc": arm_started, "monitor": monitor_result,
                 "resource_before": resources_before, "resource_after": resources_after,
                 "candidate_source_sha256": source_hash, "public_test_sha256": public_hash,
                 "grader_error": grade_error,
                 "agent_metrics": None if agent_record is None else {key: agent_record.get(key) for key in
                    ("stop_reason", "elapsed_seconds", "new_tokens", "tool_turns", "edit_attempted",
                     "edit_applied", "format_parse_failure_count", "public_test_result", "max_rss_bytes",
                     "runtime", "candidate_diff_sha256", "transcript")},
                 "independent_grade_status": None if grade_record is None else grade_record["status"]}
        (attempt / "attempt-summary.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n",
                                                        encoding="utf-8")
        arms.append(entry)
        if monitor_result["status"] == "ABORTED_RESOURCE":
            abort_all = True

    attempted = [arm for arm in arms if arm.get("status") != "NOT_RUN"]
    accepted = sum(arm.get("independent_grade_status") == "PASS" for arm in attempted)
    resource_abort = any(arm.get("status") == "ABORTED_RESOURCE" for arm in attempted)
    all_completed = len(attempted) == len(lock["arms"]) and all(
        arm.get("status") == "COMPLETED" and arm.get("independent_grade_status") in {"PASS", "FAIL"}
        for arm in attempted)
    status = "COMPLETED" if all_completed else "ABORTED_RESOURCE" if resource_abort else "FAILED"
    summary = {"schema_version": 1, "protocol_id": lock["protocol_id"], "status": status,
               "arms": arms, "attempt_successes": accepted, "attempt_denominator": len(attempted),
               "unique_development_task_successes": 1 if accepted else 0,
               "unique_development_task_denominator": 1, "binary_reward_mean": accepted / len(attempted) if attempted else None,
               "binary_reward_population_variance": (accepted / len(attempted)) * (1 - accepted / len(attempted)) if attempted else None,
               "model_loaded_attempts": sum(arm.get("agent_metrics") is not None for arm in attempted),
               "optimizer_steps": 0, "heldout_tasks_accessed": False,
               "task_exposure": "DEVELOPMENT_RETROSPECTIVE_DEBUG_ONLY",
               "claim_limit": "Two attempts on one previously exposed Development task; not fresh-task, held-out, or learning evidence.",
               "elapsed_seconds": round(time.monotonic() - started, 3)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "host-runtime.json").write_text(json.dumps({"python": sys.version,
        "platform": platform.platform(), "machine": platform.machine(), "repository_head": preflight["repository_head"],
        "preflight_resources": preflight["resources"], "post_run_resources": observe()},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_evidence_manifest(output)
    print(json.dumps({key: summary[key] for key in ("status", "attempt_successes", "attempt_denominator",
          "unique_development_task_successes", "unique_development_task_denominator", "elapsed_seconds")}))
    return 0 if status == "COMPLETED" else 3 if resource_abort else 2


if __name__ == "__main__":
    raise SystemExit(main())
