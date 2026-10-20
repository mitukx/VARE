#!/usr/bin/env python3
"""Run the frozen separated coding evaluation.

The E0 harness grades a synthetic JSON fixture with TrustedJSONScorer.
Train-split tasks are scored on the host with ``vare.isolated_candidate``.
A child exit status is not a pass. A task without
``evaluator/isolated_calls.json`` fails closed. The Mac command is
``local_measurement.command`` in the protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time

from vare.agent_edit_format import (
    classify_outcome, inference_arms, pass_at_k, per_task_reward_variance,
)
from vare.agent_task_contract import assert_train_only, load_agent_task
from vare.container_isolation import DockerIsolationRunner, IsolationError, IsolationLimits
from vare.environments.spec import EngineeringTaskSpec
from vare.heldout_exact_integer import wilson_interval
from vare.separated_grading import (
    SeparatedCandidateGrader, grade_isolated_subject, harness_required_record,
    source_only_snapshot,
)
from vare.trusted_scoring import JSONIOCase


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_docker_isolation_boundary_v1 import _grader_probe  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mem_available_from_meminfo(text: str) -> int:
    """Linux MemAvailable, in bytes. The gate thresholds stay in GiB."""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def mem_available_from_vm_stat(text: str) -> int:
    """macOS available bytes from ``vm_stat``.

    ``Pages free`` already includes speculative pages. Available memory is
    free plus inactive, the same split used by the host memory statistic.
    Page size comes from the ``vm_stat`` header.
    """
    match = re.search(r"page size of (\d+) bytes", text)
    if match is None:
        raise RuntimeError("vm_stat did not report a page size")
    page_size = int(match.group(1))
    pages: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        token = raw.strip().rstrip(".").replace(",", "")
        if token.isdigit():
            pages[key.strip()] = int(token)
    missing = [name for name in ("Pages free", "Pages inactive") if name not in pages]
    if missing:
        raise RuntimeError("vm_stat is missing " + ", ".join(missing))
    return (pages["Pages free"] + pages["Pages inactive"]) * page_size


def mem_available_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        return mem_available_from_meminfo(meminfo.read_text(encoding="utf-8"))
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["vm_stat"], check=False, text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("vm_stat failed")
        return mem_available_from_vm_stat(completed.stdout)
    raise RuntimeError("available memory is not readable on this platform")


def host_resource_snapshot() -> dict:
    """Capture host memory, swap, and disk before model load and between runs."""
    available = mem_available_bytes()
    swap_text = subprocess.run(
        ["sysctl", "vm.swapusage"], check=True, text=True, capture_output=True,
    ).stdout
    swap_match = re.search(r"free = ([0-9.]+)([KMG])", swap_text)
    if swap_match is None:
        raise RuntimeError("could not parse free swap from sysctl vm.swapusage")
    scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[swap_match.group(2)]
    disk_free = shutil.disk_usage(ROOT).free
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "available_memory_bytes": available,
        "swap_free_bytes": int(float(swap_match.group(1)) * scale),
        "disk_free_bytes": disk_free,
    }


def resource_preflight(protocol: dict) -> tuple[dict, list[str]]:
    observed = host_resource_snapshot()
    limits = protocol["budgets"]["resource_preflight"]
    failures = []
    for key, limit_key in (
        ("available_memory_bytes", "minimum_available_memory_bytes"),
        ("swap_free_bytes", "minimum_free_swap_bytes"),
        ("disk_free_bytes", "minimum_disk_free_bytes"),
    ):
        if observed[key] < limits[limit_key]:
            failures.append(f"{key} below {limit_key}")
    return observed, failures


def container_environment_error(returncodes: list[int], stderr: str = "") -> str | None:
    """Distinguish a Docker mount failure from an E0 expectation mismatch.

    Colima bind-mounts ``$HOME`` by default. ``docker run`` exits 125 when the
    workspace is outside that share, including under ``/tmp``.
    """
    markers = (
        "mounts denied",
        "error while creating mount",
        "forbidden path outside",
    )
    lowered = stderr.lower()
    if any(code == 125 for code in returncodes) or any(marker in lowered for marker in markers):
        return (
            "container environment error: Docker could not start the candidate container. "
            "Colima bind-mounts $HOME by default; a checkout or output path under /tmp exits 125. "
            "This is not an E0 expectation mismatch."
        )
    return None


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in protocol["public_files"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"public task file hash mismatch: {relative}")
    return protocol


def cases_from(protocol: dict) -> tuple[JSONIOCase, ...]:
    return tuple(
        JSONIOCase(item["case_id"], item["request"], item["expected"])
        for item in protocol["hidden_cases"]
    )


def scoring_payload(report) -> dict:
    return {
        "passed": report.passed,
        "score": report.score,
        "evaluation_complete": report.evaluation_complete,
        "cleanup_confirmed": report.cleanup_confirmed,
        "case_set_sha256": report.case_set_sha256,
        "total_wall_s": report.total_wall_s,
        "cases": [
            {
                "case_id": case.case_id,
                "passed": case.passed,
                "failure": case.failure,
                "returncode": case.returncode,
                "response_sha256": case.response_sha256,
                "wall_s": case.wall_s,
            }
            for case in report.cases
        ],
    }


def limits_for(protocol: dict) -> IsolationLimits:
    return IsolationLimits(
        protocol["execution"]["container_image"],
        timeout_s=20, memory_mb=256, cpus=0.5, pids_limit=32, max_output_bytes=16_000,
    )


def prepare_git_workspace(workspace: Path, solve_bytes: bytes) -> str:
    workspace.mkdir(parents=True)
    (workspace / "solve.py").write_bytes(solve_bytes)
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "solve.py"], cwd=workspace, check=True, capture_output=True)
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "vare",
        "GIT_AUTHOR_EMAIL": "vare@local",
        "GIT_COMMITTER_NAME": "vare",
        "GIT_COMMITTER_EMAIL": "vare@local",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    })
    subprocess.run(
        ["git", "commit", "-m", "base"], cwd=workspace, check=True,
        capture_output=True, env=env,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True, capture_output=True,
    ).stdout.strip()
    return revision


def write_task_root(task_root: Path, revision: str) -> None:
    source = ROOT / "benchmarks/separated_json_square_v1"
    task_root.mkdir(parents=True)
    shutil.copy(source / "TASK.md", task_root / "TASK.md")
    task = json.loads((source / "task.json").read_text(encoding="utf-8"))
    task["source"]["base_revision"] = revision
    (task_root / "task.json").write_text(
        json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    evaluator = task_root / "evaluator"
    evaluator.mkdir()
    shutil.copy(source / "evaluator/grade.py", evaluator / "grade.py")
    (task_root / "protocol.lock.json").write_text(json.dumps({
        "schema_version": 1,
        "task_id": task["task_id"],
        "protocol_id": "separated-coding-eval-cpu-v1",
        "note": "Hidden cases are not stored in the candidate task root.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fresh_workspace(parent: Path, solve_bytes: bytes) -> Path:
    workspace = parent / ("trial-" + secrets.token_hex(4))
    workspace.mkdir()
    (workspace / "solve.py").write_bytes(solve_bytes)
    (workspace / "solve.py").chmod(0o666)
    workspace.chmod(0o777)
    return workspace


async def grade_workspace(
    protocol: dict, *, workspace_root: Path, workspace: Path, snapshot_root: Path,
    policy_id: str, candidate_command: tuple[str, ...],
    check_oracles: bool,
) -> dict:
    runner = DockerIsolationRunner()
    grader = SeparatedCandidateGrader(
        runner=runner, limits=limits_for(protocol),
        allowed_workspace_root=workspace_root, snapshot_root=snapshot_root,
    )
    report = await grader.evaluate(
        task_id=protocol["task_id"], policy_id=policy_id, policy_version=1,
        workspace=workspace, candidate_command=candidate_command,
        scoring_command=tuple(protocol["execution"]["scoring_command"]),
        cases=cases_from(protocol), check_workspace_for_oracles=check_oracles,
    )
    payload = scoring_payload(report.scoring)
    payload["changed_files"] = list(report.changed_files)
    payload["candidate_mounts"] = list(report.candidate_mounts)
    payload["snapshot_files"] = report.snapshot.file_sha256
    solve = report.snapshot.path / "solve.py"
    if solve.is_file():
        payload["snapshot_solve_py"] = solve.read_text(encoding="utf-8")
    return payload


async def run_harness(protocol: dict, out: Path) -> dict:
    root = out / "harness-workspaces"
    snapshots = out / "harness-snapshots"
    root.mkdir()
    snapshots.mkdir()
    solve_bytes = (ROOT / "benchmarks/separated_json_square_v1/fixture/solve.py").read_bytes()
    expectations = protocol["harness_expectations"]
    runner = DockerIsolationRunner()
    probes = []
    for replicate in range(expectations["canary_replicates"]):
        canary_root = out / f"canary-{replicate + 1}"
        workspace = canary_root / "workspace"
        grader_root = canary_root / "grader-root"
        workspace.mkdir(parents=True)
        grader_root.mkdir()
        canary = "vare-synthetic-grader-canary-" + secrets.token_hex(16)
        (grader_root / "synthetic-hidden-canary.txt").write_text(canary, encoding="utf-8")
        observed = await _grader_probe(
            runner, canary_root, workspace, grader_root, canary,
            image=protocol["execution"]["container_image"],
        )
        observed["replicate"] = replicate + 1
        observed.pop("stdout", None)
        probes.append(observed)
    visible = sum(bool(item["canary_visible"]) for item in probes)

    broken = fresh_workspace(root, solve_bytes)
    known_bad = await grade_workspace(
        protocol, workspace_root=root, workspace=broken, snapshot_root=snapshots,
        policy_id="known-bad", candidate_command=("python", "-c", "print('unchanged')"),
        check_oracles=True,
    )
    fixed = (
        "import json\nimport sys\n\n"
        "def main():\n"
        "    request = json.loads(sys.stdin.read() or '{}')\n"
        "    number = request['n']\n"
        "    sys.stdout.write(json.dumps({'value': number * number}))\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    good_workspace = fresh_workspace(root, solve_bytes)
    known_good = await grade_workspace(
        protocol, workspace_root=root, workspace=good_workspace, snapshot_root=snapshots,
        policy_id="known-good",
        candidate_command=(
            "python", "-c",
            "from pathlib import Path; Path('/workspace/solve.py').write_text(" + json.dumps(fixed) + ")",
        ),
        check_oracles=True,
    )
    forged_workspace = fresh_workspace(root, solve_bytes)
    forged_command = (
        "from pathlib import Path\n"
        "Path('/workspace/result.json').write_text('{\"passed\": true}\\n')\n"
        "Path('/workspace/solve.py').write_text("
        + json.dumps("import json\nprint(json.dumps({'passed': True, 'score': 1}))\n")
        + ")\n"
    )
    forged = await grade_workspace(
        protocol, workspace_root=root, workspace=forged_workspace, snapshot_root=snapshots,
        policy_id="forged-score", candidate_command=("python", "-c", forged_command),
        check_oracles=True,
    )
    planted_workspace = fresh_workspace(root, solve_bytes)
    planted_command = (
        "from pathlib import Path\n"
        "Path('/workspace/oracle.json').write_text('{\"value\": 999}\\n')\n"
        "Path('/workspace/solve.py').write_text("
        + json.dumps(
            "import json\nfrom pathlib import Path\n"
            "print(Path('/workspace/oracle.json').read_text())\n"
        )
        + ")\n"
    )
    planted = await grade_workspace(
        protocol, workspace_root=root, workspace=planted_workspace, snapshot_root=snapshots,
        policy_id="planted-oracle", candidate_command=("python", "-c", planted_command),
        check_oracles=True,
    )
    grader_read_workspace = fresh_workspace(root, solve_bytes)
    grader_read_command = (
        "from pathlib import Path\n"
        "Path('/workspace/solve.py').write_text("
        + json.dumps(
            "from pathlib import Path\n"
            "try:\n"
            "    print(Path('/grader/synthetic-hidden-canary.txt').read_text())\n"
            "except OSError:\n"
            "    print('{\"value\": null}')\n"
        )
        + ")\n"
    )
    grader_read = await grade_workspace(
        protocol, workspace_root=root, workspace=grader_read_workspace, snapshot_root=snapshots,
        policy_id="grader-path-read", candidate_command=("python", "-c", grader_read_command),
        check_oracles=True,
    )
    network_workspace = fresh_workspace(root, solve_bytes)
    network = await runner.run_candidate(
        limits_for(protocol), allowed_workspace_root=root, workspace=network_workspace,
        command=[
            "python", "-c",
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('203.0.113.1', 65000), timeout=0.5)\n"
            "    print('NETWORK_OPEN')\n"
            "except OSError:\n"
            "    print('NETWORK_CLOSED')\n",
        ],
    )
    bad_ids = [case["case_id"] for case in known_bad["cases"] if case["passed"]]
    returncodes = [item["returncode"] for item in probes]
    returncodes.append(network.returncode)
    for check in (known_bad, known_good, forged, planted, grader_read):
        for case in check["cases"]:
            if type(case.get("returncode")) is int:
                returncodes.append(case["returncode"])
    environment_error = container_environment_error(returncodes, network.stderr)
    harness = {
        "evidence_level": "E0",
        "canary_visible": visible,
        "canary_replicates": len(probes),
        "canary_probes": probes,
        "known_bad_passing_case_ids": bad_ids,
        "known_good_passed": known_good["passed"],
        "forged_score_passed_cases": sum(case["passed"] for case in forged["cases"]),
        "planted_oracle_passed_cases": sum(case["passed"] for case in planted["cases"]),
        "grader_path_read_passed_cases": sum(case["passed"] for case in grader_read["cases"]),
        "network_stdout": network.stdout.strip(),
        "environment_error": environment_error,
        "checks": {
            "known_bad": known_bad,
            "known_good": known_good,
            "forged_score": forged,
            "planted_oracle": planted,
            "grader_path_read": grader_read,
        },
    }
    harness["matches_protocol"] = environment_error is None and bool(
        visible == expectations["canary_visible_replicates"]
        and bad_ids == expectations["known_bad_passing_case_ids"]
        and known_good["passed"] is True
        and harness["forged_score_passed_cases"] == 0
        and harness["planted_oracle_passed_cases"] == 0
        and harness["grader_path_read_passed_cases"] == 0
        and network.stdout.strip() == "NETWORK_CLOSED"
    )
    return harness


def diagnose(agent: dict, scoring: dict) -> str:
    edited = agent.get("source_hashes_before") != agent.get("source_hashes_after")
    tool_calls = [
        call for turn in agent.get("transcript", [])
        for call in turn.get("tool_calls", [])
        if isinstance(call, dict) and "name" in call
    ]
    if scoring.get("passed") is True or scoring.get("success") is True:
        return "passed"
    if not edited and not tool_calls:
        return "no_edit_no_tool_call"
    if not edited:
        return "no_source_change"
    return "edited_but_hidden_cases_failed"


def ensure_model(repository: str, revision: str, file_hashes: dict[str, str] | None) -> Path:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repository, revision=revision))
    if path.name != revision:
        raise SystemExit(f"snapshot directory {path.name} does not match pinned revision {revision}")
    if file_hashes:
        for name, expected in file_hashes.items():
            actual = sha256_file(path / name)
            if actual != expected:
                raise SystemExit(f"model file hash mismatch: {repository} {name}")
    return path


def run_agent(
    protocol: dict, *, model_path: Path, repository: str, seed: int,
    workspace: Path, task_root: Path, output: Path, instruction: str,
    arm: dict,
) -> dict:
    budgets = protocol["budgets"]
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(budgets["cpu_threads"]),
    })
    top_p = 0.95 if arm.get("top_p") is None else arm["top_p"]
    top_k = 20 if arm.get("top_k") is None else arm["top_k"]
    command = [
        sys.executable, str(ROOT / "scripts/run_local_agent.py"),
        "--workspace", str(workspace),
        "--model-path", str(model_path),
        "--model-repository", repository,
        "--backend", str(protocol["models"]["primary"].get("backend", "transformers_cpu")),
        "--public-test-image", str(protocol["execution"]["container_image"]),
        "--task-root", str(task_root),
        "--output", str(output),
        "--instruction", instruction,
        "--seed", str(seed),
        "--temperature", str(arm["temperature"]),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--edit-format", str(arm["edit_format"]),
        "--no-do-sample" if not arm["do_sample"] else "--do-sample",
        "--require-split", "train",
        "--max-turns", str(budgets["max_turns"]),
        "--max-new-tokens-per-turn", str(budgets["max_new_tokens_per_turn"]),
        "--max-total-new-tokens", str(budgets["max_total_new_tokens"]),
        "--max-context-tokens", str(budgets.get("max_context_tokens", 8192)),
        "--wall-seconds", str(budgets["wall_seconds"]),
    ]
    started = time.perf_counter()
    samples: list[dict] = []
    interval = max(1, int(budgets.get("resource_preflight", {}).get("sample_interval_seconds", 1)))
    thresholds = budgets.get("resource_preflight", {})
    proc = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    resource_limited = False
    deadline = started + budgets["wall_seconds"] + 120
    stdout = stderr = ""
    try:
        while proc.poll() is None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                timed_out = True
                proc.kill()
                break
            try:
                stdout, stderr = proc.communicate(timeout=min(interval, remaining))
            except subprocess.TimeoutExpired:
                try:
                    sample = host_resource_snapshot()
                    samples.append(sample)
                    if (sample["available_memory_bytes"] < thresholds.get("minimum_available_memory_bytes", 0)
                            or sample["swap_free_bytes"] < thresholds.get("minimum_free_swap_bytes", 0)):
                        resource_limited = True
                        proc.kill()
                        break
                except Exception as exc:
                    samples.append({"sampling_error": f"{type(exc).__name__}: {exc}"})
        stdout, stderr = proc.communicate()
    except Exception:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()
        raise
    record = {
        "returncode": proc.returncode,
        "wall_s": time.perf_counter() - started,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "timed_out": timed_out,
        "resource_limited": resource_limited,
        "host_resource_samples": samples,
    }
    if output.is_file():
        record["agent"] = json.loads(output.read_text(encoding="utf-8"))
    return record


def _outcome_for(agent: dict, *, task_passed: bool, timed_out: bool, source_changed: bool) -> str:
    return classify_outcome(
        task_passed=task_passed,
        stop_reason=None if not isinstance(agent.get("stop_reason"), str) else agent.get("stop_reason"),
        source_changed=source_changed,
        edit_attempted=bool(agent.get("edit_attempted")),
        edit_applied=bool(agent.get("edit_applied")),
        format_parse_failure=agent.get("format_parse_failure") if isinstance(agent.get("format_parse_failure"), str) else None,
        timed_out=timed_out,
    )


def attach_outcome(row: dict, agent: dict, *, task_passed: bool, timed_out: bool) -> None:
    source_changed = bool(row.get("source_changed"))
    row["edit_format"] = agent.get("edit_format")
    row["edit_attempted"] = bool(agent.get("edit_attempted"))
    row["edit_applied"] = bool(agent.get("edit_applied"))
    row["format_parse_failure"] = agent.get("format_parse_failure")
    row["outcome"] = _outcome_for(
        agent, task_passed=task_passed, timed_out=timed_out, source_changed=source_changed,
    )


def _arm_summary(arm: dict, rows: list[dict]) -> dict:
    summary = summarize_tasks(rows)
    summary.update({
        "arm": arm["name"],
        "edit_format": arm["edit_format"],
        "temperature": arm["temperature"],
        "do_sample": arm["do_sample"],
        "top_p": arm["top_p"],
        "top_k": arm["top_k"],
        "seeds": arm["seeds"],
        "per_task_reward_variance": per_task_reward_variance(rows),
        "pass_at_k": pass_at_k(rows),
    })
    return summary


def summarize_tasks(rows: list[dict]) -> dict:
    passed = sum(1 for row in rows if row.get("task_passed") is True)
    total = len(rows)
    if total == 0:
        raise ValueError("no train tasks were graded")
    low, high = wilson_interval(passed, total)
    return {
        "passed_tasks": passed,
        "tasks": total,
        "pass_rate": passed / total,
        "wilson95_low": low,
        "wilson95_high": high,
        "interval": "Wilson 95% on the pre-registered train tasks for the samples in this arm",
    }


PERFORMANCE_KIND = "hidden-correctness-and-repeated-timing"


def abort_reason_for_unpatched(
    task_id: str, reward_kind: str | None, *, decision: dict | None, error: str | None,
) -> str | None:
    """Return why the model must not start, or None when this baseline is usable."""
    if error:
        if reward_kind == PERFORMANCE_KIND and "wall-time" in error:
            return None
        return f"unpatched grader failed closed: {task_id}"
    if decision is not None and decision.get("success") is True:
        return f"unpatched train task was a protocol success: {task_id}"
    return None


def rebuild_manifest_text(pack_dir: Path) -> str:
    """Match scripts/hash_mlsys_coding_pack_v1.py without importing that branch."""
    files = []
    for path in sorted(item for item in pack_dir.rglob("*") if item.is_file()):
        if path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        files.append({
            "bytes": path.stat().st_size,
            "path": path.relative_to(pack_dir).as_posix(),
            "sha256": sha256_file(path),
        })
    manifest = {"files": files, "pack_id": "mlsys-coding-v1"}
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _git_head(start: Path) -> str:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            completed = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=False, text=True, capture_output=True,
            )
            if completed.returncode != 0:
                raise ValueError("pack git checkout has no HEAD")
            return completed.stdout.strip()
    raise ValueError("pack checkout has no git commit to verify")


def verify_pack(pack_root: Path, protocol: dict) -> Path:
    """Return the pack directory after the frozen commit and manifest match."""
    root = pack_root.expanduser().resolve()
    nested = root / "benchmarks" / "mlsys_coding_v1" / "manifest.json"
    if nested.is_file():
        pack_dir = nested.parent
    elif (root / "manifest.json").is_file():
        pack_dir = root
    else:
        raise ValueError("pack root is missing manifest.json")
    head = _git_head(pack_dir)
    if head != protocol["pack"]["git_commit"]:
        raise ValueError("pack git HEAD does not match the frozen commit")
    raw = (pack_dir / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != protocol["pack"]["manifest_sha256"]:
        raise ValueError("pack manifest SHA-256 does not match the frozen protocol")
    if raw.decode("utf-8") != rebuild_manifest_text(pack_dir):
        raise ValueError("pack manifest.json does not match the rebuilt file list")
    return pack_dir


def load_decide(pack_dir: Path):
    """Load reward.decide from the pack checkout. The file is not vendored."""
    path = pack_dir / "reward.py"
    if not path.is_file():
        raise ValueError("pack reward.py is missing")
    spec = importlib.util.spec_from_file_location("mlsys_coding_v1_reward", path)
    if spec is None or spec.loader is None:
        raise ValueError("pack reward.py cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decide = getattr(module, "decide", None)
    if not callable(decide):
        raise ValueError("pack reward.py has no decide function")
    return decide


def select_train_tasks(pack_dir: Path, protocol: dict, task_root_dir: Path) -> list[dict]:
    """Copy task.json only for train ids. Held-out tasks are not returned."""
    split = json.loads((pack_dir / "split.json").read_text(encoding="utf-8"))
    train = list(protocol["pack"]["train"])
    heldout = list(protocol["pack"]["heldout_not_used"])
    if split.get("train") != train or split.get("heldout") != heldout:
        raise ValueError("pack split.json does not match the frozen protocol")
    tasks_dir = pack_dir / "tasks"
    if not tasks_dir.is_dir():
        raise ValueError("pack tasks directory is missing")
    by_id: dict[str, tuple[Path, EngineeringTaskSpec]] = {}
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        spec_path = task_dir / "task.json"
        if not spec_path.is_file():
            continue
        spec = EngineeringTaskSpec.load(spec_path)
        if spec.id in by_id:
            raise ValueError(f"duplicate task id: {spec.id}")
        by_id[spec.id] = (task_dir, spec)
    missing = [task_id for task_id in train + heldout if task_id not in by_id]
    if missing:
        raise ValueError(f"pack is missing task ids: {missing}")
    extra = sorted(set(by_id) - set(train) - set(heldout))
    if extra:
        raise ValueError(f"pack has task ids outside the frozen split: {extra}")
    task_root_dir.mkdir(parents=True)
    selected = []
    for task_id in train:
        task_dir, spec = by_id[task_id]
        if spec.metadata.get("grader_visibility") != "hidden":
            raise ValueError(f"{task_id} is not a hidden-grader task")
        if spec.source.local_path != "subject":
            raise ValueError(f"{task_id} source is not the subject directory")
        destination = task_root_dir / task_id
        destination.mkdir()
        (destination / "task.json").write_bytes((task_dir / "task.json").read_bytes())
        contract = load_agent_task(destination)
        assert_train_only(contract, {"train": train, "heldout": heldout})
        if (destination / "evaluator").exists() or (destination / "subject").exists():
            raise ValueError("stripped task root contains hidden or subject files")
        selected.append({
            "task_id": task_id,
            "task_dir": task_dir,
            "task_root": destination,
            "allowed_files": contract.allowed_files,
            "prompt": spec.prompt,
            "reward_kind": spec.metadata.get("reward_kind"),
            "spec": spec,
        })
    return selected


def prepare_subject_workspace(
    subject: Path, workspace: Path, allowed_files: tuple[str, ...],
) -> str:
    if workspace.exists():
        raise ValueError("workspace already exists")
    workspace.mkdir(parents=True)
    for name in allowed_files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("editable path escapes the subject")
        source = subject / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"subject file missing: {name}")
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        target.chmod(0o644)
    workspace.chmod(0o755)
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "--", *allowed_files], cwd=workspace, check=True, capture_output=True)
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "vare",
        "GIT_AUTHOR_EMAIL": "vare@local",
        "GIT_COMMITTER_NAME": "vare",
        "GIT_COMMITTER_EMAIL": "vare@local",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    })
    subprocess.run(
        ["git", "commit", "-m", "base"], cwd=workspace, check=True,
        capture_output=True, env=env,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True, capture_output=True,
    ).stdout.strip()


async def grade_repository_workspace(
    *, image: str, workspace_root: Path, workspace: Path, snapshot_root: Path,
    stage_root: Path, task: dict, policy_id: str, decide,
) -> dict:
    """Exit a closed candidate container, then grade the snapshot on the host.

    ``decide`` is the pack reward. This path does not use it. A pass comes
    only from ``grade_isolated_subject``. Exit status is not a pass.
    """
    if not callable(decide):
        raise IsolationError("pack reward.decide is missing")
    calls_path = Path(task["task_dir"]) / "evaluator" / "isolated_calls.json"
    if not calls_path.is_file():
        return harness_required_record()
    probe_limits = IsolationLimits(
        image, timeout_s=20, memory_mb=256, cpus=0.5, pids_limit=32, max_output_bytes=16_000,
    )
    runner = DockerIsolationRunner()
    grader = SeparatedCandidateGrader(
        runner=runner, limits=probe_limits,
        allowed_workspace_root=workspace_root, snapshot_root=snapshot_root,
    )
    probe = await runner.run_candidate(
        probe_limits, allowed_workspace_root=workspace_root, workspace=workspace,
        command=["python", "-c", "from pathlib import Path; print(Path('/workspace').is_dir())"],
    )
    if probe.returncode != 0 or probe.stdout.strip() != "True":
        raise IsolationError("candidate container did not exit cleanly before the snapshot")
    snapshot = grader.capture_snapshot(workspace, candidate_exited=True)
    grade_tree = source_only_snapshot(snapshot, task["allowed_files"], snapshot_root)
    try:
        return grade_isolated_subject(grade_tree.path, calls_path, image=image)
    except IsolationError as exc:
        reason = abort_reason_for_unpatched(
            task["task_id"], task["reward_kind"], decision=None, error=str(exc),
        )
        if reason is None:
            return {
                "decision": {
                    "success": False, "reward": 0.0,
                    "conservative_speedup": None, "oracle_subject_ratio": None,
                    "workload_sufficient": False, "reason": "grader_wall_time",
                },
                "grader_error": str(exc),
            }
        raise


def maybe_secondary(protocol: dict, primary_rows: list[dict]) -> dict:
    walls = sorted(row["agent_wall_s"] for row in primary_rows if "agent_wall_s" in row)
    median = walls[len(walls) // 2] if walls else None
    available = mem_available_bytes()
    decision = {
        "primary_median_wall_s": median,
        "mem_available_bytes": available,
    }
    coder = protocol["models"]["secondary_if_feasible"]
    if median is None or median > 90 or available < 6 * 1024 ** 3:
        decision["coder_0_5b"] = "skipped_pre_registered_resource_condition"
        return decision
    decision["coder_0_5b"] = "eligible"
    large = protocol["models"]["not_run_unless"]
    if available < 8 * 1024 ** 3:
        decision["coder_1_5b"] = "skipped_insufficient_memory"
    else:
        decision["coder_1_5b"] = "eligible"
    decision["coder_0_5b_revision"] = coder["revision"]
    decision["coder_1_5b_revision"] = large["revision"]
    return decision


def resolve_image(protocol: dict, requested: str | None) -> str:
    accepted = set(protocol["execution"]["accepted_container_images"].values())
    image = protocol["execution"]["container_image"] if requested is None else requested
    if image not in accepted:
        raise ValueError("container image is not in the pre-registered set")
    return image


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def run_unpatched(image: str, tasks: list[dict], decide, out: Path) -> tuple[list[dict], str | None]:
    rows = []
    for task in tasks:
        task_out = out / task["task_id"]
        workspace_root = task_out / "workspaces"
        workspace_root.mkdir(parents=True)
        workspace = workspace_root / "trial"
        prepare_subject_workspace(task["task_dir"] / "subject", workspace, task["allowed_files"])
        try:
            graded = await grade_repository_workspace(
                image=image, workspace_root=workspace_root, workspace=workspace,
                snapshot_root=task_out / "snapshots", stage_root=task_out / "grader-stage",
                task=task, policy_id="unpatched-" + task["task_id"], decide=decide,
            )
        except IsolationError as exc:
            error = str(exc)
            rows.append({
                "task_id": task["task_id"], "task_passed": False, "grader_error": error,
            })
            reason = abort_reason_for_unpatched(
                task["task_id"], task["reward_kind"], decision=None, error=error,
            )
            if reason:
                return rows, reason
            continue
        if "grade" in graded:
            _write_json(task_out / "grade.json", graded["grade"])
        row = {
            "task_id": task["task_id"],
            "task_passed": graded["decision"]["success"] is True,
            "reward": graded["decision"]["reward"],
            "reward_reason": graded["decision"]["reason"],
        }
        if "grader_error" in graded:
            row["grader_error"] = graded["grader_error"]
        rows.append(row)
        reason = abort_reason_for_unpatched(
            task["task_id"], task["reward_kind"],
            decision=graded["decision"], error=graded.get("grader_error"),
        )
        if reason:
            return rows, reason
    return rows, None


async def run_model_task(
    protocol: dict, *, out: Path, model_path: Path, repository: str, seed: int,
    task: dict, image: str, decide, arm: dict,
) -> dict:
    task_out = out / task["task_id"]
    workspace_root = task_out / "workspaces"
    workspace_root.mkdir(parents=True)
    workspace = workspace_root / "trial"
    revision = prepare_subject_workspace(
        task["task_dir"] / "subject", workspace, task["allowed_files"],
    )
    agent_path = task_out / "agent.json"
    agent_record = run_agent(
        protocol, model_path=model_path, repository=repository, seed=seed,
        workspace=workspace, task_root=task["task_root"], output=agent_path,
        instruction=task["prompt"], arm=arm,
    )
    row = {
        "task_id": task["task_id"],
        "seed": seed,
        "repository": repository,
        "revision": model_path.name,
        "workspace_revision": revision,
        "agent_returncode": agent_record["returncode"],
        "agent_wall_s": agent_record["wall_s"],
        "host_resource_samples": agent_record.get("host_resource_samples", []),
        "resource_limited": agent_record.get("resource_limited", False),
        "stderr_tail": agent_record["stderr_tail"],
    }
    agent = agent_record.get("agent")
    if agent is None:
        row.update({
            "task_passed": False,
            "diagnosis": "agent_record_missing",
            "edit_format": arm["edit_format"],
            "format_parse_failure": None,
            "outcome": "timeout" if agent_record.get("timed_out") else "no_tool_call",
        })
        return row
    row["stop_reason"] = agent.get("stop_reason")
    row["new_tokens"] = agent.get("new_tokens")
    row["tool_turns"] = agent.get("tool_turns")
    row["candidate_diff_sha256"] = agent.get("candidate_diff_sha256")
    row["source_changed"] = agent.get("source_hashes_before") != agent.get("source_hashes_after")
    try:
        graded = await grade_repository_workspace(
            image=image, workspace_root=workspace_root, workspace=workspace,
            snapshot_root=task_out / "snapshots", stage_root=task_out / "grader-stage",
            task=task, policy_id=f"seed-{seed}", decide=decide,
        )
    except IsolationError as exc:
        message = str(exc)
        row.update({
            "task_passed": False,
            "diagnosis": "grader_failed_closed",
            "grader_error": message,
        })
        attach_outcome(
            row, agent, task_passed=False,
            timed_out=bool(agent_record.get("timed_out")) or "wall-time" in message,
        )
        return row
    if "grade" in graded:
        stored = dict(graded["grade"])
        _write_json(task_out / "grade.json", stored)
    row["task_passed"] = graded["decision"]["success"] is True
    row["reward"] = graded["decision"]["reward"]
    row["reward_reason"] = graded["decision"]["reason"]
    row["diagnosis"] = diagnose(agent, graded["decision"])
    test_timed_out = graded.get("test_timed_out") is True
    attach_outcome(
        row, agent, task_passed=row["task_passed"] is True,
        timed_out=bool(agent_record.get("timed_out")) or test_timed_out,
    )
    if "grader_error" in graded:
        row["grader_error"] = graded["grader_error"]
    return row


def _stopped(protocol: dict, reason: str, **extra) -> dict:
    payload = {
        "protocol_id": protocol["protocol_id"],
        "model_started": False,
        "reason": reason,
        "claim_limit": protocol["claim_limit"],
    }
    payload.update(extra)
    return payload


async def async_main(args: argparse.Namespace) -> int:
    out = args.output.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing evidence directory: {out}")
    protocol_path = args.protocol.expanduser().resolve()
    protocol = load_protocol(protocol_path)
    try:
        image = resolve_image(protocol, args.container_image)
        pack_dir = verify_pack(args.pack_root, protocol)
        decide = load_decide(pack_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    protocol["execution"]["container_image"] = image
    out.mkdir(parents=True)
    (out / "protocol.lock.json").write_bytes(protocol_path.read_bytes())
    source_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    source_dirty = bool(subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], check=True,
        text=True, capture_output=True,
    ).stdout.strip())
    _write_json(out / "resolved.json", {
        "container_image": image,
        "pack_commit": protocol["pack"]["git_commit"],
        "pack_dir": str(pack_dir),
        "runner_source_commit": source_commit,
        "runner_source_dirty_at_start": source_dirty,
    })
    try:
        tasks = select_train_tasks(pack_dir, protocol, out / "task-roots")
    except ValueError as exc:
        _write_json(out / "summary.json", _stopped(protocol, str(exc)))
        raise SystemExit(str(exc)) from exc
    heldout_roots = [
        path.name for path in (out / "task-roots").iterdir()
        if path.name in set(protocol["pack"]["heldout_not_used"])
    ]
    if heldout_roots:
        raise SystemExit("held-out task root was created")
    harness = await run_harness(protocol, out)
    _write_json(out / "harness.json", harness)
    if harness.get("environment_error"):
        _write_json(out / "summary.json", _stopped(protocol, harness["environment_error"]))
        print(harness["environment_error"], file=sys.stderr)
        return 2
    if not harness["matches_protocol"]:
        _write_json(out / "summary.json", _stopped(
            protocol, "E0 harness did not match the frozen expectations",
        ))
        print("harness mismatch; model tasks were not started", file=sys.stderr)
        return 2
    unpatched_rows, unpatched_abort = await run_unpatched(
        image, tasks, decide, out / "unpatched",
    )
    _write_json(out / "unpatched.json", {"rows": unpatched_rows, "abort": unpatched_abort})
    if unpatched_abort:
        _write_json(out / "summary.json", _stopped(protocol, unpatched_abort))
        print(unpatched_abort, file=sys.stderr)
        return 2

    resources, resource_failures = resource_preflight(protocol)
    _write_json(out / "resource-preflight.json", {
        "passed": not resource_failures, "observed": resources,
        "failures": resource_failures,
    })
    if resource_failures:
        reason = "resource preflight failed: " + ", ".join(resource_failures)
        _write_json(out / "summary.json", _stopped(protocol, reason, resources=resources))
        print(reason, file=sys.stderr)
        return 2

    primary = protocol["models"]["primary"]
    model_path = ensure_model(
        primary["repository"], primary["revision"], primary.get("file_hashes"),
    )
    arms = inference_arms(protocol)
    arm_reports = []
    rows: list[dict] = []
    primary_summary: dict | None = None
    resource_abort: str | None = None
    for arm in arms:
        arm_rows = []
        for seed in arm["seeds"]:
            for task in tasks:
                task_row = await run_model_task(
                    protocol, out=out / arm["name"] / f"seed-{seed}",
                    model_path=model_path, repository=primary["repository"], seed=seed,
                    task=task, image=image, decide=decide, arm=arm,
                )
                arm_rows.append(task_row)
                _write_json(out / f"{arm['name']}-tasks.json", {"rows": arm_rows})
                if task_row.get("resource_limited") is True:
                    resource_abort = "host resource floor reached; remaining model tasks were not started"
                    break
            if resource_abort:
                break
        arm_summary = _arm_summary(arm, arm_rows)
        arm_reports.append({"name": arm["name"], "summary": arm_summary, "rows": arm_rows})
        if primary_summary is None:
            rows = arm_rows
            primary_summary = arm_summary
        if resource_abort:
            break
    assert primary_summary is not None
    if len(arms) == 1 and arms[0]["name"] == "primary":
        _write_json(out / "primary-tasks.json", {"rows": rows})
    if "secondary_if_feasible" not in protocol["models"]:
        secondary = {"status": "not_in_protocol"}
    else:
        secondary = maybe_secondary(protocol, rows)
        if secondary.get("coder_0_5b") == "eligible":
            coder = protocol["models"]["secondary_if_feasible"]
            coder_path = ensure_model(coder["repository"], coder["revision"], None)
            coder_arm = dict(arms[0])
            coder_arm["name"] = "coder-0.5b"
            coder_rows = []
            for seed in coder["seeds"]:
                for task in tasks:
                    coder_rows.append(await run_model_task(
                        protocol, out=out / "coder-0.5b" / f"seed-{seed}",
                        model_path=coder_path, repository=coder["repository"], seed=seed,
                        task=task, image=image, decide=decide, arm=coder_arm,
                    ))
            secondary["summary"] = _arm_summary(coder_arm, coder_rows)
            secondary["rows"] = coder_rows
        available = mem_available_bytes()
        secondary["mem_available_before_coder_1_5b"] = available
        if available < 8 * 1024 ** 3:
            secondary["coder_1_5b"] = "skipped_insufficient_memory"
        else:
            large = protocol["models"]["not_run_unless"]
            large_path = ensure_model(large["repository"], large["revision"], None)
            large_arm = dict(arms[0])
            large_arm["name"] = "coder-1.5b"
            large_rows = []
            for seed in large["seeds"]:
                for task in tasks:
                    large_rows.append(await run_model_task(
                        protocol, out=out / "coder-1.5b" / f"seed-{seed}",
                        model_path=large_path, repository=large["repository"], seed=seed,
                        task=task, image=image, decide=decide, arm=large_arm,
                    ))
            secondary["coder_1_5b"] = "ran"
            secondary["coder_1_5b_summary"] = _arm_summary(large_arm, large_rows)
            secondary["coder_1_5b_rows"] = large_rows
    summary = {
        "protocol_id": protocol["protocol_id"],
        "evidence_levels": {
            "harness": "E0",
            "model": "real local CPU model; not a training result",
        },
        "container_image": image,
        "pack_commit": protocol["pack"]["git_commit"],
        "runner_source_commit": source_commit,
        "runner_source_dirty_at_start": source_dirty,
        "train_task_ids": [task["task_id"] for task in tasks],
        "model_started": True,
        "resource_abort": resource_abort,
        "attempted_task_runs": len(rows),
        "planned_task_runs": sum(len(arm["seeds"]) for arm in arms) * len(tasks),
        "primary": primary_summary,
        "primary_rows": rows,
        "arms": [{"name": item["name"], "summary": item["summary"]} for item in arm_reports],
        "secondary": secondary,
        "canary_visible": harness["canary_visible"],
        "canary_replicates": harness["canary_replicates"],
        "claim_limit": protocol["claim_limit"],
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps({
        "primary_passed_tasks": primary_summary["passed_tasks"],
        "primary_tasks": primary_summary["tasks"],
        "wilson95": [primary_summary["wilson95_low"], primary_summary["wilson95_high"]],
        "canary_visible": harness["canary_visible"],
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols/separated_coding_eval_cpu_v1.lock.json")
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--container-image", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
