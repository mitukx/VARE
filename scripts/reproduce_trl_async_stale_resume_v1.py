#!/usr/bin/env python3
"""Replay the pinned TRL stale-drop checkpoint/resume comparison on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/trl_async_stale_resume_v2.lock.json"
LOCK_PATHS = {
    "trl-async-stale-resume-v1": ROOT / "protocols/trl_async_stale_resume_v1.lock.json",
    "trl-async-stale-resume-v2": LOCK_PATH,
    "trl-async-stale-resume-v3": ROOT / "protocols/trl_async_stale_resume_v3.lock.json",
    "trl-async-stale-resume-v4": ROOT / "protocols/trl_async_stale_resume_v4.lock.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+~-]+)$")
TIMEOUT_MARKER = "VARE_RUNNER_TIMEOUT_SECONDS="
EXIT_CODE_MARKER = "VARE_RUNNER_EXIT_CODE="
LOG_PATTERNS = {
    "first_run_trained_rows": r"run 1 trained rows\s*:\s*(\d+)",
    "resumed_row": r"resumed at row\s*:\s*(\d+)",
    "duplicated_rows": r"rows trained TWICE\s*:\s*(\d+)",
    "fresh_rows": r"rows new in run 2\s*:\s*(\d+)",
    "saved_prompt_index": r"saved prompt_index\s*:\s*(\d+)",
    "groups_before_resume": r"_groups_before_resume\s*:\s*(\d+)",
}
PYTEST_COUNT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|warnings?)\b")
PACKAGE_NAMES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "pytest": "pytest",
    "tokenizers": "tokenizers",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "pyarrow": "pyarrow",
}
RUNNER_SNAPSHOTS = {
    "4450fcebb731178f448bd68d5650219e851c7db09f59c1f91be0632f37ad63c3": (
        ROOT / "results/trl-async-stale-resume-runner-snapshots/"
        "runner-4450fcebb731178f448bd68d5650219e851c7db09f59c1f91be0632f37ad63c3.py"
    ),
    "362907df0e07bcd0c6ab75d3f5390551cd0fcf054c38e0d2d1488c606faaeef4": (
        ROOT / "results/trl-async-stale-resume-runner-snapshots/"
        "runner-362907df0e07bcd0c6ab75d3f5390551cd0fcf054c38e0d2d1488c606faaeef4.py"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fail(message: str) -> None:
    raise ValueError(message)


def normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def resolved_requirements(environment: dict[str, Any]) -> dict[str, str] | None:
    relative = environment.get("resolved_requirements_lock")
    if not relative:
        return None
    requirements: dict[str, str] = {}
    for line_number, raw_line in enumerate((ROOT / relative).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = REQUIREMENT_RE.fullmatch(line)
        if not match:
            fail(f"resolved requirements must be exact package pins: {relative}:{line_number}")
        name, version = match.groups()
        normalized = normalize_distribution_name(name)
        if normalized in requirements:
            fail(f"duplicate package in resolved requirements: {normalized}")
        requirements[normalized] = version
    return requirements


def validate_local_inputs(lock: dict[str, Any]) -> None:
    sources = lock["source"]
    reproduction = lock["reproducer"]
    environment = lock["environment"]
    files = (
        (sources["fix_patch"], sources["fix_patch_sha256"]),
        (reproduction["current_api_patch"], reproduction["current_api_patch_sha256"]),
        (
            reproduction["baseline_tail_tolerance_patch"],
            reproduction["baseline_tail_tolerance_patch_sha256"],
        ),
        (
            reproduction["candidate_adaptation_patch"],
            reproduction["candidate_adaptation_patch_sha256"],
        ),
        (environment["requirements_lock"], environment["requirements_lock_sha256"]),
    )
    stream_patch = reproduction.get("stream_position_patch")
    if stream_patch:
        files += ((stream_patch, reproduction["stream_position_patch_sha256"]),)
    resolved_lock = environment.get("resolved_requirements_lock")
    if resolved_lock:
        files += ((resolved_lock, environment["resolved_requirements_lock_sha256"]),)
    for relative, expected in files:
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            fail(f"locked input hash mismatch: {relative}")


def source_revision(source: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        fail(f"source checkout must be clean before reproducing: {source}: {status.strip()}")
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def execution_platform(python: str) -> dict[str, Any]:
    probe = """import json, platform, subprocess, sys
def sysctl_value(name):
    if sys.platform != "darwin":
        return None
    result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None
memory = sysctl_value("hw.memsize")
print(json.dumps({
    "sys_platform": sys.platform,
    "system": platform.system(),
    "release": platform.release(),
    "machine": platform.machine(),
    "mac_ver": platform.mac_ver()[0],
    "hardware_model": sysctl_value("hw.model"),
    "processor": sysctl_value("machdep.cpu.brand_string"),
    "memory_bytes": int(memory) if memory and memory.isdigit() else None,
}))
"""
    result = subprocess.run([python, "-c", probe], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def runtime_environment(python: str, source: Path, expected: dict[str, Any]) -> dict[str, Any]:
    resolved = resolved_requirements(expected)
    if resolved is None:
        names = list(PACKAGE_NAMES.values())
        probe = (
            "import importlib.metadata, json, sys; "
            f"names = {names!r}; "
            "versions = {name: importlib.metadata.version(name) for name in names}; "
            "print(json.dumps({'python': '.'.join(map(str, sys.version_info[:3])), 'versions': versions}))"
        )
    else:
        probe = (
            "import importlib.metadata as metadata\n"
            "import json, re, sys\n"
            "normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()\n"
            "versions = {}\n"
            "duplicates = []\n"
            "for distribution in metadata.distributions():\n"
            "    name = normalize(distribution.metadata.get('Name', ''))\n"
            "    if name in versions:\n"
            "        duplicates.append(name)\n"
            "    else:\n"
            "        versions[name] = distribution.version\n"
            "print(json.dumps({'python': '.'.join(map(str, sys.version_info[:3])), "
            "'versions': versions, 'duplicate_names': duplicates}))"
        )
    result = subprocess.run(
        [python, "-c", probe], cwd=source, check=True, capture_output=True, text=True
    )
    observed = json.loads(result.stdout)
    if observed["python"] != expected["python"]:
        fail(f"Python version mismatch: expected {expected['python']}, got {observed['python']}")
    expected_versions = resolved or {
        PACKAGE_NAMES[key]: value for key, value in expected.items() if key in PACKAGE_NAMES
    }
    if observed.get("duplicate_names"):
        fail(f"duplicate installed distribution names: {observed['duplicate_names']}")
    if observed["versions"] != expected_versions:
        changed = sorted(
            name
            for name in set(expected_versions) & set(observed["versions"])
            if expected_versions[name] != observed["versions"][name]
        )
        fail(
            "dependency version inventory mismatch: "
            f"missing={sorted(set(expected_versions) - set(observed['versions']))}, "
            f"unexpected={sorted(set(observed['versions']) - set(expected_versions))}, "
            f"changed={changed}"
        )
    observed.pop("duplicate_names", None)
    if "execution_platform" in expected:
        observed_platform = execution_platform(python)
        if observed_platform != expected["execution_platform"]:
            fail(
                "execution platform mismatch: "
                f"expected {expected['execution_platform']}, got {observed_platform}"
            )
        observed["execution_platform"] = observed_platform
        observed["device_used"] = expected["device_used"]
    return observed


def expected_runtime_environment(expected: dict[str, Any]) -> dict[str, Any]:
    resolved = resolved_requirements(expected)
    runtime = {
        "python": expected["python"],
        "versions": resolved
        or {PACKAGE_NAMES[key]: value for key, value in expected.items() if key in PACKAGE_NAMES},
    }
    if "execution_platform" in expected:
        runtime["execution_platform"] = expected["execution_platform"]
        runtime["device_used"] = expected["device_used"]
    return runtime


def run_pytest_arm(
    command: list[str], source: Path, env: dict[str, str], timeout_seconds: int
) -> tuple[str, int, bool]:
    try:
        result = subprocess.run(
            command,
            cwd=source,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\n{TIMEOUT_MARKER}{timeout_seconds}\n"
        return output, 124, True
    return result.stdout, result.returncode, False


def protocol_timeout_seconds(lock: dict[str, Any]) -> int:
    timeout = lock["reproducer"].get("per_run_timeout_seconds")
    if type(timeout) is not int or timeout <= 0:
        fail("protocol has no valid locked per-run timeout; use trl-async-stale-resume-v3")
    return timeout


def validate_new_run_protocol(lock: dict[str, Any]) -> None:
    environment = lock.get("environment", {})
    if not isinstance(environment.get("execution_platform"), dict) or not environment.get(
        "device_used"
    ):
        fail("new runs require a protocol lock with execution-platform fields (v4 or newer)")


def download_gist(lock: dict[str, Any], gist_path: Path | None) -> bytes:
    source = lock["reproducer"]
    if gist_path:
        content = gist_path.read_bytes()
    else:
        with urllib.request.urlopen(source["public_source"], timeout=30) as response:
            content = response.read()
    observed = hashlib.sha256(content).hexdigest()
    if observed != source["public_source_sha256"]:
        fail(f"public reproduction source hash mismatch: {observed}")
    return content


def apply_patch(script_dir: Path, patch_path: Path) -> None:
    result = subprocess.run(
        ["patch", "-p0"],
        cwd=script_dir,
        input=patch_path.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        fail(f"could not apply {patch_path.name}: {result.stdout}{result.stderr}")


def prepare_runner(
    gist_bytes: bytes, temp_root: Path, arm: str, lock: dict[str, Any]
) -> Path:
    script_dir = temp_root / arm
    script_dir.mkdir()
    script_path = script_dir / "async_grpo_resume_cursor_repro.py"
    script_path.write_bytes(gist_bytes)
    source = lock["reproducer"]
    patch_files = [source["current_api_patch"]]
    if arm == "baseline":
        patch_files.append(source["baseline_tail_tolerance_patch"])
    else:
        patch_files.append(source["candidate_adaptation_patch"])
    if source.get("stream_position_patch"):
        patch_files.append(source["stream_position_patch"])
    for relative in patch_files:
        apply_patch(script_dir, ROOT / relative)
    return script_path


def parse_run_log(text: str, *, allow_incomplete: bool = False) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for name, pattern in LOG_PATTERNS.items():
        match = re.search(pattern, text)
        if not match and not allow_incomplete:
            fail(f"reproduction output lacks {name}")
        if match:
            parsed[name] = int(match.group(1))
    return parsed


def parse_pytest_outcome(text: str, *, allow_missing: bool = False) -> dict[str, int]:
    summaries = []
    for line in text.splitlines():
        if re.search(r"\bin\s+\d+(?:\.\d+)?s\s*$", line):
            counts = {name: int(count) for count, name in PYTEST_COUNT_PATTERN.findall(line)}
            if counts:
                summaries.append(counts)
    if len(summaries) > 1:
        fail("raw log contains multiple pytest result summaries")
    if not summaries:
        if allow_missing:
            return {}
        fail("raw log lacks a pytest result summary")
    return summaries[0]


def run_acceptance(arm: str, observed: dict[str, int], exit_code: int, lock: dict[str, Any]) -> bool:
    if not set(LOG_PATTERNS).issubset(observed):
        return False
    stale_id = lock["reproducer"]["stale_group_id"]
    tail = lock["reproducer"]["tail_tolerance_groups"]
    if arm == "baseline":
        return (
            exit_code == 0
            and observed["saved_prompt_index"] == stale_id
            and observed["resumed_row"] == stale_id
            and observed["duplicated_rows"] >= observed["first_run_trained_rows"] - tail
        )
    return (
        exit_code == 0
        and observed["saved_prompt_index"] > stale_id
        and observed["resumed_row"] > stale_id
        and observed["groups_before_resume"] == observed["first_run_trained_rows"]
        and observed["duplicated_rows"] < tail
    )


def write_manifest(bundle: Path) -> dict[str, str]:
    files = {
        path.relative_to(bundle).as_posix(): sha256_file(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(bundle / "manifest.json", {"files": files})
    return files


def audit_bundle(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, dict):
        fail("manifest files must be a mapping")
    expected: set[str] = set()
    for name, digest in entries.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            fail(f"unsafe manifest path: {name}")
        if not SHA256_RE.fullmatch(str(digest)):
            fail(f"malformed hash for {name}")
        path = bundle.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file():
            fail(f"missing manifest entry: {name}")
        if sha256_file(path) != digest:
            fail(f"hash mismatch: {name}")
        expected.add(relative.as_posix())
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        fail(f"manifest inventory mismatch: missing={sorted(expected-actual)}, unlisted={sorted(actual-expected)}")

    summary = read_json(bundle / "summary.json")
    if type(summary.get("all_checks_passed")) is not bool:
        fail("overall decision must be a boolean")
    lock_path = LOCK_PATHS.get(summary.get("protocol_id"))
    if lock_path is None:
        fail(f"unknown protocol id: {summary.get('protocol_id')}")
    lock = read_json(lock_path)
    if summary.get("protocol_id") != lock["protocol_id"]:
        fail("protocol id mismatch")
    if summary.get("protocol_sha256") != sha256_file(lock_path):
        fail("protocol hash mismatch")
    runner_sha = summary.get("runner_sha256")
    if not isinstance(runner_sha, str) or not SHA256_RE.fullmatch(runner_sha):
        fail("summary has no valid runner source hash")
    bundled_runner = bundle / "runner.py"
    runner_snapshot = bundled_runner if bundled_runner.is_file() else RUNNER_SNAPSHOTS.get(runner_sha)
    if runner_snapshot is None or not runner_snapshot.is_file():
        fail(f"runner source is not retained for hash: {runner_sha}")
    if sha256_file(runner_snapshot) != runner_sha:
        fail("retained runner source hash mismatch")
    validate_local_inputs(lock)
    if summary.get("environment") != expected_runtime_environment(lock["environment"]):
        fail("recorded runtime environment does not match the locked dependency versions")
    if summary.get("gist_source_sha256") != lock["reproducer"]["public_source_sha256"]:
        fail("recorded Gist source hash does not match the locked reproducer")
    expected_repetitions = lock["reproducer"]["repetitions_per_arm"]
    expected_count = expected_repetitions * 2
    expected_revisions = {
        "baseline": lock["source"]["baseline_revision"],
        "candidate": lock["source"]["candidate_revision"],
    }
    if summary.get("source_revisions") != expected_revisions:
        fail("source revisions do not match the locked baseline and candidate")
    if summary.get("repetitions_per_arm") != expected_repetitions:
        fail("repetitions per arm mismatch")
    runs = summary.get("runs")
    if not isinstance(runs, list) or summary.get("run_count") != expected_count or len(runs) != expected_count:
        fail("run count mismatch")

    expected_runs = {
        (arm, repetition)
        for arm in ("baseline", "candidate")
        for repetition in range(1, expected_repetitions + 1)
    }
    observed_runs: set[tuple[str, int]] = set()
    for run in runs:
        if not isinstance(run, dict):
            fail("run inventory contains a non-object entry")
        if type(run.get("accepted")) is not bool:
            fail("run acceptance must be a boolean")
        arm = run.get("arm")
        repetition = run.get("repetition")
        if arm not in ("baseline", "candidate") or type(repetition) is not int:
            fail("run inventory contains an invalid arm or repetition")
        run_id = (arm, repetition)
        if run_id in observed_runs:
            fail(f"duplicate run identity: {run_id}")
        observed_runs.add(run_id)
        expected_log = f"{arm}-{repetition}.log"
        if run.get("log") != expected_log:
            fail(f"run log does not match its identity: {run_id}")
    if observed_runs != expected_runs:
        fail(
            "run inventory mismatch: "
            f"missing={sorted(expected_runs - observed_runs)}, "
            f"unexpected={sorted(observed_runs - expected_runs)}"
        )

    for run in runs:
        raw_log = (bundle / run["log"]).read_text(encoding="utf-8")
        exit_code = run.get("exit_code")
        if type(exit_code) is not int:
            fail(f"recorded exit code is not an integer: {run['log']}")
        exit_code_markers = re.findall(rf"(?m)^{re.escape(EXIT_CODE_MARKER)}(\d+)\s*$", raw_log)
        if exit_code_markers and (
            len(exit_code_markers) != 1 or int(exit_code_markers[0]) != exit_code
        ):
            fail(f"raw log exit code marker contradicts recorded result: {run['log']}")
        timed_out = run.get("timed_out", False)
        if "per_run_timeout_seconds" in lock["reproducer"]:
            timeout_seconds = lock["reproducer"]["per_run_timeout_seconds"]
            if run.get("timeout_seconds") != timeout_seconds or type(timed_out) is not bool:
                fail(f"timeout metadata does not match the locked protocol: {run['log']}")
            marker = f"{TIMEOUT_MARKER}{timeout_seconds}"
            if timed_out:
                if run.get("exit_code") != 124 or marker not in raw_log:
                    fail(f"timed-out arm is not recorded as a failed result: {run['log']}")
            elif marker in raw_log:
                fail(f"raw log contains an unrecorded timeout: {run['log']}")
            if timed_out:
                pytest_passed = False
                parsed = parse_run_log(raw_log, allow_incomplete=True)
            else:
                pytest_outcome = parse_pytest_outcome(raw_log, allow_missing=exit_code != 0)
                pytest_passed = (
                    pytest_outcome.get("passed", 0) > 0
                    and pytest_outcome.get("failed", 0) == 0
                    and pytest_outcome.get("error", pytest_outcome.get("errors", 0)) == 0
                )
                parsed = parse_run_log(raw_log, allow_incomplete=exit_code != 0)
        else:
            pytest_outcome = parse_pytest_outcome(raw_log, allow_missing=exit_code != 0)
            pytest_passed = (
                pytest_outcome.get("passed", 0) > 0
                and pytest_outcome.get("failed", 0) == 0
                and pytest_outcome.get("error", pytest_outcome.get("errors", 0)) == 0
            )
            parsed = parse_run_log(raw_log, allow_incomplete=exit_code != 0)
        if (exit_code == 0) != pytest_passed:
            fail(f"recorded exit code contradicts raw pytest result: {run['log']}")
        if parsed != run["observed"]:
            fail(f"summary/log mismatch: {run['log']}")
        accepted = run_acceptance(run["arm"], parsed, run["exit_code"], lock)
        if accepted != run["accepted"]:
            fail(f"acceptance/log mismatch: {run['log']}")
    if summary["all_checks_passed"] != all(run["accepted"] for run in summary["runs"]):
        fail("overall decision does not match run decisions")
    return summary


def reproduce(args: argparse.Namespace) -> int:
    protocol_path = LOCK_PATHS[args.protocol]
    lock = read_json(protocol_path)
    validate_new_run_protocol(lock)
    timeout_seconds = protocol_timeout_seconds(lock)
    validate_local_inputs(lock)
    base = args.baseline_source.expanduser().resolve()
    candidate = args.candidate_source.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        fail(f"refusing to overwrite nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve()
    runner_bytes = runner_path.read_bytes()
    runner_sha = sha256_bytes(runner_bytes)
    (out / "runner.py").write_bytes(runner_bytes)

    expected_source = lock["source"]
    revisions = {"baseline": source_revision(base), "candidate": source_revision(candidate)}
    if revisions["baseline"] != expected_source["baseline_revision"]:
        fail(f"baseline source revision mismatch: {revisions['baseline']}")
    if revisions["candidate"] != expected_source["candidate_revision"]:
        fail(f"candidate source revision mismatch: {revisions['candidate']}")
    env = runtime_environment(args.python, base, lock["environment"])
    gist_bytes = download_gist(lock, args.gist_file)
    runs: list[dict[str, Any]] = []
    test_env = dict(os.environ)
    test_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    test_env["TRL_EXPERIMENTAL_SILENCE"] = "1"

    with tempfile.TemporaryDirectory(prefix="trl-async-stale-resume-") as temp_name:
        temp_root = Path(temp_name)
        scripts = {
            arm: prepare_runner(gist_bytes, temp_root, arm, lock)
            for arm in ("baseline", "candidate")
        }
        for repetition in range(1, lock["reproducer"]["repetitions_per_arm"] + 1):
            for arm, source_root in (("baseline", base), ("candidate", candidate)):
                if source_revision(source_root) != revisions[arm]:
                    fail(f"{arm} source revision changed during reproduction")
                log_rel = f"{arm}-{repetition}.log"
                log_path = out / log_rel
                output, exit_code, timed_out = run_pytest_arm(
                    [args.python, "-m", "pytest", str(scripts[arm]), "-q", "-s"],
                    source_root,
                    test_env,
                    timeout_seconds,
                )
                output = f"{output.rstrip()}\n{EXIT_CODE_MARKER}{exit_code}\n"
                if source_revision(source_root) != revisions[arm]:
                    fail(f"{arm} source checkout changed while running repetition {repetition}")
                log_path.write_text(output, encoding="utf-8")
                observed = parse_run_log(output, allow_incomplete=timed_out or exit_code != 0)
                accepted = run_acceptance(arm, observed, exit_code, lock)
                run = {
                    "arm": arm,
                    "repetition": repetition,
                    "exit_code": exit_code,
                    "log": log_rel,
                    "observed": observed,
                    "accepted": accepted,
                }
                run["timed_out"] = timed_out
                run["timeout_seconds"] = timeout_seconds
                runs.append(run)
                print(f"{arm}-{repetition}: exit={exit_code}, timed_out={timed_out}, accepted={accepted}, {observed}", flush=True)

    summary = {
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": runner_sha,
        "python_executable": str(Path(args.python).resolve()),
        "environment": env,
        "source_revisions": revisions,
        "gist_source_sha256": hashlib.sha256(gist_bytes).hexdigest(),
        "repetitions_per_arm": lock["reproducer"]["repetitions_per_arm"],
        "run_count": len(runs),
        "runs": runs,
        "all_checks_passed": all(run["accepted"] for run in runs),
    }
    if sha256_file(runner_path) != runner_sha:
        fail("reproducer source changed during execution")
    write_json(out / "summary.json", summary)
    write_manifest(out)
    audited = audit_bundle(out)
    print(json.dumps({"bundle": str(out), "all_checks_passed": audited["all_checks_passed"]}, indent=2))
    return 0 if audited["all_checks_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True, help="TRL checkout at the pinned baseline revision")
    parser.add_argument("--candidate-source", type=Path, required=True, help="TRL checkout at the pinned candidate revision")
    parser.add_argument("--python", default=sys.executable, help="Python executable with the pinned CPU dependencies")
    parser.add_argument("--gist-file", type=Path, help="use a local copy of the pinned public Gist instead of downloading it")
    parser.add_argument("--output", type=Path, required=True, help="new or empty directory for logs and manifest")
    parser.add_argument(
        "--protocol",
        choices=sorted(LOCK_PATHS),
        default="trl-async-stale-resume-v4",
        help="frozen protocol to run (new runs require platform-bound v4 or newer)",
    )
    args = parser.parse_args()
    try:
        return reproduce(args)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
        print(f"reproduction failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
