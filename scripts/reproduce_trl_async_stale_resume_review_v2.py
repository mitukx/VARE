#!/usr/bin/env python3
"""Reproduce the replay-suffix epoch-count regression with source provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/trl_async_stale_resume_replay_suffix_v2.lock.json"
PROTOCOL_V3_PATH = ROOT / "protocols/trl_async_stale_resume_replay_suffix_v3.lock.json"
EXPECTED_CANDIDATE_TREES = {
    "baseline-failing-test": "1a6661764365ef38b07a0177711b8fac2d67b44d",
    "corrected-targeted-test": "5719aadfebf9ec559e98edae6076e1d1a29e5096",
    "corrected-full-test-file": "5719aadfebf9ec559e98edae6076e1d1a29e5096",
}
sys.path.insert(0, str(ROOT))
from scripts import reproduce_trl_async_stale_resume_v1 as stale_resume  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=text,
        timeout=30,
    )
    return result.stdout.strip() if text else result.stdout


def fail(message: str) -> None:
    raise ValueError(message)


def parse_provenance_markers(log: str) -> dict[str, str]:
    markers: dict[str, str] = {}
    for line in log.splitlines():
        if "VARE_REVIEW_" not in line:
            continue
        match = re.fullmatch(r"(VARE_REVIEW_[A-Z0-9_]+)=(.*)", line)
        if match is None:
            fail(f"malformed raw provenance marker: {line}")
        key, value = match.groups()
        if key in markers:
            fail(f"raw provenance marker is repeated: {key}")
        markers[key] = value
    return markers


def load_protocol(
    protocol_path: Path = PROTOCOL_PATH,
    *,
    require_current_runner: bool = True,
    require_current_helper: bool = True,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    runner = protocol["runner"]
    if runner["path"] != "scripts/reproduce_trl_async_stale_resume_review_v2.py":
        fail("protocol runner path mismatch")
    if require_current_runner and sha256_file(Path(__file__).resolve()) != runner["sha256"]:
        fail("runner source does not match the frozen protocol")
    helper = ROOT / runner["helper"]
    if require_current_helper and sha256_file(helper) != runner["helper_sha256"]:
        fail("shared stale-resume helper does not match the frozen protocol")
    source = protocol["source"]
    for key in ("original_candidate_patch", "regression_test_patch", "correction_patch"):
        path = ROOT / source[key]
        if sha256_file(path) != source[f"{key}_sha256"]:
            fail(f"patch hash mismatch: {source[key]}")
    environment = protocol["environment"]
    lock = ROOT / environment["environment_lock"]
    if sha256_file(lock) != environment["environment_lock_sha256"]:
        fail("locked environment protocol hash mismatch")
    return protocol


def write_manifest(bundle: Path) -> dict[str, str]:
    files = {
        path.relative_to(bundle).as_posix(): sha256_file(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(bundle / "manifest.json", {"files": files})
    return files


def capture_state(
    repo: Path,
    expected_paths: list[str],
    *,
    allow_unexpected: bool = False,
) -> dict[str, Any]:
    paths = sorted(str(git(repo, "diff", "--name-only", "HEAD")).splitlines())
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    status_entries = status.splitlines()
    unexpected_status = [
        entry
        for entry in status_entries
        if entry[3:] not in expected_paths or entry[:2] in {"??", "!!"}
    ]
    if not allow_unexpected and (
        paths != sorted(expected_paths) or unexpected_status
    ):
        fail(
            "candidate path inventory mismatch: "
            f"expected={sorted(expected_paths)}, got={paths}, "
            f"unexpected_status={unexpected_status}"
        )
    if paths:
        git(repo, "add", "--", *paths)
    tree = str(git(repo, "write-tree"))
    diff = git(repo, "diff", "--binary", "HEAD", text=False)
    if not isinstance(diff, bytes):
        fail("git did not return a binary source diff")
    return {
        "revision": str(git(repo, "rev-parse", "HEAD")),
        "base_tree": str(git(repo, "rev-parse", "HEAD^{tree}")),
        "candidate_tree": tree,
        "changed_paths": paths,
        "unexpected_status": unexpected_status,
        "diff": diff,
        "diff_sha256": sha256_bytes(diff),
        "test_file_sha256": sha256_file(repo / "tests/experimental/test_async_stale_resume.py"),
    }


def apply_patch(repo: Path, relative_patch: str) -> None:
    patch = ROOT / relative_patch
    subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(repo), "apply", str(patch)], check=True)


def execute_stage(
    *,
    protocol: dict[str, Any],
    python: str,
    repo: Path,
    bundle: Path,
    name: str,
    args: list[str],
    expected_paths: list[str],
    source_patch_name: str,
    runner_sha256: str,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    state = capture_state(repo, expected_paths)
    diff_path = bundle / source_patch_name
    diff_path.write_bytes(state.pop("diff"))
    state["diff_artifact"] = source_patch_name
    state["diff_artifact_sha256"] = sha256_file(diff_path)
    pytest_flags = (
        protocol.get("reproduction", {}).get("pytest_flags", [])
        if protocol.get("schema_version", 1) >= 3
        else []
    )
    command = [python, "-m", "pytest", "-o", "addopts=", "-q", *pytest_flags, *args]
    source = protocol["source"]
    environment = stale_resume.runtime_environment(
        python,
        repo,
        read_json(ROOT / protocol["environment"]["environment_lock"])["environment"],
    )
    header = [
        f"VARE_REVIEW_STAGE={name}",
        f"VARE_REVIEW_SOURCE_REVISION={state['revision']}",
        f"VARE_REVIEW_BASE_TREE={state['base_tree']}",
        f"VARE_REVIEW_CANDIDATE_TREE={state['candidate_tree']}",
        f"VARE_REVIEW_DIFF_SHA256={state['diff_artifact_sha256']}",
        f"VARE_REVIEW_TEST_SHA256={state['test_file_sha256']}",
        f"VARE_REVIEW_PROTOCOL_SHA256={sha256_file(protocol_path)}",
        f"VARE_REVIEW_RUNNER_SHA256={runner_sha256}",
        f"VARE_REVIEW_TEST_PATCH_SHA256={source['regression_test_patch_sha256']}",
        f"VARE_REVIEW_CORRECTION_PATCH_SHA256={source['correction_patch_sha256'] if name != 'baseline-failing-test' else 'not-applied'}",
    ]
    test_environment = stale_resume.pytest_environment()
    if protocol.get("schema_version", 1) >= 3:
        test_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    output, exit_code, timed_out = stale_resume.run_pytest_arm(
        command,
        repo,
        test_environment,
        protocol["environment"]["timeout_seconds"],
    )
    log_path = bundle / f"{name}.log"
    log_path.write_text(
        "\n".join(header)
        + "\n"
        + output.rstrip()
        + f"\nVARE_REVIEW_EXIT_CODE={exit_code}\n"
        + f"VARE_REVIEW_TIMED_OUT={'true' if timed_out else 'false'}\n",
        encoding="utf-8",
    )
    after: dict[str, Any] | None = None
    source_validation_error: str | None = None
    try:
        after = capture_state(repo, expected_paths, allow_unexpected=True)
    except Exception as exc:
        source_validation_error = f"{type(exc).__name__}: {exc}"
    source_stable = (
        after is not None
        and not after["unexpected_status"]
        and after["changed_paths"] == sorted(expected_paths)
        and after["candidate_tree"] == state["candidate_tree"]
        and after["diff_sha256"] == state["diff_artifact_sha256"]
    )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"VARE_REVIEW_SOURCE_VALID={'true' if source_stable else 'false'}\n")
        stream.write(f"VARE_REVIEW_SOURCE_STABLE={'true' if source_stable else 'false'}\n")
    record = {
        "name": name,
        **state,
        "diff": None,
        "command": command,
        "environment": environment,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "source_stable": source_stable,
        "source_validation_error": source_validation_error,
        "post_test_changed_paths": after["changed_paths"] if after else None,
        "post_test_unexpected_status": after["unexpected_status"] if after else None,
        "post_test_candidate_tree": after["candidate_tree"] if after else None,
        "post_test_diff_sha256": after["diff_sha256"] if after else None,
        "log": log_path.name,
        "raw_log_sha256": sha256_file(log_path),
    }
    record["pytest_outcome"] = stale_resume.parse_pytest_outcome(log_path.read_text(encoding="utf-8"), allow_missing=exit_code != 0)
    record.pop("diff", None)
    write_json(bundle / f"{name}.json", record)
    return record


def audit_bundle(
    bundle: Path,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    # The frozen protocol binds the retained execution runner in each bundle.
    # This auditor may evolve to reject defects found in that old runner, while
    # reproduction itself remains pinned to the original runner bytes.
    protocol = load_protocol(
        protocol_path,
        require_current_runner=False,
        require_current_helper=False,
    )
    manifest_path = bundle / "manifest.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, dict):
        fail("manifest files must be a mapping")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(entries):
        fail("manifest inventory does not match the bundle")
    for name, digest in entries.items():
        path = bundle / name
        if sha256_file(path) != digest:
            fail(f"manifest hash mismatch: {name}")

    summary = read_json(bundle / "summary.json")
    if summary.get("protocol_id") != protocol["protocol_id"]:
        fail("summary protocol id mismatch")
    protocol_snapshot = bundle / "protocol.json"
    if sha256_file(protocol_snapshot) != sha256_file(protocol_path):
        fail("retained protocol does not match the frozen protocol")
    if summary.get("protocol_sha256") != sha256_file(protocol_snapshot):
        fail("summary protocol hash mismatch")
    runner = bundle / "runner.py"
    if sha256_file(runner) != summary.get("runner_sha256") or summary.get("runner_sha256") != protocol["runner"]["sha256"]:
        fail("summary runner hash mismatch")
    if summary.get("helper_sha256") != protocol["runner"]["helper_sha256"]:
        fail("summary shared helper hash mismatch")
    source = protocol["source"]
    if summary.get("source_revision") != source["candidate_revision"] or summary.get("source_base_tree") != source["candidate_base_tree"]:
        fail("summary candidate source identity mismatch")
    if summary.get("source_runner_valid") is not True:
        fail("runner source changed during the reproduction")
    if summary.get("python_executable") != protocol["environment"]["python_executable"]:
        fail("summary Python executable differs from the locked interpreter")
    environment_lock = read_json(ROOT / protocol["environment"]["environment_lock"])
    expected_environment = stale_resume.expected_runtime_environment(environment_lock["environment"])
    if summary.get("environment") != expected_environment:
        fail("summary runtime environment differs from the locked dependency/platform inventory")

    records = summary.get("runs")
    expected_names = ["baseline-failing-test", "corrected-targeted-test", "corrected-full-test-file"]
    if not isinstance(records, list) or [record.get("name") for record in records] != expected_names:
        fail("run inventory mismatch")
    if summary.get("environment") != records[0].get("environment"):
        fail("summary environment does not match the recorded test runtime")
    expected_path_sets = {
        expected_names[0]: source["test_only_paths"],
        expected_names[1]: source["corrected_paths"],
        expected_names[2]: source["corrected_paths"],
    }
    for record in records:
        name = record["name"]
        if record.get("changed_paths") != expected_path_sets[name]:
            fail(f"changed source paths mismatch: {name}")
        if record.get("environment") != expected_environment:
            fail(f"recorded runtime environment differs from the lock: {name}")
        expected_command = [
            protocol["environment"]["python_executable"],
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
        ]
        if protocol.get("schema_version", 1) >= 3:
            expected_command.extend(protocol["reproduction"]["pytest_flags"])
        expected_command.append("tests/experimental/test_async_stale_resume.py")
        if name != "corrected-full-test-file":
            expected_command += ["-k", "resume_cursor_does_not_count_replayed_suffix"]
        if record.get("command") != expected_command:
            fail(f"executed test command differs from the protocol: {name}")
        if protocol.get("schema_version", 1) >= 3:
            if (
                record.get("source_validation_error") is not None
                or record.get("post_test_changed_paths") != expected_path_sets[name]
                or record.get("post_test_unexpected_status") != []
                or record.get("post_test_candidate_tree") != record.get("candidate_tree")
                or record.get("post_test_diff_sha256") != record.get("diff_sha256")
            ):
                fail(f"post-test source validation metadata is not clean: {name}")
        if record.get("revision") != source["candidate_revision"] or record.get("base_tree") != source["candidate_base_tree"]:
            fail(f"candidate revision or base tree mismatch: {name}")
        if record.get("candidate_tree") != EXPECTED_CANDIDATE_TREES[name]:
            fail(f"candidate tree differs from the frozen source state: {name}")
        if not record.get("source_stable") or record.get("timed_out"):
            fail(f"source changed or test timed out: {name}")
        if record.get("raw_log_sha256") != sha256_file(bundle / record["log"]):
            fail(f"raw log digest mismatch: {name}")
        expected_diff = "baseline-source.patch" if name == "baseline-failing-test" else "corrected-source.patch"
        if record.get("diff_artifact") != expected_diff or record.get("diff_sha256") != record.get("diff_artifact_sha256"):
            fail(f"source diff identity mismatch: {name}")
        if record.get("diff_artifact_sha256") != sha256_file(bundle / record["diff_artifact"]):
            fail(f"source diff digest mismatch: {name}")
        expected_patch_paths = [source["regression_test_patch"]]
        if name != "baseline-failing-test":
            expected_patch_paths.append(source["correction_patch"])
        expected_diff_bytes = b"".join((ROOT / path).read_bytes() for path in expected_patch_paths)
        recorded_diff_bytes = (bundle / record["diff_artifact"]).read_bytes()

        def normalize_patch(value: bytes) -> bytes:
            # `git diff --binary` abbreviates index object IDs according to the
            # checkout's config. Patch hunks and paths remain byte-identical.
            return b"\n".join(line for line in value.splitlines() if not line.startswith(b"index ")) + b"\n"

        if normalize_patch(recorded_diff_bytes) != normalize_patch(expected_diff_bytes):
            fail(f"source diff does not match the frozen patches: {name}")
        if read_json(bundle / f"{name}.json") != record:
            fail(f"retained stage metadata differs from the summary: {name}")
        log = (bundle / record["log"]).read_text(encoding="utf-8")
        expected_markers = {
            "VARE_REVIEW_STAGE": name,
            "VARE_REVIEW_SOURCE_REVISION": record["revision"],
            "VARE_REVIEW_BASE_TREE": record["base_tree"],
            "VARE_REVIEW_CANDIDATE_TREE": record["candidate_tree"],
            "VARE_REVIEW_DIFF_SHA256": record["diff_artifact_sha256"],
            "VARE_REVIEW_TEST_SHA256": record["test_file_sha256"],
            "VARE_REVIEW_PROTOCOL_SHA256": summary["protocol_sha256"],
            "VARE_REVIEW_RUNNER_SHA256": summary["runner_sha256"],
            "VARE_REVIEW_TEST_PATCH_SHA256": source["regression_test_patch_sha256"],
            "VARE_REVIEW_CORRECTION_PATCH_SHA256": (
                source["correction_patch_sha256"]
                if name != "baseline-failing-test"
                else "not-applied"
            ),
            "VARE_REVIEW_EXIT_CODE": str(record["exit_code"]),
            "VARE_REVIEW_TIMED_OUT": "true" if record["timed_out"] else "false",
            "VARE_REVIEW_SOURCE_VALID": "true",
            "VARE_REVIEW_SOURCE_STABLE": "true",
        }
        if parse_provenance_markers(log) != expected_markers:
            fail(f"raw log provenance markers mismatch: {name}")
        if record.get("pytest_outcome") != stale_resume.parse_pytest_outcome(log, allow_missing=record["exit_code"] != 0):
            fail(f"pytest outcome does not match raw log: {name}")
        outcome = record["pytest_outcome"]
        failures = sum(outcome.get(key, 0) for key in ("failed", "error", "errors"))
        if name == "baseline-failing-test":
            stage_passed = record["exit_code"] == 1 and outcome.get("failed") == 2 and failures == 2
        else:
            expected_passes = 2 if name == "corrected-targeted-test" else 11
            stage_passed = record["exit_code"] == 0 and outcome.get("passed") == expected_passes and failures == 0
        if not stage_passed:
            fail(f"pytest outcome or exit code does not satisfy the locked stage: {name}")

    baseline, targeted, full = records
    accepted = (
        baseline["exit_code"] == 1
        and baseline["pytest_outcome"].get("failed") == 2
        and targeted["exit_code"] == 0
        and targeted["pytest_outcome"].get("passed") == 2
        and full["exit_code"] == 0
        and full["pytest_outcome"].get("passed") == 11
        and all(record["source_stable"] and not record["timed_out"] for record in records)
    )
    if summary.get("all_checks_passed") is not accepted:
        fail("overall decision does not match the retained test evidence")
    if protocol.get("schema_version", 1) >= 3 and summary.get("status") != "complete":
        fail("complete v3 bundle has an unexpected status")
    return summary


def reproduce(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.expanduser().resolve() if args.protocol else PROTOCOL_PATH
    protocol = load_protocol(protocol_path)
    if args.python != protocol["environment"]["python_executable"]:
        fail(f"use the locked interpreter path: {protocol['environment']['python_executable']}")
    # Preserve the venv executable path; resolving its symlink would select
    # the base Python and bypass the locked environment.
    python = str(Path(args.python).expanduser().absolute())
    source_repo = args.source_repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        fail(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve()
    runner_bytes = runner_path.read_bytes()
    runner_sha = sha256_bytes(runner_bytes)
    (output / "runner.py").write_bytes(runner_bytes)
    protocol_bytes = protocol_path.read_bytes()
    (output / "protocol.json").write_bytes(protocol_bytes)
    helper_sha = sha256_file(ROOT / protocol["runner"]["helper"])
    environment_lock = read_json(ROOT / protocol["environment"]["environment_lock"])
    stale_resume.validate_local_inputs(environment_lock)
    with tempfile.TemporaryDirectory(prefix="trl-stale-resume-review-v2-") as temp:
        checkout = Path(temp) / "candidate"
        subprocess.run(
            ["git", "-C", str(source_repo), "worktree", "add", "--detach", str(checkout), protocol["source"]["candidate_revision"]],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            if str(
                git(
                    checkout,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--ignored=matching",
                )
            ):
                fail("materialized candidate is not clean before recording its revision")
            revision = str(git(checkout, "rev-parse", "HEAD"))
            base_tree = str(git(checkout, "rev-parse", "HEAD^{tree}"))
            if revision != protocol["source"]["candidate_revision"] or base_tree != protocol["source"]["candidate_base_tree"]:
                fail("materialized candidate does not match the locked source revision/tree")
            environment = stale_resume.runtime_environment(python, checkout, environment_lock["environment"])
            apply_patch(checkout, protocol["source"]["regression_test_patch"])
            runs = [execute_stage(
                protocol=protocol,
                python=python,
                repo=checkout,
                bundle=output,
                name="baseline-failing-test",
                args=["tests/experimental/test_async_stale_resume.py", "-k", "resume_cursor_does_not_count_replayed_suffix"],
                expected_paths=protocol["source"]["test_only_paths"],
                source_patch_name="baseline-source.patch",
                runner_sha256=runner_sha,
                protocol_path=protocol_path,
            )]
            if runs[-1]["source_stable"]:
                apply_patch(checkout, protocol["source"]["correction_patch"])
                runs.append(execute_stage(
                    protocol=protocol,
                    python=python,
                    repo=checkout,
                    bundle=output,
                    name="corrected-targeted-test",
                    args=["tests/experimental/test_async_stale_resume.py", "-k", "resume_cursor_does_not_count_replayed_suffix"],
                    expected_paths=protocol["source"]["corrected_paths"],
                    source_patch_name="corrected-source.patch",
                    runner_sha256=runner_sha,
                    protocol_path=protocol_path,
                ))
            if len(runs) == 2 and runs[-1]["source_stable"]:
                runs.append(execute_stage(
                    protocol=protocol,
                    python=python,
                    repo=checkout,
                    bundle=output,
                    name="corrected-full-test-file",
                    args=["tests/experimental/test_async_stale_resume.py"],
                    expected_paths=protocol["source"]["corrected_paths"],
                    source_patch_name="corrected-source.patch",
                    runner_sha256=runner_sha,
                    protocol_path=protocol_path,
                ))
        finally:
            subprocess.run(
                ["git", "-C", str(source_repo), "worktree", "remove", "--force", str(checkout)],
                check=False,
                capture_output=True,
                text=True,
            )
        try:
            source_runner_valid = sha256_file(runner_path) == runner_sha
        except OSError:
            source_runner_valid = False

    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": runner_sha,
        "helper_sha256": helper_sha,
        "source_revision": revision,
        "source_base_tree": base_tree,
        "python_executable": python,
        "environment": environment,
        "source_runner_valid": source_runner_valid,
        "runs": runs,
        "skipped_stages": [
            name
            for name in ("baseline-failing-test", "corrected-targeted-test", "corrected-full-test-file")
            if name not in {run["name"] for run in runs}
        ],
        "status": "complete" if len(runs) == 3 else "aborted_source_validation",
        "all_checks_passed": (
            source_runner_valid
            and len(runs) == 3
            and runs[0]["exit_code"] == 1
            and runs[0]["pytest_outcome"].get("failed") == 2
            and runs[1]["exit_code"] == 0
            and runs[1]["pytest_outcome"].get("passed") == 2
            and runs[2]["exit_code"] == 0
            and runs[2]["pytest_outcome"].get("passed") == 11
            and all(run["source_stable"] and not run["timed_out"] for run in runs)
        ),
    }
    write_json(output / "summary.json", summary)
    write_manifest(output)
    if len(runs) != 3:
        print(json.dumps({"bundle": str(output), "all_checks_passed": False, "status": summary["status"]}, indent=2))
        return 1
    audited = audit_bundle(output, protocol_path)
    print(json.dumps({"bundle": str(output), "all_checks_passed": audited["all_checks_passed"]}, indent=2))
    return 0 if audited["all_checks_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, help="local TRL clone containing the locked candidate commit")
    parser.add_argument("--python", default="/tmp/vare-trl-v4/bin/python", help="locked v4 Python 3.12 environment")
    parser.add_argument("--protocol", type=Path, help="explicit frozen protocol (use the v3 lock for the hardened runner)")
    parser.add_argument("--output", type=Path, help="new output directory")
    parser.add_argument("--audit", type=Path, help="audit an existing retained v2 bundle")
    args = parser.parse_args()
    try:
        if args.audit:
            protocol_path = args.protocol.expanduser().resolve() if args.protocol else PROTOCOL_PATH
            audited = audit_bundle(args.audit.expanduser().resolve(), protocol_path)
            print(json.dumps(audited, indent=2))
            return 0 if audited.get("all_checks_passed") is True else 1
        if args.source_repo is None or args.output is None:
            parser.error("--source-repo and --output are required unless --audit is supplied")
        return reproduce(args)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
        print(f"reproduction failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
