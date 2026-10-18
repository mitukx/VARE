#!/usr/bin/env python3
"""Replay the pinned TRL stale-drop checkpoint/resume comparison on CPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata  # noqa: F401
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
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOG_PATTERNS = {
    "first_run_trained_rows": r"run 1 trained rows\s*:\s*(\d+)",
    "resumed_row": r"resumed at row\s*:\s*(\d+)",
    "duplicated_rows": r"rows trained TWICE\s*:\s*(\d+)",
    "fresh_rows": r"rows new in run 2\s*:\s*(\d+)",
    "saved_prompt_index": r"saved prompt_index\s*:\s*(\d+)",
    "groups_before_resume": r"_groups_before_resume\s*:\s*(\d+)",
}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fail(message: str) -> None:
    raise ValueError(message)


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
    for relative, expected in files:
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            fail(f"locked input hash mismatch: {relative}")


def source_revision(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def runtime_environment(python: str, source: Path, expected: dict[str, str]) -> dict[str, Any]:
    probe = (
        "import importlib.metadata, json, sys; "
        "names = " + repr(list(PACKAGE_NAMES.values())) + "; "
        "versions = {name: importlib.metadata.version(name) for name in names}; "
        "print(json.dumps({'python': '.'.join(map(str, sys.version_info[:3])), 'versions': versions}))"
    )
    result = subprocess.run(
        [python, "-c", probe], cwd=source, check=True, capture_output=True, text=True
    )
    observed = json.loads(result.stdout)
    if observed["python"] != expected["python"]:
        fail(f"Python version mismatch: expected {expected['python']}, got {observed['python']}")
    expected_versions = {PACKAGE_NAMES[key]: value for key, value in expected.items() if key in PACKAGE_NAMES}
    if observed["versions"] != expected_versions:
        fail(f"dependency version mismatch: expected {expected_versions}, got {observed['versions']}")
    return observed


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


def parse_run_log(text: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for name, pattern in LOG_PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            fail(f"reproduction output lacks {name}")
        parsed[name] = int(match.group(1))
    return parsed


def run_acceptance(arm: str, observed: dict[str, int], exit_code: int, lock: dict[str, Any]) -> bool:
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
    lock_path = LOCK_PATHS.get(summary.get("protocol_id"))
    if lock_path is None:
        fail(f"unknown protocol id: {summary.get('protocol_id')}")
    lock = read_json(lock_path)
    if summary.get("protocol_id") != lock["protocol_id"]:
        fail("protocol id mismatch")
    if summary.get("protocol_sha256") != sha256_file(lock_path):
        fail("protocol hash mismatch")
    validate_local_inputs(lock)
    if summary.get("run_count") != lock["reproducer"]["repetitions_per_arm"] * 2:
        fail("run count mismatch")
    for run in summary["runs"]:
        parsed = parse_run_log((bundle / run["log"]).read_text(encoding="utf-8"))
        if parsed != run["observed"]:
            fail(f"summary/log mismatch: {run['log']}")
        accepted = run_acceptance(run["arm"], parsed, run["exit_code"], lock)
        if accepted != run["accepted"]:
            fail(f"acceptance/log mismatch: {run['log']}")
    if summary["all_checks_passed"] != all(run["accepted"] for run in summary["runs"]):
        fail("overall decision does not match run decisions")
    return summary


def reproduce(args: argparse.Namespace) -> int:
    lock = read_json(LOCK_PATH)
    validate_local_inputs(lock)
    base = args.baseline_source.expanduser().resolve()
    candidate = args.candidate_source.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        fail(f"refusing to overwrite nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

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
                log_rel = f"{arm}-{repetition}.log"
                log_path = out / log_rel
                result = subprocess.run(
                    [args.python, "-m", "pytest", str(scripts[arm]), "-q", "-s"],
                    cwd=source_root,
                    env=test_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                log_path.write_text(result.stdout, encoding="utf-8")
                observed = parse_run_log(log_path.read_text(encoding="utf-8"))
                accepted = run_acceptance(arm, observed, result.returncode, lock)
                runs.append(
                    {
                        "arm": arm,
                        "repetition": repetition,
                        "exit_code": result.returncode,
                        "log": log_rel,
                        "observed": observed,
                        "accepted": accepted,
                    }
                )
                print(f"{arm}-{repetition}: exit={result.returncode}, accepted={accepted}, {observed}", flush=True)

    summary = {
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": sha256_file(LOCK_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "python_executable": str(Path(args.python).resolve()),
        "environment": env,
        "source_revisions": revisions,
        "gist_source_sha256": hashlib.sha256(gist_bytes).hexdigest(),
        "repetitions_per_arm": lock["reproducer"]["repetitions_per_arm"],
        "run_count": len(runs),
        "runs": runs,
        "all_checks_passed": all(run["accepted"] for run in runs),
    }
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
    args = parser.parse_args()
    try:
        return reproduce(args)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
        print(f"reproduction failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
