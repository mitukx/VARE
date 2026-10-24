from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/benchmark_verifier_consistent_async_grpo.py"


def _snapshot(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _invoke_existing_output(directory: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-S",
            RUNNER,
            "--output",
            str(directory),
            "--model-dir",
            str(tmp_path / "missing-model"),
            "--rvl-source",
            str(tmp_path / "missing-rvl"),
            "--seed",
            "82106",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=10,
    )


def test_rejected_existing_successful_run_is_byte_for_byte_unchanged(tmp_path):
    output = tmp_path / "completed-run"
    output.mkdir()
    records = {
        "protocol.json": b'{"protocol":"frozen"}\n',
        "summary.json": b'{"status":"complete","seed":82106}\n',
        "progress.json": b'{"status":"complete","completed_arms":["sync"]}\n',
    }
    for name, payload in records.items():
        (output / name).write_bytes(payload)
    manifest = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(records.items())
    }
    (output / "SHA256.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    before = _snapshot(output)

    result = _invoke_existing_output(output, tmp_path)

    assert result.returncode != 0
    assert "output directory already exists" in result.stderr
    assert _snapshot(output) == before
    assert "failure.json" not in _snapshot(output)


def test_rejected_existing_failure_record_is_byte_for_byte_unchanged(tmp_path):
    output = tmp_path / "failed-run"
    output.mkdir()
    failure = b'{"type":"OriginalFailure","message":"retain this record"}\n'
    progress = b'{"status":"failed","seed":82106}\n'
    (output / "failure.json").write_bytes(failure)
    (output / "progress.json").write_bytes(progress)
    before = _snapshot(output)

    result = _invoke_existing_output(output, tmp_path)

    assert result.returncode != 0
    assert "output directory already exists" in result.stderr
    assert _snapshot(output) == before
    assert (output / "failure.json").read_bytes() == failure

