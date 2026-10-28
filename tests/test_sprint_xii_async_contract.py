"""Cheap falsifiers for the frozen synthetic transition, before model execution."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "xii", Path(__file__).parents[1] / "scripts/sprint_xii/async_verifier.py"
)
xii = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xii)


def test_verifier_transition_is_nonconstant_and_opposite():
    assert [xii.reward(x, 0) for x in "ABCD"] == [1.0, 1.0, 0.0, 0.0]
    assert [xii.reward(x, 1) for x in "ABCD"] == [0.0, 0.0, 1.0, 1.0]


def test_durable_chain_binds_events(tmp_path):
    path = tmp_path / "journal.jsonl"
    xii.journal(path, {"kind": "prepare", "group": 1})
    xii.journal(path, {"kind": "commit", "group": 1})
    import json

    a, b = [json.loads(s) for s in path.read_text().splitlines()]
    assert b["previous"] == a["sha"]
    for row in [a, b]:
        digest = row.pop("sha")
        assert xii.sha(xii.canonical(row)) == digest


def test_atomic_writer_preserves_complete_artifact(tmp_path):
    path = tmp_path / "dir" / "artifact"
    xii.write(path, b"first")
    xii.write(path, b"second")
    assert path.read_bytes() == b"second"
    assert not list(path.parent.glob("*.tmp-*"))


def test_existing_partial_run_is_not_modified_by_cli(tmp_path):
    import subprocess
    import sys

    folder = tmp_path / "prior-run"
    folder.mkdir()
    original = b'{"failure":"historical"}\n'
    (folder / "failure.json").write_bytes(original)
    script = Path(__file__).parents[1] / "scripts/sprint_xii/async_verifier.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--protocol",
            str(tmp_path / "missing.json"),
            "--output",
            str(folder),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "append-only output already exists" in proc.stderr
    assert (folder / "failure.json").read_bytes() == original
    assert sorted(p.name for p in folder.iterdir()) == ["failure.json"]
