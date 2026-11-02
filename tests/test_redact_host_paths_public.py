"""Tests for scripts/redact_host_paths_public.py (synthetic fixtures plus the committed copies)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "redact_host_paths_public", ROOT / "scripts" / "redact_host_paths_public.py")
assert _spec and _spec.loader
red = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = red
_spec.loader.exec_module(red)

SESSION = "/Users/someone/Documents/App/2026-01-02/some-task-label"
TMPDIR = "/private/var/folders/ab/cdef0123/T"


def test_rules_are_ordered_deterministic_and_idempotent() -> None:
    text = json.dumps({
        "a": f"{SESSION}/work/VARE/tests/test_x.py",
        "b": "/Users/someone/.cache/huggingface/hub/models--m/snapshots/0",
        "c": "/Users/someone/.cache/tool/mirror",
        "d": f"{TMPDIR}/pytest-of-someone/pytest-1/x",
        "e": "/var/folders/ab/cdef0123/T/tmpq",
        "f": "/private/tmp/scratch-checkout/x",
        "g": "/opt/homebrew/bin/python3.11",
        "h": "/workspace/scratch/0a1b/VARE/tests",
        "i": "/mnt/vol/VARE/artifacts/x",
        "j": "/workspace/box1/lp-cache/datasets/f.parquet",
        "k": "/Users/someone",
    })
    out, counts = red.redact_text(text)
    assert out == red.redact_text(out)[0]
    assert json.loads(out) == {
        "a": "<SESSION_ROOT>/work/VARE/tests/test_x.py",
        "b": "<HF_HUB_CACHE>/models--m/snapshots/0",
        "c": "<HOME>/.cache/tool/mirror",
        "d": "<TMPDIR>/pytest-of-someone/pytest-1/x",
        "e": "<TMPDIR>/tmpq",
        "f": "<TMP>/scratch-checkout/x",
        "g": "<HOMEBREW_PREFIX>/bin/python3.11",
        "h": "<SESSION_ROOT>/VARE/tests",
        "i": "<REPO_ROOT>/artifacts/x",
        "j": "<DATA_CACHE>/datasets/f.parquet",
        "k": "<HOME>",
    }
    assert counts == {"<SESSION_ROOT>": 2, "<HF_HUB_CACHE>": 1, "<HOME>": 2, "<TMPDIR>": 2, "<TMP>": 1,
                      "<HOMEBREW_PREFIX>": 1, "<REPO_ROOT>": 1, "<DATA_CACHE>": 1}
    assert red.host_path_hits(out) == []


def test_bounded_inside_reprs_markup_and_truncation() -> None:
    cases = {
        f"args=['{SESSION}/work/venv/bin/python', '-c']": "args=['<SESSION_ROOT>/work/venv/bin/python', '-c']",
        f"\x1b[1m{SESSION}/x\x1b[0m": "\x1b[1m<SESSION_ROOT>/x\x1b[0m",
        "args: ['/Users/someone/Documents/App/2026-01-02/some-ta...>": "args: ['<SESSION_ROOT>...>",
        "&lt;Popen: ['/Users/someone/Documents/App/2026-01-02/so...&gt;": "&lt;Popen: ['<SESSION_ROOT>...&gt;",
        "'/Users/someone/Documents/App/2026-01-...s_opt1'": "'<SESSION_ROOT>...s_opt1'",
        "at /var/folders/ab/cdef01...c/vare/engine.py:86>": "at <TMPDIR>...c/vare/engine.py:86>",
        "unix:///Users/someone/.vm/default/sock": "unix://<HOME>/.vm/default/sock",
    }
    for src, want in cases.items():
        out = red.redact_text(src)[0]
        assert out == want, src
        assert red.host_path_hits(out) == [], src


def test_non_host_and_relative_forms_are_kept_and_not_flagged() -> None:
    kept = ("/home/runner/work/VARE/VARE/tests/t.py", "file:///Users/runner/miniforge3/conda-bld/x/work",
            "file:///home/conda/feedstock_root/build_artifacts/x", "python /workspace/solve.py",
            "PYTHONPATH=/workspace/src", "reviewer /root/some_audit", "/home/user/src/solution.py",
            "under /private/var/folders, which", "`/private/var/folders/...` temporary",
            "task/workspace/public", "../work/private/x.parquet", "<TMPDIR>/x")
    for s in kept:
        assert red.redact_text(s) == (s, {}), s
        assert red.host_path_hits(s) == [], s


def test_unknown_host_paths_are_detected_not_silently_kept() -> None:
    for s in ("/home/someone/x", "/Volumes/disk/x", "/workspace/box1/checkout/x", "/root/.cache/x",
              "/private/etc/hosts", "/mnt/vol/other", "C:\\Users\\a\\x", "C:\\\\Users\\\\a",
              "/private/var/folders/ab"):
        out, _ = red.redact_text(s)
        assert red.host_path_hits(out), s


def _fixture(tmp: Path) -> dict[str, str]:
    files = {
        "results/r1/run.json": json.dumps({"p": f"{SESSION}/work/VARE", "n": [1, 2]}, indent=2),
        "results/r1/rows.jsonl": json.dumps({"p": f"{TMPDIR}/x"}) + "\n" + json.dumps({"n": 1}) + "\n",
        "results/r1/stdout.json": json.dumps({"ok": True}) + "\nmanifest: /private/tmp/out-1\n",
        "results/r2/train.log": f"\x1b[1m{SESSION}/work/x\x1b[0m\nrootdir: /workspace/scratch/ab12/VARE\n",
        "docs/note.md": "Run `/opt/homebrew/bin/python3.11 -m pytest`.\n",
        "results/ci/job.log": "Working directory is '/home/runner/work/VARE/VARE'\n",
        "scripts/tool.py": 'PY = "/opt/homebrew/bin/python3.11"\n',
        "results/plain.txt": "no paths here\n",
    }
    for rel, text in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    readme = tmp / red.README
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(f"# copies\n\n{red.TABLE_BEGIN}\n{red.TABLE_END}\n", encoding="utf-8")
    return files


def test_write_then_check_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    assert red.write(tmp_path) == []
    assert red.check(tmp_path) == []
    mapping = json.loads((tmp_path / red.MAPPING).read_text())
    copied = {f["original"] for f in mapping["files"]}
    assert copied == {"results/r1/run.json", "results/r1/rows.jsonl", "results/r1/stdout.json",
                      "results/r2/train.log", "docs/note.md"}
    reasons = {e["path"]: e["reason"] for e in mapping["excluded"]}
    assert set(reasons) == {"results/ci/job.log", "scripts/tool.py"}
    assert reasons["results/ci/job.log"].startswith("only non-host path forms")
    assert reasons["scripts/tool.py"].startswith("executable source")
    for f in mapping["files"]:
        orig = (tmp_path / f["original"]).read_bytes()
        assert hashlib.sha256(orig).hexdigest() == f["original_sha256"]
        assert orig.decode() == files[f["original"]]  # originals untouched
        assert red.host_path_hits((tmp_path / f["redacted"]).read_text()) == []
    readme = (tmp_path / red.README).read_text()
    assert "`results/r1/run.json`" in readme and "`scripts/tool.py`" in readme
    # a changed original, a tampered copy, a new host-path file, a stray file and a stale table
    (tmp_path / "results/r1/run.json").write_text(files["results/r1/run.json"].replace("VARE", "VARE-2"))
    copy = tmp_path / red.DEST / "results/r1/rows.jsonl"
    copy.write_text(copy.read_text().replace("<TMPDIR>", "/Users/someone/x"))
    (tmp_path / "results/r3/extra.json").parent.mkdir(parents=True)
    (tmp_path / "results/r3/extra.json").write_text(json.dumps({"p": "/Users/someone/x"}))
    (tmp_path / red.DEST / "stray.txt").write_text("x")
    problems = red.check(tmp_path)
    assert any("results/r1/run.json: original sha256 differs" in p for p in problems)
    assert any("rows.jsonl: committed copy differs" in p for p in problems)
    assert any("host path present" in p for p in problems)
    assert any("results/r3/extra.json: contains a host path but has no redacted copy" in p for p in problems)
    assert any("stray.txt: file in" in p for p in problems)
    assert any("table is missing or stale" in p for p in problems)
    # --write repairs everything it owns and removes the stray file
    assert red.write(tmp_path) == []
    assert red.check(tmp_path) == []
    assert not (tmp_path / red.DEST / "stray.txt").exists()


def test_write_refuses_when_a_host_path_would_remain(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "results/r4.log").write_text(f"{SESSION}/x and /Volumes/disk/y\n")
    problems = red.write(tmp_path)
    assert any("results/r4.log: host path remains after redaction" in p for p in problems)
    assert not (tmp_path / red.MAPPING).exists()


def test_json_structure_is_preserved_or_rejected(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "results/r5.json").write_text('{"' + SESSION + '/k": "' + SESSION + '/v"}')
    mapping, copies, problems = red.build(tmp_path)
    assert problems == []
    assert json.loads(copies[red.DEST / "results/r5.json"]) == {"<SESSION_ROOT>/k": "<SESSION_ROOT>/v"}


def test_oversized_copy_is_excluded_not_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(red, "MAX_COPY_BYTES", 40)
    mapping, copies, _ = red.build(tmp_path)
    big = {e["path"]: e["reason"] for e in mapping["excluded"]}
    assert "tracked-blob budget" in big["results/r1/run.json"]
    assert red.DEST / "results/r1/run.json" not in copies


def test_pending_review_paths_are_excluded_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(red, "PENDING_REVIEW", frozenset({"results/r1/run.json"}))
    mapping, copies, problems = red.build(tmp_path)
    assert not problems
    reasons = {e["path"]: e["reason"] for e in mapping["excluded"]}
    assert reasons["results/r1/run.json"] == red.PENDING_REVIEW_REASON
    assert red.DEST / "results/r1/run.json" not in copies
    assert all(f["original"] != "results/r1/run.json" for f in mapping["files"])


def test_sibling_tree_files_are_not_copied_twice(tmp_path: Path) -> None:
    _fixture(tmp_path)
    rel = "results/learning-pipeline-v1/run-001/run_meta.json"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text(json.dumps({"p": "/Users/someone/checkout/protocols/p.json"}))
    sib = tmp_path / red.SIBLING_MAPPING
    sib.parent.mkdir(parents=True)
    digest = hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
    sib.write_text(json.dumps({"files": [{"original": rel, "original_sha256": digest}],
                               "rule": {"verification_pattern": "/Users/"}}))
    mapping, copies, problems = red.build(tmp_path)
    assert problems == []
    assert red.DEST / rel not in copies
    assert any(e["path"] == rel and "public-redacted" in e["reason"] for e in mapping["excluded"])
    # the original changed, so it no longer matches the sibling mapping
    (tmp_path / rel).write_text(json.dumps({"p": "/Users/someone/elsewhere"}))
    assert any("sha256 differs" in p for p in red.build(tmp_path)[2])


def test_committed_copies_verify() -> None:
    assert red.check(ROOT) == []


def test_junit_hostname_attributes_are_replaced() -> None:
    xml = ('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
           '<testsuite name="pytest" errors="0" tests="1" hostname="build-box-07.example">'
           f'<testcase classname="t" name="a" file="{SESSION}/t.py"/></testsuite>'
           "<testsuite name='other' hostname='ci-node.internal'></testsuite></testsuites>")
    out, counts = red.redact_text(xml)
    assert "build-box-07.example" not in out and "ci-node.internal" not in out
    assert 'hostname="redacted-host"' in out and "hostname='redacted-host'" in out
    assert counts == {"<SESSION_ROOT>": 1, "redacted-host": 2}
    assert red.redact_text(out) == (out, {})
    assert red.junit_hostname_hits(xml) == ["build-box-07.example", "ci-node.internal"]
    assert red.junit_hostname_hits(out) == []


def test_hostname_outside_junit_testsuite_is_kept() -> None:
    text = '<testcase name="x" hostname="kept"/> {"hostname": "kept"} hostname="kept"'
    assert red.redact_text(text) == (text, {})
