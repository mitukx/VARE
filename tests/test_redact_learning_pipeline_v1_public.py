"""Tests for scripts/redact_learning_pipeline_v1_public.py (synthetic fixtures plus the committed copies)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "redact_learning_pipeline_v1_public", ROOT / "scripts" / "redact_learning_pipeline_v1_public.py")
assert _spec and _spec.loader
red = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = red
_spec.loader.exec_module(red)

RUN = "/Users/someone/checkout-x/artifacts/learning_pipeline_v1/run-001"


def test_rules_are_ordered_and_deterministic() -> None:
    text = (f'{{"a": "{RUN}/adapters/rft", "b": "/Users/someone/checkout-x/protocols/p.json", '
            '"c": "/Users/someone/.cache/huggingface/hub/models--m/snapshots/0@local+' + RUN + '/x", '
            '"d": "/workspace/box1/lp-cache/datasets/f.parquet"}')
    out, counts = red.redact_text(text)
    assert out == red.redact_text(text)[0] == red.redact_text(out)[0]
    assert json.loads(out) == {"a": "<RUN_ROOT>/adapters/rft", "b": "<REPO_ROOT>/protocols/p.json",
                               "c": "<HF_HUB_CACHE>/models--m/snapshots/0@local+<RUN_ROOT>/x",
                               "d": "<DATA_CACHE>/datasets/f.parquet"}
    assert counts == {"<RUN_ROOT>": 2, "<REPO_ROOT>": 1, "<HF_HUB_CACHE>": 1, "<DATA_CACHE>": 1}
    assert red.host_path_hits(out) == []


def test_ansi_wrapped_log_paths_are_redacted() -> None:
    out, _ = red.redact_text(f"\x1b[1m{RUN}/data/rft_sel\x1b[0m  |")
    assert out == "\x1b[1m<RUN_ROOT>/data/rft_sel\x1b[0m  |"


def test_unknown_host_paths_are_detected_not_silently_kept() -> None:
    for s in ("/Users/a/elsewhere/file", "/home/a/x", "/private/var/folders/ab/T/x", "/workspace/a/b",
              "C:\\Users\\a\\x", "C:\\\\Users\\\\a"):
        out, _ = red.redact_text(s)
        assert red.host_path_hits(out), s
    assert red.host_path_hits("<RUN_ROOT>/data and results/learning-pipeline-v1/run-001/x") == []


def _fixture(tmp: Path) -> None:
    for rel in red.SOURCES:
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".jsonl"):
            p.write_text(json.dumps({"k": f"{RUN}/{p.name}"}) + "\n" + json.dumps({"k": 1}) + "\n")
        elif rel.endswith(".json"):
            p.write_text(json.dumps({"k": f"{RUN}/{p.name}", "n": [1, 2]}, indent=2))
        else:
            p.write_text(f"\x1b[1m{RUN}/{p.name}\x1b[0m\n")


def test_write_then_check_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    _fixture(tmp_path)
    assert red.write(tmp_path) == []
    assert red.check(tmp_path) == []
    mapping = json.loads((tmp_path / red.MAPPING).read_text())
    for f in mapping["files"]:
        orig = (tmp_path / f["original"]).read_bytes()
        assert hashlib.sha256(orig).hexdigest() == f["original_sha256"]
        assert RUN.encode() in orig  # originals untouched
    # a changed original, a tampered copy, and an unlisted file with a host path are all reported
    first = tmp_path / red.SOURCES[0]
    first.write_text(first.read_text().replace("[\n    1", "[\n    3"))
    copy = tmp_path / red.DEST / red.SOURCES[1]
    copy.write_text(copy.read_text().replace("<RUN_ROOT>", "/Users/someone/x"))
    extra = tmp_path / "results/learning-pipeline-v1/run-001/extra.json"
    extra.write_text(json.dumps({"p": "/home/someone/x"}))
    problems = red.check(tmp_path)
    assert any("MAPPING.json differs" in p for p in problems)
    assert any(red.SOURCES[1] in p and "differs" in p for p in problems)
    assert any("host path present" in p for p in problems)
    assert any("extra.json" in p and "no redacted copy" in p for p in problems)
    assert red.write(tmp_path) != []  # refuses to write while an unlisted host path exists


def test_invalid_json_after_redaction_is_rejected(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / red.SOURCES[0]).write_text('{"k": "x"')
    assert any("not valid JSON" in p for p in red.build(tmp_path)[2])


def test_committed_copies_verify() -> None:
    assert red.check(ROOT) == []
