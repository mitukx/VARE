"""Long-horizon harness wired to the v2 isolated grader (PR #170), on a toy repository.

Skipped when the v2 grader modules are absent; live cases need bubblewrap (Linux).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import long_horizon_eval as lh

tre = pytest.importorskip("scripts.learning_pipeline_trusted_repo_env")
ig = pytest.importorskip("scripts.learning_pipeline_isolated_grading")
from scripts import long_horizon_trusted_grader as ltg  # noqa: E402
from vare.learning_pipeline.repo_env import RepoEnv  # noqa: E402
from vare.trajectory_store import PolicyCheckpoint  # noqa: E402

BUGGY = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
REFERENCE = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
TESTS = ("from pkg.core import add, mul\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
         "def test_mul():\n    assert mul(2, 3) == 6\n")
NAME = "Org__pkg.0000000"


def _live() -> bool:
    return sys.platform.startswith("linux") and ig.BwrapBackend(venv=Path(sys.prefix)).available()


needs_bwrap = pytest.mark.skipif(not _live(), reason="bubblewrap user namespaces unavailable")


def make_task(cache: Path) -> tuple[RepoEnv, dict]:
    path = cache / "repos" / NAME
    (path / "pkg").mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "pkg" / "__init__.py").write_text("")
    (path / "pkg" / "core.py").write_text(REFERENCE)
    (path / "tests" / "test_core.py").write_text(TESTS)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@invalid"]
    subprocess.run([*git, "init", "-q"], cwd=path, check=True)
    subprocess.run([*git, "add", "-A"], cwd=path, check=True)
    subprocess.run([*git, "commit", "-qm", "reference"], cwd=path, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True,
                            check=True).stdout.strip()
    (path / "pkg" / "core.py").write_text(BUGGY)
    patch = subprocess.run(["git", "diff"], cwd=path, capture_output=True, text=True, check=True).stdout
    subprocess.run(["git", "checkout", "-q", "-f", commit], cwd=path, check=True)
    record: dict[str, Any] = {"instance_id": "Org__pkg.0000000.toy_add", "upstream_repo": "Org/pkg", "mirror": f"mirror/{NAME}",
              "mirror_commit": commit, "split": "train", "target_file": "pkg/core.py",
              "fail_to_pass": ["tests/test_core.py::test_add"], "pass_to_pass": ["tests/test_core.py::test_mul"],
              "_patch": patch, "_problem": "add(2, 3) returns -1 instead of 5."}
    return RepoEnv(f"mirror/{NAME}", "Org/pkg", commit, cache), record


def call(name: str, **arguments) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def script(new_text: str) -> tuple[str, ...]:
    return (call("list_files"), call("read_file", path="pkg/core.py"),
            call("replace_text", path="pkg/core.py", old_text="return a - b", new_text=new_text),
            call("finish", summary="done"))


def run(tmp_path: Path, new_text: str, factory) -> dict:
    env, record = make_task(tmp_path / "cache")
    task = lh.learning_pipeline_task(record, lh.readable_python_files(env.path, env.commit))
    policy = lh.ScriptedPolicy(script(new_text))
    arm = lh.PolicyArm("stub", policy, PolicyCheckpoint("stub", 0, policy.checkpoint_sha256()))
    grader = factory(record, env)
    h = lh.EpisodeHarness(tmp_path / "run", arm, lh.CONTROLLER_PRESETS["minimal"],
                          factories={task.task_id: lh.repo_workspace_factory(env.path, env.commit, record["_patch"])},
                          grader_for=lambda _t: grader)
    h.enroll([lh.EpisodeSpec(task, 0)])
    h.run()
    return lh.run_metrics(tmp_path / "run")["episodes"][0]


def bwrap_factory(tmp_path: Path, venv: Path | None = Path(sys.prefix)):
    return ltg.make_factory(backend="bwrap", cache=tmp_path / "cache", venv=venv,
                            evidence_dir=tmp_path / "admission", timeout=60)


@needs_bwrap
def test_live_honest_fix_is_a_trusted_success(tmp_path):
    episode = run(tmp_path, "return a + b", bwrap_factory(tmp_path))
    assert episode["trusted_grade"] is True and episode["final_success"] is True, episode["grade_reason"]
    admission = json.loads((tmp_path / "admission" / "Org__pkg.0000000.toy_add.json").read_text())
    assert admission["admitted"] is True and admission["reference"]["passed"] is True
    assert admission["bug"]["passed"] is False and admission["bug"]["executed"] is True


@needs_bwrap
def test_live_wrong_fix_is_a_trusted_failure(tmp_path):
    episode = run(tmp_path, "return a * b", bwrap_factory(tmp_path))
    assert episode["trusted_grade"] is True and episode["final_success"] is False
    assert episode["grade_reason"] == "tests_failed"


@needs_bwrap
def test_live_forged_exit_is_never_success(tmp_path):
    episode = run(tmp_path, "import os; os._exit(0)", bwrap_factory(tmp_path))
    assert episode["final_success"] is False


def test_no_backend_fails_closed_without_running_candidate(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("candidate code must not run")

    monkeypatch.setattr(tre.TrustedRepoEnv, "grade", forbidden)
    for factory in (bwrap_factory(tmp_path, venv=None),
                    ltg.make_factory(backend="docker", cache=tmp_path / "cache", images={},
                                     evidence_dir=tmp_path / "admission")):
        episode = run(tmp_path / str(id(factory)), "return a + b", factory)
        assert episode["final_success"] is False and episode["trusted_grade"] is False
        assert episode["grade_reason"] == "untrusted_grade_schema"  # refused grader, not the v2 schema


def test_unattested_backend_fails_closed(tmp_path):
    factory = ltg.make_factory(backend="bwrap", cache=tmp_path / "cache", venv=tmp_path / "missing-venv",
                               evidence_dir=tmp_path / "admission")
    episode = run(tmp_path, "return a + b", factory)
    assert episode["final_success"] is False and episode["trusted_grade"] is False
    admission = json.loads((tmp_path / "admission" / "Org__pkg.0000000.toy_add.json").read_text())
    assert admission["selection"][0]["ok"] is False


def test_not_admitted_task_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ltg, "_reference_text", lambda env, target: BUGGY)  # reference would fail
    factory = bwrap_factory(tmp_path)
    if not _live():
        pytest.skip("admission runs inside the sandbox")
    episode = run(tmp_path, "return a + b", factory)
    assert episode["final_success"] is False and episode["trusted_grade"] is False
    admission = json.loads((tmp_path / "admission" / "Org__pkg.0000000.toy_add.json").read_text())
    assert admission["admitted"] is False
