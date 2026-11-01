"""A scripted, synthetic end-to-end fixture for the coding-agent lifecycle.

This verifies tool orchestration, source-bound editing, a public smoke check,
snapshot capture, and separate JSON grading. The scripted model and local
subprocess runner are test doubles; this is not model-success or sandbox evidence.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from vare.container_isolation import DockerIsolationRunner, IsolationLimits, IsolationResult
from vare.separated_grading import SeparatedCandidateGrader
from vare.trusted_scoring import JSONIOCase
import scripts.run_local_agent as agent


PINNED_IMAGE = "python@sha256:" + "a" * 64


class _FixtureTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(str(message.get("content", "")) for message in messages)

    def encode(self, text, **_kwargs):
        return [0] * max(1, len(text) // 8)


class _FixtureModel:
    def parameters(self):
        return iter(())

    def to(self, _device):
        return self

    def eval(self):
        return self


class _LocalScoringRunner:
    """Test double that checks Docker argv but executes candidate code locally."""

    def __init__(self):
        self.argv_builder = DockerIsolationRunner()

    def candidate_argv(self, *args, **kwargs):
        return self.argv_builder.candidate_argv(*args, **kwargs)

    async def run_candidate_io(
        self, limits, *, allowed_workspace_root, workspace, command, stdin_data,
    ):
        mapped = [
            str(Path(workspace) / value.removeprefix("/workspace/"))
            if value.startswith("/workspace/") else value
            for value in command
        ]
        try:
            completed = subprocess.run(
                mapped, cwd=workspace, input=stdin_data, capture_output=True,
                timeout=limits.timeout_s, check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired as exc:
            return IsolationResult(124, "", str(exc))
        return IsolationResult(
            completed.returncode,
            completed.stdout.decode("utf-8", errors="replace"),
            completed.stderr.decode("utf-8", errors="replace"),
        )


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(workspace), *args], check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def test_scripted_agent_edit_is_snapshotted_and_independently_graded(tmp_path, monkeypatch):
    workspace_root = tmp_path / "candidate-workspaces"
    workspace = workspace_root / "trial"
    workspace.mkdir(parents=True)
    source = (
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({'value': request['n']}))\n"
    ) + "# " + "x" * 12_500 + "\n"
    (workspace / "solve.py").write_text(source, encoding="utf-8")
    public_test = (
        "import json, subprocess, sys\n"
        "from pathlib import Path\n"
        "run = subprocess.run(\n"
        "    [sys.executable, str(Path(__file__).with_name('solve.py'))],\n"
        "    input='{" + '"n":3' + "}\\n', text=True, capture_output=True, check=True,\n"
        ")\n"
        "assert json.loads(run.stdout) == {'value': 9}\n"
    )
    (workspace / "test_public.py").write_text(public_test, encoding="utf-8")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "fixture@example.invalid")
    _git(workspace, "config", "user.name", "VARE fixture")
    _git(workspace, "add", "solve.py", "test_public.py")
    _git(workspace, "commit", "--quiet", "-m", "synthetic baseline")
    base_revision = _git(workspace, "rev-parse", "HEAD")

    task_root = tmp_path / "task"
    task_root.mkdir()
    (task_root / "task.json").write_text(json.dumps({
        "task_id": "synthetic-source-edit-fixture-v1",
        "source": {"base_revision": base_revision},
        "prompt": "Square request.n in solve.py and preserve the JSON response shape.",
        "metadata": {
            "editable_files": ["solve.py", "test_public.py"],
            "split": "development",
        },
    }), encoding="utf-8")
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "model.fixture").write_text("scripted model test double", encoding="utf-8")

    fake_torch = SimpleNamespace(
        __version__="fixture",
        float32="fixture-float32",
        manual_seed=lambda _seed: None,
        use_deterministic_algorithms=lambda _enabled: None,
        set_num_threads=lambda _count: None,
        get_num_threads=lambda: 1,
        inference_mode=lambda: __import__("contextlib").nullcontext(),
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _FixtureModel()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _FixtureTokenizer()),
        __version__="fixture",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(agent, "torch", fake_torch, raising=False)
    generation = iter((
        ("list_files", {}),
        ("read_file", {"path": "solve.py"}),
        ("replace_from_read", None),
        ("check_syntax", {"path": "solve.py"}),
        ("run_public_tests", {}),
        ("finish", {"summary": "Square request.n."}),
    ))
    observed_read_results = []

    def scripted_generate(_backend, _model, _tokenizer, messages, _tools, *, limit, args):
        name, arguments = next(generation)
        if name == "replace_from_read":
            read_result = next(
                json.loads(message["content"])
                for message in reversed(messages)
                if message.get("role") == "tool" and message.get("name") == "read_file"
            )
            observed_read_results.append(read_result)
            arguments = {
                "path": "solve.py",
                "source_handle": read_result["source_handle"],
                "old_text": "request['n']",
                "new_text": "request['n'] ** 2",
            }
        raw = "<tool_call>" + json.dumps(
            {"name": name, "arguments": arguments}, separators=(",", ":"),
        ) + "</tool_call>"
        return raw, 1, 1, limit

    monkeypatch.setattr(agent, "generate_turn", scripted_generate)

    def local_public_check(candidate_workspace, _image, expected_hash):
        test_file = candidate_workspace / "test_public.py"
        if agent.sha256(test_file) != expected_hash:
            return {"passed": False, "error": "public test changed"}
        completed = subprocess.run(
            [sys.executable, str(test_file)], cwd=candidate_workspace,
            capture_output=True, text=True, check=False, timeout=15,
        )
        return {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "isolation": "fixture-local-subprocess-test-double",
        }

    monkeypatch.setattr(agent, "run_public_tests", local_public_check)
    record_path = tmp_path / "attempt.json"
    monkeypatch.setattr(sys, "argv", [
        "run_local_agent.py",
        "--workspace", str(workspace),
        "--model-path", str(model_dir),
        "--backend", "transformers_cpu",
        "--task-root", str(task_root),
        "--output", str(record_path),
        "--instruction", "Square request.n in solve.py.",
        "--seed", "17",
        "--max-turns", "8",
        "--max-total-new-tokens", "100",
        "--source-read-handles",
        "--edit-format", "unified_diff",
        "--require-split", "development",
        "--public-test-image", PINNED_IMAGE,
    ])
    assert agent.main() == 0

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["stop_reason"] == "agent_finished"
    assert record["edit_applied"] is True
    assert record["source_read_handles_enabled"] is True
    assert len(observed_read_results) == 1
    assert observed_read_results[0]["source_handle"].startswith("source-")
    assert "response_truncated" not in observed_read_results[0]
    assert observed_read_results[0]["next_cursor"] is not None
    assert len(observed_read_results[0]["content"]) > 10_000
    assert record["candidate_diff"]
    assert "request['n'] ** 2" in record["candidate_diff"]
    assert record["source_hashes_before"] != record["source_hashes_after"]
    assert record["public_test_result"]["passed"] is True
    assert record["public_test_result"]["isolation"] == "fixture-local-subprocess-test-double"

    grader = SeparatedCandidateGrader(
        runner=_LocalScoringRunner(),
        limits=IsolationLimits(PINNED_IMAGE, timeout_s=10, memory_mb=256, cpus=1),
        allowed_workspace_root=workspace_root,
        snapshot_root=tmp_path / "snapshots",
    )
    snapshot = grader.capture_snapshot(workspace, candidate_exited=True)
    assert snapshot.captured_after_exit is True
    assert snapshot.file_sha256["solve.py"]
    assert snapshot.file_sha256["test_public.py"]
    cases = (
        JSONIOCase("positive", {"n": 4}, {"value": 16}),
        JSONIOCase("negative", {"n": -3}, {"value": 9}),
    )
    report = asyncio.run(grader.grade_snapshot(
        snapshot,
        task_id="synthetic-source-edit-fixture-v1",
        policy_id="scripted-agent-fixture",
        policy_version=1,
        command=[sys.executable, "/workspace/solve.py"],
        cases=cases,
        total_timeout_s=30,
    ))
    assert report.passed is True
    assert report.evaluation_complete is True
    assert report.cleanup_confirmed is True
    assert all(result.passed for result in report.cases)
