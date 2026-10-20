import asyncio
import hashlib
import pytest

from vare.container_isolation import IsolationError, IsolationLimits, IsolationResult
from vare.trusted_scoring import JSONIOCase, TrustedJSONScorer


class StubCandidateRunner:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    async def run_candidate_io(
        self, limits, *, allowed_workspace_root, workspace, command, stdin_data,
    ):
        self.requests.append(stdin_data)
        return IsolationResult(0, next(self.outputs), "")


def test_trusted_scorer_sends_request_only_and_keeps_expected_outputs_operator_side(tmp_path):
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "one"
    workspace.mkdir(parents=True)
    runner = StubCandidateRunner(['{"value":4}', '{"value":9}'])
    scorer = TrustedJSONScorer(
        runner=runner,
        limits=IsolationLimits("local/test@sha256:" + "a" * 64),
        allowed_workspace_root=workspace_root,
    )
    report = asyncio.run(scorer.evaluate(
        task_id="json-double-v1",
        policy_id="candidate-a",
        policy_version=2,
        workspace=workspace,
        command=("python", "/workspace/solve.py"),
        cases=(
            JSONIOCase("case-a", {"n": 2}, {"value": 4}),
            JSONIOCase("case-b", {"n": 3}, {"value": 9}),
        ),
    ))

    assert report.passed is True
    assert report.score == 1.0
    assert [case.passed for case in report.cases] == [True, True]
    assert runner.requests == [b'{"n":2}\n', b'{"n":3}\n']
    assert all(b'"value"' not in request for request in runner.requests)


def test_candidate_forged_score_is_scored_as_an_ordinary_wrong_output(tmp_path):
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "one"
    workspace.mkdir(parents=True)
    runner = StubCandidateRunner(['{"passed":true,"score":1.0}'])
    scorer = TrustedJSONScorer(
        runner=runner,
        limits=IsolationLimits("local/test@sha256:" + "a" * 64),
        allowed_workspace_root=workspace_root,
    )
    report = asyncio.run(scorer.evaluate(
        task_id="json-double-v1",
        policy_id="candidate-a",
        policy_version=2,
        workspace=workspace,
        command=("python", "/workspace/solve.py"),
        cases=(JSONIOCase("case-a", {"n": 2}, {"value": 4}),),
    ))

    assert report.passed is False
    assert report.score == 0.0
    assert report.cases[0].failure == "wrong_json_response"
    assert len(report.cases[0].response_sha256) == 64


def test_ambiguous_duplicate_json_keys_are_rejected(tmp_path):
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "one"
    workspace.mkdir(parents=True)
    scorer = TrustedJSONScorer(
        runner=StubCandidateRunner(['{"value":0,"value":4}']),
        limits=IsolationLimits("local/test@sha256:" + "a" * 64),
        allowed_workspace_root=workspace_root,
    )
    report = asyncio.run(scorer.evaluate(
        task_id="json-double-v1", policy_id="candidate-a",
        policy_version=0, workspace=workspace,
        command=("python", "/workspace/solve.py"),
        cases=(JSONIOCase("case-a", {"n": 2}, {"value": 4}),),
    ))
    assert report.passed is False
    assert report.cases[0].failure == "invalid_json_response"


def test_scoring_evidence_uses_runner_raw_stdout_digest(tmp_path):
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "one"
    workspace.mkdir(parents=True)
    raw_digest = hashlib.sha256(b"\xff").hexdigest()

    class InvalidUtf8Runner:
        async def run_candidate_io(self, *args, **kwargs):
            return IsolationResult(0, "\ufffd", "", raw_digest)

    scorer = TrustedJSONScorer(
        runner=InvalidUtf8Runner(),
        limits=IsolationLimits("local/test@sha256:" + "a" * 64),
        allowed_workspace_root=workspace_root,
    )
    report = asyncio.run(scorer.evaluate(
        task_id="json-double-v1", policy_id="candidate-a",
        policy_version=0, workspace=workspace,
        command=("python", "/workspace/solve.py"),
        cases=(JSONIOCase("case-a", {"n": 2}, {"value": 4}),),
    ))
    assert report.cases[0].failure == "invalid_json_response"
    assert report.cases[0].response_sha256 == raw_digest


def test_json_scoring_rejects_nonfinite_case_values_and_duplicate_case_ids(tmp_path):
    try:
        JSONIOCase("bad", float("nan"), None)
    except ValueError as exc:
        assert "finite JSON" in str(exc)
    else:
        raise AssertionError("nonfinite input was accepted")

    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "one"
    workspace.mkdir(parents=True)
    scorer = TrustedJSONScorer(
        runner=StubCandidateRunner([]),
        limits=IsolationLimits("local/test@sha256:" + "a" * 64),
        allowed_workspace_root=workspace_root,
    )
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(scorer.evaluate(
            task_id="json-double-v1", policy_id="candidate-a",
            policy_version=0, workspace=workspace,
            command=("python", "/workspace/solve.py"),
            cases=(
                JSONIOCase("same", 1, 1),
                JSONIOCase("same", 2, 2),
            ),
        ))


def test_isolation_error_is_retained_as_incomplete_failed_evaluation(tmp_path):
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "one"
    workspace.mkdir(parents=True)

    class FailingRunner:
        async def run_candidate_io(self, *args, **kwargs):
            raise IsolationError("container exceeded wall-time limit")

    scorer = TrustedJSONScorer(
        runner=FailingRunner(),
        limits=IsolationLimits("local/test@sha256:" + "a" * 64),
        allowed_workspace_root=workspace_root,
    )
    report = asyncio.run(scorer.evaluate(
        task_id="json-double-v1", policy_id="candidate-a",
        policy_version=0, workspace=workspace,
        command=("python", "/workspace/solve.py"),
        cases=(JSONIOCase("case-a", 1, 1), JSONIOCase("case-b", 2, 4)),
    ))
    assert report.passed is False
    assert report.evaluation_complete is False
    assert report.cleanup_confirmed is True
    assert [case.failure for case in report.cases] == [
        "container exceeded wall-time limit", "not_run_after_isolation_error",
    ]


def test_uncertain_cleanup_is_never_a_passing_score(tmp_path):
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "one"
    workspace.mkdir(parents=True)

    class FailingRunner:
        async def run_candidate_io(self, *args, **kwargs):
            raise IsolationError("could not confirm Docker container cleanup")

    scorer = TrustedJSONScorer(
        runner=FailingRunner(),
        limits=IsolationLimits("local/test@sha256:" + "a" * 64),
        allowed_workspace_root=workspace_root,
    )
    report = asyncio.run(scorer.evaluate(
        task_id="json-double-v1", policy_id="candidate-a",
        policy_version=0, workspace=workspace,
        command=("python", "/workspace/solve.py"),
        cases=(JSONIOCase("case-a", 1, 1),),
    ))
    assert report.passed is False
    assert report.cleanup_confirmed is False
