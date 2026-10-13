"""CPU contract checks for Docker role separation; no daemon or container claims."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pytest

from vare.attestation import EvidenceTrustRoot
from vare.checkpoint_artifacts import CheckpointIntegrityError, TrustedCheckpointStore
from vare.container_isolation import (
    DockerIsolationRunner,
    DockerWorkspaceAgent,
    IsolationError,
    IsolationLimits,
    IsolationResult,
)
from vare.types import EvaluationReport


IMAGE = "local/vare-worker@sha256:" + "a" * 64
KEY = bytes(range(32))


def _trust():
    return EvidenceTrustRoot(
        key=KEY,
        protocol_sha256=hashlib.sha256(b"locked protocol").hexdigest(),
        evaluator_sha256=hashlib.sha256(b"pinned worker image source").hexdigest(),
        run_id="isolated-worker-run-01",
        evaluation_set_sha256=hashlib.sha256(b"secret test data").hexdigest(),
        promotion_nonce="intent-nonce-00000001",
        strict_checkpoint_binding=True,
    )


def test_pinned_image_and_bounded_configuration_required():
    with pytest.raises(ValueError, match="pinned"):
        IsolationLimits("python:latest")
    with pytest.raises(ValueError, match="resource"):
        IsolationLimits(IMAGE, timeout_s=0)
    with pytest.raises(ValueError, match="resource"):
        IsolationLimits(IMAGE, memory_mb=0)
    with pytest.raises(ValueError, match="resource"):
        IsolationLimits(IMAGE, uid=0)


def test_agent_only_mounts_its_workspace_with_no_network_or_privileges(tmp_path):
    root = tmp_path / "workspaces"
    ws = root / "trial"
    ws.mkdir(parents=True)
    argv = DockerIsolationRunner().candidate_argv(
        IsolationLimits(IMAGE), name="vare-agent-" + "a" * 24,
        allowed_workspace_root=root, workspace=ws,
        command=("python", "/app/agent.py", "--input", "x"),
    )
    joined = " ".join(argv)
    assert "--pull=never" in argv
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    assert "--pids-limit" in argv and "--memory" in argv
    mounts = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--mount"]
    assert mounts == [f"type=bind,source={ws},target=/workspace"]
    assert "/operator/key" not in joined
    assert "/operator/heldout" not in joined
    with pytest.raises(IsolationError, match="escapes"):
        DockerIsolationRunner().candidate_argv(
            IsolationLimits(IMAGE), name="vare-agent-" + "a" * 24,
            allowed_workspace_root=root, workspace=tmp_path,
            command=("true",),
        )
    alias = root / "alias"
    alias.symlink_to(ws, target_is_directory=True)
    with pytest.raises(IsolationError, match="non-symlink"):
        DockerIsolationRunner().candidate_argv(
            IsolationLimits(IMAGE), name="vare-agent-" + "a" * 24,
            allowed_workspace_root=root, workspace=alias,
            command=("true",),
        )


def test_evaluator_only_sees_read_only_trusted_inputs(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    heldout = tmp_path / "heldout.json"
    heldout.write_text("[]")
    key = tmp_path / "signing.key"
    key.write_bytes(KEY)
    args = DockerIsolationRunner().evaluator_argv(
        IsolationLimits(IMAGE), name="vare-grader-" + "b" * 24,
        policy_id="policy-1", checkpoint_store=root,
        heldout_set=heldout, key_file=key, trust=_trust(),
    )
    mounts = [args[i + 1] for i, value in enumerate(args[:-1]) if value == "--mount"]
    assert mounts == [
        f"type=bind,source={root},target=/operator/checkpoints,readonly",
        f"type=bind,source={heldout},target=/operator/heldout.json,readonly",
        f"type=bind,source={key},target=/operator/key,readonly",
    ]
    assert "--network=none" in args and "--read-only" in args
    assert "--heldout-sha256" in args
    assert _trust().evaluation_set_sha256 in args
    assert KEY.hex() not in " ".join(args)
    # Never accept a secret or dataset nested in the checkpoint store.
    nested = root / "key"
    nested.write_bytes(KEY)
    with pytest.raises(IsolationError, match="separate"):
        DockerIsolationRunner().evaluator_argv(
            IsolationLimits(IMAGE), name="vare-grader-" + "c" * 24,
            policy_id="p1", checkpoint_store=root,
            heldout_set=heldout, key_file=nested, trust=_trust(),
        )


def test_read_only_checkpoint_store_queries_without_mutation(tmp_path):
    store = TrustedCheckpointStore(tmp_path / "operator-checkpoints")
    source = tmp_path / "weights.bin"
    source.write_bytes(b"independently saved snapshot")
    expected = store.register_file("p0", source)
    ro = TrustedCheckpointStore(store.root, read_only=True)
    assert ro.digest("p0") == expected
    with pytest.raises(CheckpointIntegrityError, match="read-only"):
        ro.register_file("p1", source)
    with pytest.raises(CheckpointIntegrityError, match="object directory is missing"):
        TrustedCheckpointStore(tmp_path / "missing", read_only=True)


def test_campaign_agent_placeholder_expansion_is_once_only(tmp_path):
    ws = tmp_path / "workspaces" / "trial"
    ws.mkdir(parents=True)

    class FakeRunner:
        def __init__(self):
            self.cmd = []

        async def run_candidate(self, limits, *, allowed_workspace_root, workspace, command):
            self.cmd = list(command)
            return IsolationResult(0, "ok", "")

    runner = FakeRunner()
    agent = DockerWorkspaceAgent(
        limits=IsolationLimits(IMAGE),
        allowed_workspace_root=ws.parent,
        command=("python", "/app/main.py", "{prompt}", "{workspace}", "{step}"),
        runner=runner,
    )
    result = asyncio.run(agent.run(prompt="literal {step} {workspace}", workspace=ws, step=7))
    assert json.loads(result)["returncode"] == 0
    assert runner.cmd == ["python", "/app/main.py", "literal {step} {workspace}", "/workspace", "7"]


def test_evaluator_response_must_have_valid_signature(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    heldout = tmp_path / "set.json"
    heldout.write_text("{}")
    key = tmp_path / "key"
    key.write_bytes(KEY)
    trust = _trust()
    signed = trust.attest(EvaluationReport("p1", 1.0, n=2), checkpoint_sha256="c" * 64)

    class FakeRunner(DockerIsolationRunner):
        async def execute(self, argv, limits):
            return IsolationResult(0, json.dumps(asdict(signed)), "")

    runner = FakeRunner()
    options = dict(
        policy_id="p1", checkpoint_store=root, heldout_set=heldout,
        key_file=key, trust=trust,
    )
    assert asyncio.run(runner.run_evaluator(IsolationLimits(IMAGE), **options)).primary == 1.0
    signed.primary = 0.0
    with pytest.raises(IsolationError, match="unsigned"):
        asyncio.run(runner.run_evaluator(IsolationLimits(IMAGE), **options))


def _fake_docker(tmp_path: Path, behavior: str) -> tuple[str, Path]:
    binary = tmp_path / "fake-docker"
    audit = tmp_path / "cleanup.txt"
    script = (
        "#!" + sys.executable + "\n"
        "import sys,time\n"
        "from pathlib import Path\n"
        f"audit = Path({str(audit)!r})\n"
        "if sys.argv[1] == 'rm':\n"
        "    audit.write_text('removed')\n"
        "    print('No such container', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "if sys.argv[1] == 'run':\n"
        + (("    time.sleep(10)\n") if behavior == "timeout" else
           ("    print('x'*20000)\n") if behavior == "overflow" else
           ("    print('ok')\n"))
    )
    binary.write_text(script, encoding="utf-8")
    binary.chmod(0o755)
    return str(binary), audit


@pytest.mark.parametrize("behavior", ("ok", "timeout", "overflow"))
def test_supervisor_always_requests_forced_container_cleanup(tmp_path, behavior):
    binary, audit = _fake_docker(tmp_path, behavior)
    root = tmp_path / "ws"
    work = root / "w"
    work.mkdir(parents=True)
    limits = IsolationLimits(
        IMAGE, timeout_s=0.3 if behavior == "timeout" else 6,
        max_output_bytes=256 if behavior == "overflow" else 100_000,
    )
    runner = DockerIsolationRunner(docker_binary=binary)
    if behavior == "ok":
        result = asyncio.run(runner.run_candidate(
            limits, allowed_workspace_root=root, workspace=work,
            command=("echo", "hello"),
        ))
        assert result.stdout.strip() == "ok"
    else:
        with pytest.raises(IsolationError, match="limit"):
            asyncio.run(runner.run_candidate(
                limits, allowed_workspace_root=root, workspace=work,
                command=("echo", "hello"),
            ))
    assert audit.read_text(encoding="utf-8") == "removed"


def test_controller_cancellation_also_requests_forced_cleanup(tmp_path):
    binary, audit = _fake_docker(tmp_path, "timeout")
    root = tmp_path / "workspaces"
    work = root / "trial"
    work.mkdir(parents=True)
    runner = DockerIsolationRunner(docker_binary=binary)
    limits = IsolationLimits(IMAGE, timeout_s=8)

    async def scenario():
        task = asyncio.create_task(runner.run_candidate(
            limits, allowed_workspace_root=root,
            workspace=work, command=("echo", "hello"),
        ))
        await asyncio.sleep(0.25)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert audit.read_text(encoding="utf-8") == "removed"


def test_isolated_hooks_never_call_learner_for_promotion_evidence(tmp_path):
    from vare.isolated_hooks import IsolatedEvaluationHooks

    root = tmp_path / "store"
    root.mkdir()
    heldout = tmp_path / "heldout"
    heldout.write_bytes(b"[]")
    key = tmp_path / "key"
    key.write_bytes(KEY)
    signer = _trust()
    signed = signer.attest(EvaluationReport("p1", 0.9, n=2), checkpoint_sha256="f" * 64)

    class Learner:
        def active_policy(self):
            return ("p0", 0)

        async def rollout(self, task, policy_id, policy_version, step):
            raise NotImplementedError

        async def train_candidate(self, incumbent_id, experiences):
            return "p1"

        async def evaluate(self, policy_id):
            raise AssertionError("learner must not grade promotion")

        async def promote(self, candidate_id):
            return None

        async def discard(self, candidate_id):
            return None

    class FakeRunner(DockerIsolationRunner):
        async def run_evaluator(self, limits, *, policy_id, **kwargs):
            assert policy_id == "p1"
            return signed

    hooks = IsolatedEvaluationHooks(
        underlying=Learner(),
        runner=FakeRunner(),
        evaluator_limits=IsolationLimits(IMAGE),
        trust=signer,
        checkpoint_store=root,
        heldout_set=heldout,
        key_file=key,
    )
    assert hooks.active_policy() == ("p0", 0)
    assert asyncio.run(hooks.evaluate("p1")) is signed
