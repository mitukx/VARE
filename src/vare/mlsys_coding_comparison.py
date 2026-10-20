"""Comparison runner for the frozen MLSys coding protocol.

The default command path still refuses an authorized run. This module adds
three things around that refusal:

- a canary-read check of the repository grader's mount plan;
- a train-split guard keyed by the frozen pack manifest hash;
- a synthetic CPU dry-run that calls the existing GRPO loss and the existing
  trajectory workers.

The synthetic policy is a labeled Bernoulli stand-in. It is not a language
model, and its scores are not a capability result. Real model patches are not
graded here.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from minigrpo.grpo import TokenRow, cpu_policy_step, group_advantages, sequence_loss

from vare.container_isolation import DockerIsolationRunner, IsolationLimits
from vare.environments.runner import ExecutableEvaluator, Workspace
from vare.environments.spec import EngineeringTaskSpec
from vare.evidence import sha256_file
from vare.sandboxed_repository import DockerRepositoryEvaluator
from vare.trajectory_admission import TrajectoryRewardRoot, admit_terminal_grpo_group
from vare.trajectory_store import AgentStep, PolicyCheckpoint, TrajectoryStore
from vare.trajectory_workers import execute_trajectories, owned_io, verify_trajectories
from vare.types import Task, Verification

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "benchmarks" / "mlsys_coding_v1"
PROTOCOL_PATH = REPO_ROOT / "protocols" / "mlsys_coding_policy_comparison_v1.json"
FROZEN_MANIFEST_SHA256 = (
    "2c67d85a6806b3bbcbc63676f64fc955c5c7b3c9c4127a5a3caade1541efb4ab"
)
SYNTHETIC_TASK_ID = "mlsys-v1-row-log-softmax"
PERFORMANCE_PROXY_TASK_ID = "mlsys-v1-range-sums"
GROUP_SIZE = 4
UPDATE_LR = 0.05
SYNTHETIC_MODEL = "synthetic-mock-policy"
BEHAVIOR = "neutral_temperature_scaled_policy"
# Analysis assumption fixed before the Monte Carlo below. Not a measured rate.
REALISTIC_BASELINE_P = 0.05
REALISTIC_ARM_P = 0.30
POWER_TASKS = 4
POWER_TRIALS = 20000
POWER_SEED = 20261019


class HeldOutAccessError(RuntimeError):
    """Training or tuning asked for a task outside the frozen train split."""


class PackIntegrityError(RuntimeError):
    """The pack bytes do not match the frozen manifest hash."""


class RealGradingRefused(RuntimeError):
    """A real model patch was not graded."""

    def __init__(self, decision: Mapping[str, Any]) -> None:
        self.decision = dict(decision)
        super().__init__(decision.get("reason", "real model grading refused"))


def _softplus(value: float) -> float:
    if value > 40.0:
        return value
    if value < -40.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def bernoulli_logp(theta: float, oracle_patch: bool) -> float:
    """Log probability of one synthetic action. ``theta`` favors the oracle patch."""
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    return -_softplus(-theta) if oracle_patch else -_softplus(theta)


def finite_difference(
    loss_at: Callable[[float], float], *, lr: float = UPDATE_LR
) -> dict[str, float]:
    """Same central-difference step as ``minigrpo.grpo.cpu_policy_step``."""
    before = loss_at(0.0)
    grad = (loss_at(1e-4) - loss_at(-1e-4)) / 2e-4
    if not math.isfinite(grad) or not math.isfinite(before):
        raise RuntimeError("non-finite synthetic gradient")
    theta = -lr * grad
    after = loss_at(theta)
    if not math.isfinite(after):
        raise RuntimeError("non-finite synthetic loss")
    return {
        "grad": grad,
        "learning_rate": lr,
        "loss_after": after,
        "loss_before": before,
        "theta": theta,
    }


def grpo_update(choices: Sequence[bool], rewards: Sequence[float]) -> dict[str, Any]:
    """One intact-group GRPO step on synthetic token rows.

    Advantages and the clipped token loss come from ``minigrpo``. The scalar
    step is the finite-difference update already used by ``cpu_policy_step``.
    """
    if len(choices) != GROUP_SIZE or len(rewards) != GROUP_SIZE:
        raise RuntimeError("refusing to update a truncated GRPO group")
    advantages = group_advantages(list(rewards), GROUP_SIZE)
    if len(advantages) != GROUP_SIZE:
        raise RuntimeError("group was truncated")

    def loss_at(theta: float) -> float:
        rows = []
        for choice, advantage in zip(choices, advantages, strict=True):
            rows.append(
                TokenRow(
                    logp=(bernoulli_logp(theta, choice),),
                    logp_old=(bernoulli_logp(0.0, choice),),
                    mask=(1,),
                    advantage=advantage,
                )
            )
        loss = sequence_loss(rows)["loss"]
        if not isinstance(loss, float):
            raise TypeError("GRPO loss was not a float")
        return loss

    step = finite_difference(loss_at)
    step["advantages"] = advantages
    step["group_size"] = GROUP_SIZE
    step["objective"] = "minigrpo.sequence_loss"
    step["reward_mean"] = sum(rewards) / len(rewards)
    return step


def sft_update() -> dict[str, Any]:
    """Bernoulli negative log-likelihood of the train-split oracle action.

    This is the synthetic stand-in for oracle-patch behavior cloning. It does
    not backpropagate through a language model.
    """

    def loss_at(theta: float) -> float:
        return -bernoulli_logp(theta, True)

    step = finite_difference(loss_at)
    step["objective"] = "bernoulli_nll_oracle_patch"
    step["gradient_updates"] = 1
    return step


def _load_reward():
    path = PACK_DIR / "reward.py"
    spec = importlib.util.spec_from_file_location("mlsys_coding_v1_reward", path)
    if spec is None or spec.loader is None:
        raise PackIntegrityError("reward module is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_text(root: Path) -> str:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    payload = {"files": files, "pack_id": "mlsys-coding-v1"}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FrozenPack:
    """Train/tune access limited to the hashed train split."""

    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = root
        self.pack = root / "benchmarks" / "mlsys_coding_v1"
        self.protocol_path = (
            root / "protocols" / "mlsys_coding_policy_comparison_v1.json"
        )
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        if self.protocol.get("protocol_id") != "mlsys-coding-policy-comparison-v1":
            raise PackIntegrityError("unexpected comparison protocol id")
        declared = self.protocol.get("task_pack_manifest_sha256")
        if declared != FROZEN_MANIFEST_SHA256:
            raise PackIntegrityError("protocol manifest hash is not the frozen hash")
        text = _manifest_text(self.pack)
        digest = _sha256_text(text)
        recorded = self.pack / "manifest.json"
        if (
            not recorded.is_file()
            or recorded.read_text(encoding="utf-8") != text
            or digest != FROZEN_MANIFEST_SHA256
        ):
            raise PackIntegrityError("pack manifest does not match the frozen hash")
        manifest = json.loads(text)
        split_entry = next(
            item for item in manifest["files"] if item["path"] == "split.json"
        )
        split_path = self.pack / "split.json"
        if sha256_file(split_path) != split_entry["sha256"]:
            raise PackIntegrityError(
                "split.json does not match the frozen manifest entry"
            )
        split = json.loads(split_path.read_text(encoding="utf-8"))
        self.train_ids = tuple(split["train"])
        self.heldout_ids = tuple(split["heldout"])
        self.split_sha256 = split_entry["sha256"]
        if (
            not self.train_ids
            or not self.heldout_ids
            or set(self.train_ids) & set(self.heldout_ids)
        ):
            raise PackIntegrityError("frozen split is empty or overlapping")
        self.reward = _load_reward()

    def reject_unless_train(self, task_id: str, *, role: str) -> None:
        if role not in {"train", "tune"}:
            raise HeldOutAccessError("only train and tune may read task files")
        if task_id in self.heldout_ids or task_id not in self.train_ids:
            raise HeldOutAccessError(f"{role} cannot read task {task_id}")

    def task_directory(self, task_id: str, *, role: str) -> Path:
        self.reject_unless_train(task_id, role=role)
        name = task_id.removeprefix("mlsys-v1-").replace("-", "_")
        directory = self.pack / "tasks" / name
        spec = EngineeringTaskSpec.load(directory / "task.json")
        if spec.id != task_id or spec.metadata.get("split") != "train":
            raise HeldOutAccessError(f"{role} refused task {task_id}")
        return directory

    def oracle_file(self, task_id: str, *, role: str) -> Path:
        directory = self.task_directory(task_id, role=role)
        matches = sorted((directory / "evaluator" / "oracle").glob("*.py"))
        if len(matches) != 1:
            raise PackIntegrityError("train task does not have one oracle patch")
        return matches[0]

    def grade_fixture(
        self, task_id: str, *, oracle_patch: bool, role: str = "train"
    ) -> dict[str, Any]:
        """Grade this repository's own baseline or oracle file. Not a model patch."""
        directory = self.task_directory(task_id, role=role)
        spec = EngineeringTaskSpec.load(directory / "task.json")
        with Workspace(spec) as workspace:
            if oracle_patch:
                oracle = self.oracle_file(task_id, role=role)
                shutil.copyfile(oracle, workspace.path / oracle.name)
            result = ExecutableEvaluator(spec, evaluator_root=directory).evaluate(
                workspace.path
            )
        decision = self.reward.decide(
            result.to_dict(), task_type=str(spec.metadata["task_type"])
        )
        decision["oracle_patch"] = oracle_patch
        decision["task_id"] = task_id
        decision["policy_class"] = SYNTHETIC_MODEL
        decision["passed"] = result.passed
        return decision


def _parse_bind_mounts(argv: Sequence[str]) -> list[dict[str, str]]:
    mounts = []
    for index, arg in enumerate(argv):
        if arg != "--mount" or index + 1 >= len(argv):
            continue
        fields: dict[str, str] = {}
        for item in argv[index + 1].split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        if fields.get("type") == "bind" and "source" in fields and "target" in fields:
            mounts.append(fields)
    return mounts


def _copy_mount(source: Path, dest: Path) -> None:
    if source.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        return
    shutil.copytree(source, dest, symlinks=False)


def namespace_reads_canary(
    mounts: Sequence[Mapping[str, str]], canary_token: str, work: Path
) -> dict[str, Any]:
    """Copy only the declared mounts into a jail and search it for the canary.

    The canary token is not passed to the child. The child prints file text
    from the jail; the parent looks for the token.
    """
    jail = work / "jail"
    jail.mkdir(parents=True)
    for mount in mounts:
        source = Path(mount["source"])
        target = str(mount["target"]).lstrip("/")
        if not target or target.startswith("..") or Path(target).is_absolute():
            raise ValueError("unsafe mount target")
        _copy_mount(source, jail / target)
    script = work / "read_namespace.py"
    script.write_text(
        "import os\nfrom pathlib import Path\n"
        "root = Path(os.environ['JAIL'])\n"
        "for path in root.rglob('*'):\n"
        "    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1000000:\n"
        "        continue\n"
        "    print(path.read_text(encoding='utf-8', errors='replace'))\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=jail,
        env={
            "JAIL": str(jail),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
        },
        timeout=30,
    )
    leaked = canary_token in completed.stdout or canary_token in completed.stderr
    return {
        "exited": completed.returncode is not None,
        "returncode": completed.returncode,
        "read_canary": leaked,
    }


def assess_execution_phases(
    phases: Sequence[Mapping[str, Any]],
    *,
    canary_token: str,
    work: Path,
) -> dict[str, Any]:
    """Prove confidentiality only for a candidate phase that cannot read the canary.

    The grader phase must run after that candidate process has exited, in a
    separate phase, and must itself be able to read the canary.
    """
    if not phases:
        return {
            "proven": False,
            "reason": "repository grader exposed no execution phases",
            "candidate_read_canary": None,
            "grader_read_canary": None,
            "separate_containers": False,
            "candidate_exited_before_grader": False,
        }
    roles = [str(phase.get("role")) for phase in phases]
    candidate_indexes = [
        index for index, role in enumerate(roles) if role == "candidate"
    ]
    grader_indexes = [index for index, role in enumerate(roles) if role == "grader"]
    separate = (
        bool(candidate_indexes) and bool(grader_indexes) and "combined" not in roles
    )
    order_ok = separate and max(candidate_indexes) < min(grader_indexes)
    candidate_read = False
    candidate_exited = True
    read_roles = candidate_indexes if separate else list(range(len(phases)))
    for index in read_roles:
        probe = namespace_reads_canary(
            phases[index]["mounts"], canary_token, work / f"phase-{index}"
        )
        candidate_read = candidate_read or bool(probe["read_canary"])
        candidate_exited = candidate_exited and bool(probe["exited"])
    grader_read = False
    if order_ok and candidate_exited:
        for index in grader_indexes:
            probe = namespace_reads_canary(
                phases[index]["mounts"], canary_token, work / f"grader-{index}"
            )
            grader_read = grader_read or bool(probe["read_canary"])
    proven = (not candidate_read) and grader_read and order_ok and candidate_exited
    if proven:
        reason = "candidate phase could not read the hidden-test canary; a later grader phase could"
    elif candidate_read:
        reason = "candidate namespace read the hidden-test canary"
    elif not separate:
        reason = "candidate and hidden tests are not in separate containers"
    elif not grader_read:
        reason = "grader phase did not read the canary, so the check is vacuous"
    else:
        reason = "hidden-test confidentiality was not proven"
    return {
        "proven": proven,
        "reason": reason,
        "candidate_read_canary": candidate_read,
        "grader_read_canary": grader_read if order_ok else False,
        "separate_containers": separate and order_ok,
        "candidate_exited_before_grader": order_ok and candidate_exited,
        "phase_roles": roles,
    }


def live_repository_grader_phases(
    spec_path: Path, workspace: Path, allowed_root: Path
) -> list[dict[str, Any]]:
    """Mount plan of the current repository grader. Does not start Docker.

    If a later grader exposes ``confidential_execution_phases``, that ordered
    plan is what the canary reads. Otherwise the single ``_argv`` container,
    which bind-mounts the task tree and the candidate together, is one
    combined phase.
    """
    spec_path = Path(spec_path).resolve()
    workspace = Path(workspace).resolve()
    allowed_root = Path(allowed_root).resolve()
    limits = IsolationLimits(
        image="vare-canary@sha256:" + ("ab" * 32),
        timeout_s=1.0,
        memory_mb=32,
        cpus=0.1,
        pids_limit=16,
        max_output_bytes=1000,
    )
    evaluator = DockerRepositoryEvaluator(
        runner=DockerIsolationRunner(docker_binary="docker"),
        limits=limits,
        allowed_workspace_root=allowed_root,
    )
    hook = getattr(evaluator, "confidential_execution_phases", None)
    if callable(hook):
        phases = hook(spec_path=spec_path, workspace=workspace)
        if not isinstance(phases, list) or not phases:
            raise RuntimeError("confidential_execution_phases returned no phases")
        return phases
    argv = evaluator._argv(
        spec_path=spec_path,
        workspace=workspace,
        spec_digest="a" * 64,
        policy_id="canary-policy",
        policy_version=0,
        baseline_metrics=None,
        name="vare-repo-" + ("ab" * 12),
    )
    return [{"role": "combined", "mounts": _parse_bind_mounts(argv)}]


def prove_hidden_test_confidentiality(work: Path | None = None) -> dict[str, Any]:
    """Plant a hidden-test canary and try to read it from the candidate namespace."""
    import tempfile

    token = "CANARY-" + hashlib.sha256(os.urandom(32)).hexdigest()
    owned = work is None
    if work is not None:
        work = Path(work).resolve()
    if work is None:
        temporary = tempfile.TemporaryDirectory(prefix="vare-mlsys-canary-")
        work = Path(temporary.name)
    else:
        temporary = None
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
    try:
        task_root = work / "task"
        hidden = task_root / "evaluator"
        hidden.mkdir(parents=True)
        (hidden / "hidden_canary.txt").write_text(token + "\n", encoding="utf-8")
        spec_path = task_root / "task.json"
        spec_path.write_text("{}\n", encoding="utf-8")
        allowed = work / "allowed"
        workspace = allowed / "candidate"
        workspace.mkdir(parents=True)
        (workspace / "solution.py").write_text("VALUE = 0\n", encoding="utf-8")
        phases = live_repository_grader_phases(spec_path, workspace, allowed)
        proof = assess_execution_phases(
            phases, canary_token=token, work=work / "probes"
        )
        proof["check"] = "hidden_test_canary_read"
        proof["docker_started"] = False
        return proof
    finally:
        if owned and temporary is not None:
            temporary.cleanup()


def clearance_status(root: Path = REPO_ROOT) -> dict[str, Any]:
    protocol = json.loads(
        (root / "protocols" / "mlsys_coding_policy_comparison_v1.json").read_text(
            encoding="utf-8"
        )
    )
    relative = protocol["authorization"]["required_clearance_file"]
    path = root / relative
    reasons = []
    if protocol["authorization"]["status"] != "authorized":
        reasons.append("authorization status is not authorized")
    if protocol["runner"]["implemented"] is not True:
        reasons.append("comparison runner is not implemented")
    if not path.is_file():
        reasons.append("isolation clearance file is absent")
        return {"accepted": False, "reasons": reasons}
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = protocol["authorization"]["required_clearance_fields"]
    if payload.get("hidden_tests_unreadable_from_candidate_namespace") is not True:
        reasons.append("clearance does not record hidden-test unreadability")
    if payload.get("pack_manifest_sha256") != required["pack_manifest_sha256"]:
        reasons.append("clearance manifest hash does not match the protocol")
    return {"accepted": not reasons, "reasons": reasons}


def real_grading_decision(work: Path | None = None) -> dict[str, Any]:
    proof = prove_hidden_test_confidentiality(work)
    clearance = clearance_status()
    reasons = []
    if proof["proven"] is not True:
        reasons.append(proof["reason"])
    reasons.extend(clearance["reasons"])
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "canary": proof,
        "clearance_accepted": clearance["accepted"],
    }


def grade_real_model_patch(
    patch: bytes, grader: Callable[[bytes], Any] | None = None
) -> Any:
    """Refuse unless the live canary proves hidden-test confidentiality.

    ``grader`` is not called on the current repository grader. The argument
    exists so tests can show that a patch consumer does not run.
    """
    if not isinstance(patch, bytes) or not patch:
        raise ValueError("model patch must be non-empty bytes")
    decision = real_grading_decision()
    if not decision["allowed"]:
        raise RealGradingRefused(decision)
    if grader is None:
        raise RealGradingRefused({**decision, "reason": "no grader was supplied"})
    return grader(patch)


def _peak_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _prompt_tokens(prompt: str) -> list[int]:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return list(digest[:8])


def _generation(task: Task, response: str, token: int, logp: float) -> dict[str, Any]:
    return {
        "prompt_id": task.id,
        "prompt": task.prompt,
        "response": response,
        "logprob": logp,
        "token_count": 1,
        "latency_s": 0.0,
        "metadata": {
            "backend": "synthetic-mock",
            "model": SYNTHETIC_MODEL,
            "sampling_temperature": 1.0,
            "logprob_distribution": BEHAVIOR,
            "prompt_token_ids": _prompt_tokens(task.prompt),
            "response_token_ids": [token],
            "response_token_logprobs": [logp],
            "policy_class": SYNTHETIC_MODEL,
        },
    }


def _agent_step(
    index: int,
    policy: PolicyCheckpoint,
    task: Task,
    response: str,
    token: int,
    logp: float,
    state: dict,
) -> AgentStep:
    return AgentStep(
        index,
        f"event-{index}",
        policy,
        {"kind": "inference"},
        {"split": "train", "task_id": task.id},
        state,
        _generation(task, response, token, logp),
    )


async def _run_grpo_group(pack: FrozenPack, output: Path) -> dict[str, Any]:
    task_dir = pack.task_directory(SYNTHETIC_TASK_ID, role="train")
    spec = EngineeringTaskSpec.load(task_dir / "task.json")
    task = Task(
        spec.id,
        spec.prompt,
        spec.family,
        {
            "learning_rule": "terminal_action_on_policy_grpo_v1",
            "policy_class": SYNTHETIC_MODEL,
            "split": "train",
        },
    )
    choices = (True, False, True, False)
    policy = PolicyCheckpoint(
        "synthetic-mock-policy",
        0,
        hashlib.sha256(b"synthetic-mock-policy-v0").hexdigest(),
    )
    verifier_sha = hashlib.sha256(b"mlsys-coding-v1-fixture-reward").hexdigest()
    protocol_sha = hashlib.sha256(pack.protocol_path.read_bytes()).hexdigest()
    store = TrajectoryStore.create(
        output / "grpo-trajectories.db", run_id="mlsys-synthetic-grpo", max_inflight=8
    )
    store.register_policy(policy)
    store.register_verifier("fixture-reward", 0, verifier_sha)
    trust = TrajectoryRewardRoot(
        b"synthetic-mlsys-comparison-v1-key!!", "mlsys-synthetic-grpo", protocol_sha
    )
    graded: dict[str, dict[str, Any]] = {}

    async def step_fn(resumed: dict, trajectory_id: str) -> AgentStep:
        member = int(trajectory_id.rsplit("-", 1)[1])
        oracle_patch = choices[member]
        decision = await asyncio.to_thread(
            pack.grade_fixture, SYNTHETIC_TASK_ID, oracle_patch=oracle_patch
        )
        graded[trajectory_id] = decision
        logp = bernoulli_logp(0.0, oracle_patch)
        response = "oracle-patch" if oracle_patch else "baseline-subject"
        return _agent_step(
            resumed["next_step"],
            policy,
            task,
            response,
            int(oracle_patch),
            logp,
            {
                "member": member,
                "oracle_patch": oracle_patch,
                "reward": decision["reward"],
            },
        )

    async def verify_fn(store: TrajectoryStore, trajectory_id: str) -> Verification:
        row = await owned_io(store.trajectory, trajectory_id)
        decision = graded[trajectory_id]
        value = Verification(
            float(decision["reward"]),
            bool(decision["success"]),
            1.0,
            0,
            "fixture-reward",
            trusted=True,
            metadata={
                "trajectory_sha256": row["head"],
                "verifier_config_sha256": verifier_sha,
                "reason": decision["reason"],
            },
        )
        return trust.attest(trajectory_id, row["head"], value)

    for member in range(GROUP_SIZE):
        store.submit(
            f"grpo-{member}",
            task,
            expected_steps=1,
            group_id="grpo-group",
            member=member,
            group_size=GROUP_SIZE,
        )
    stop = asyncio.Event()
    workers = [
        asyncio.create_task(
            execute_trajectories(
                store, "grpo-exec", step_fn, stop, lease_seconds=30, poll_seconds=0.01
            )
        ),
        asyncio.create_task(
            verify_trajectories(
                store,
                "grpo-verify",
                verify_fn,
                stop,
                lease_seconds=30,
                poll_seconds=0.01,
            )
        ),
    ]
    try:
        deadline = time.monotonic() + 120
        while "grpo-group" not in await owned_io(store.verified_groups):
            if time.monotonic() > deadline:
                raise TimeoutError("synthetic GRPO group was not verified")
            await asyncio.sleep(0.01)
        admission = await owned_io(
            admit_terminal_grpo_group,
            store,
            "grpo-group",
            active_policy=policy,
            verifier_name="fixture-reward",
            verifier_version=0,
            verifier_config_sha256=verifier_sha,
            trust_root=trust,
        )
        if not admission.admitted or len(admission.experiences) != GROUP_SIZE:
            raise RuntimeError(f"GRPO group was not admitted: {admission.reasons}")
        await owned_io(
            store.close_group, "grpo-group", admission.heads, decision={"arm": "grpo"}
        )
    finally:
        stop.set()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    choices_out = []
    rewards = []
    for experience in admission.experiences:
        oracle_patch = experience.attempt.output == "oracle-patch"
        choices_out.append(oracle_patch)
        rewards.append(float(experience.verification.score))
    update = grpo_update(choices_out, rewards)
    update["admitted_group"] = True
    update["trajectory_count"] = GROUP_SIZE
    update["task_id"] = SYNTHETIC_TASK_ID
    update["split"] = "train"
    return update


def _run_sft(pack: FrozenPack, output: Path) -> dict[str, Any]:
    oracle = pack.oracle_file(SYNTHETIC_TASK_ID, role="train")
    oracle_sha = sha256_file(oracle)
    task_dir = pack.task_directory(SYNTHETIC_TASK_ID, role="train")
    spec = EngineeringTaskSpec.load(task_dir / "task.json")
    task = Task(
        spec.id,
        spec.prompt,
        spec.family,
        {"policy_class": SYNTHETIC_MODEL, "split": "train"},
    )
    policy = PolicyCheckpoint(
        "synthetic-mock-sft", 0, hashlib.sha256(b"synthetic-mock-sft-v0").hexdigest()
    )

    async def scenario() -> None:
        store = TrajectoryStore.create(
            output / "sft-trajectories.db", run_id="mlsys-synthetic-sft", max_inflight=2
        )
        store.register_policy(policy)
        store.submit(
            "sft-0",
            task,
            expected_steps=1,
            group_id="sft-group",
            member=0,
            group_size=1,
        )
        stop = asyncio.Event()

        async def step_fn(resumed: dict, trajectory_id: str) -> AgentStep:
            logp = bernoulli_logp(0.0, True)
            return _agent_step(
                resumed["next_step"],
                policy,
                task,
                "oracle-patch",
                1,
                logp,
                {"oracle_sha256": oracle_sha, "task_id": task.id},
            )

        worker = asyncio.create_task(
            execute_trajectories(
                store, "sft-exec", step_fn, stop, lease_seconds=30, poll_seconds=0.01
            )
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                row = await owned_io(store.trajectory, "sft-0")
                if row["state"] == "complete":
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("synthetic SFT trajectory did not finish")
                await asyncio.sleep(0.01)
        finally:
            stop.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())
    update = sft_update()
    update["oracle_sha256"] = oracle_sha
    update["task_id"] = SYNTHETIC_TASK_ID
    update["split"] = "train"
    update["heldout_oracle_read"] = False
    return update


def _binomial_pmf(trials: int, probability: float) -> list[float]:
    probs = [1.0]
    for _ in range(trials):
        nxt = [0.0] * (len(probs) + 1)
        for count, mass in enumerate(probs):
            nxt[count] += mass * (1.0 - probability)
            nxt[count + 1] += mass * probability
        probs = nxt
    return probs


def probability_arm_beats_baseline(
    tasks: int, baseline_p: float, arm_p: float
) -> float:
    """P(Bin(tasks, arm) > Bin(tasks, baseline)) for independent tasks."""
    arm = _binomial_pmf(tasks, arm_p)
    baseline = _binomial_pmf(tasks, baseline_p)
    total = 0.0
    for arm_count, arm_mass in enumerate(arm):
        for baseline_count, baseline_mass in enumerate(baseline):
            if arm_count > baseline_count:
                total += arm_mass * baseline_mass
    return total


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta function."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise RuntimeError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(statistic: float, df: int) -> float:
    """One-sided survival function P(T >= statistic) for Student's t."""
    if df < 1:
        raise ValueError("df must be positive")
    if statistic == 0.0:
        return 0.5
    x = df / (df + statistic * statistic)
    tail = 0.5 * regularized_incomplete_beta(x, 0.5 * df, 0.5)
    return tail if statistic > 0.0 else 1.0 - tail


def student_t_critical(df: int, alpha: float = 0.05) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if student_t_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_task_t_test(differences: Sequence[float]) -> dict[str, float]:
    """One-sided paired t-test on per-task mean success differences.

    ``differences[i]`` is the trained-arm rate minus the no-update rate on
    task ``i``. A zero sample variance is a hit only when the mean is
    positive, which is the degenerate case in the v2 power script. The
    returned ``p`` is the one-sided tail. It is not Holm-adjusted.
    """
    count = len(differences)
    if count < 2:
        raise ValueError("paired t-test needs at least two tasks")
    mean = sum(differences) / count
    variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
    df = count - 1
    if variance == 0.0:
        if mean > 0.0:
            return {"df": df, "mean": mean, "p": 0.0, "t": math.inf}
        if mean < 0.0:
            return {"df": df, "mean": mean, "p": 1.0, "t": -math.inf}
        return {"df": df, "mean": mean, "p": 0.5, "t": 0.0}
    statistic = mean / math.sqrt(variance / count)
    return {
        "df": df,
        "mean": mean,
        "p": student_t_sf(statistic, df),
        "t": statistic,
    }


def holm_rejects(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Step-down Holm decisions, one flag per input p-value.

    For two pre-specified contrasts the smaller p-value is tested at
    ``alpha / 2`` and the larger at ``alpha``, and only if the smaller one
    was rejected.
    """
    count = len(p_values)
    if count < 1:
        raise ValueError("Holm needs at least one p-value")
    if any(not math.isfinite(p) or p < 0.0 for p in p_values):
        raise ValueError("p-values must be finite and non-negative")
    ordered = sorted(range(count), key=lambda index: (p_values[index], index))
    still_rejecting = True
    decisions = [False] * count
    for rank, index in enumerate(ordered):
        threshold = alpha / (count - rank)
        if still_rejecting and p_values[index] <= threshold:
            decisions[index] = True
        else:
            still_rejecting = False
    return decisions


def _seed_difference(
    rng: random.Random, tasks: int, baseline_p: float, arm_p: float
) -> float:
    baseline = sum(rng.random() < baseline_p for _ in range(tasks))
    arm = sum(rng.random() < arm_p for _ in range(tasks))
    return (arm - baseline) / tasks


def v1_accepts(differences: Sequence[float]) -> bool:
    """Frozen v1 rule: every seed difference is strictly positive.

    With that constraint the 2.5 percentile of the resampled mean is also
    strictly positive, so the bootstrap clause does not add a rejection.
    """
    return bool(differences) and all(difference > 0.0 for difference in differences)


def t_test_accepts(differences: Sequence[float], *, alpha: float = 0.05) -> bool:
    """One-sided t-test on seed-level paired differences. Level ``alpha``."""
    count = len(differences)
    if count < 2:
        return False
    mean = sum(differences) / count
    variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
    if variance == 0.0:
        return mean > 0.0
    statistic = mean / math.sqrt(variance / count)
    return statistic > student_t_critical(count - 1, alpha)


def monte_carlo_power(
    *,
    seeds: int,
    baseline_p: float,
    arm_p: float,
    tasks: int,
    trials: int,
    seed: int,
    rule: str,
) -> float:
    rng = random.Random(seed)
    accepts = t_test_accepts if rule == "t_test" else v1_accepts
    hits = 0
    for _ in range(trials):
        differences = [
            _seed_difference(rng, tasks, baseline_p, arm_p) for _ in range(seeds)
        ]
        hits += int(accepts(differences))
    return hits / trials


DRAFT_V2_TASKS = 6
DRAFT_V2_SEEDS = 8
DRAFT_V2_BASELINE_P = 0.05
DRAFT_V2_TRIALS = 4000
DRAFT_V2_RESAMPLES = 399
DRAFT_V2_SEED = 20261019


def paired_task_means(
    rng: random.Random, tasks: int, seeds: int, baseline_p: float, arm_p: float
) -> list[float]:
    """Mean paired success difference inside each task. Seeds stay nested."""
    means = []
    for _ in range(tasks):
        total = 0.0
        for _ in range(seeds):
            total += float(rng.random() < arm_p) - float(rng.random() < baseline_p)
        means.append(total / seeds)
    return means


def task_cluster_lower_percentile(
    means: Sequence[float],
    rng: random.Random,
    resamples: int,
    percentile: float = 2.5,
) -> float:
    """Percentile of the mean after resampling tasks and keeping their seeds."""
    count = len(means)
    if count < 2 or resamples < 2:
        raise ValueError(
            "task-cluster bootstrap needs at least two tasks and resamples"
        )
    stats = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += means[rng.randrange(count)]
        stats.append(total / count)
    stats.sort()
    position = (percentile / 100.0) * (resamples - 1)
    lower = int(position)
    upper = min(lower + 1, resamples - 1)
    weight = position - lower
    return stats[lower] * (1.0 - weight) + stats[upper] * weight


def task_cluster_accepts(
    means: Sequence[float], *, resamples: int, seed: int, percentile: float = 2.5
) -> bool:
    """True when a task-bootstrap lower percentile is strictly positive.

    This helper is the secondary interval's sign check. The draft v2
    acceptance rule is the paired t-test plus Holm, not this percentile.
    The helper also does not require every seed difference to be positive.
    """
    rng = random.Random(seed)
    return task_cluster_lower_percentile(means, rng, resamples, percentile) > 0.0


def cluster_power(
    *,
    tasks: int,
    seeds: int,
    baseline_p: float,
    arm_p: float,
    trials: int,
    resamples: int,
    seed: int,
) -> float:
    """Monte Carlo power of the task-cluster rule under independent Bernoulli draws."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        means = paired_task_means(rng, tasks, seeds, baseline_p, arm_p)
        if task_cluster_lower_percentile(means, rng, resamples) > 0.0:
            hits += 1
    return hits / trials


def power_analysis() -> dict[str, Any]:
    """Design power for v1 and the smallest v2 seed count that clears 0.8.

    The Bernoulli model is an analysis assumption. Temperature 0 in v1 makes
    the three evaluation seeds copies of one greedy draw, so the independent
    seed calculation is an optimistic reading of the written rule.
    """
    beat = probability_arm_beats_baseline(
        POWER_TASKS, REALISTIC_BASELINE_P, REALISTIC_ARM_P
    )
    small_beat = probability_arm_beats_baseline(POWER_TASKS, 0.05, 0.15)
    large_beat = probability_arm_beats_baseline(POWER_TASKS, 0.0, 0.50)
    quarter_beat = probability_arm_beats_baseline(POWER_TASKS, 0.0, 0.25)
    v1_independent = beat**3
    chosen = None
    for seeds in range(4, 81):
        power = monte_carlo_power(
            seeds=seeds,
            baseline_p=REALISTIC_BASELINE_P,
            arm_p=REALISTIC_ARM_P,
            tasks=POWER_TASKS,
            trials=POWER_TRIALS,
            seed=POWER_SEED,
            rule="t_test",
        )
        if power >= 0.8:
            chosen = {"seeds": seeds, "power": power}
            break
    if chosen is None:
        raise RuntimeError("no seed count in 4..80 reached power 0.8")
    type_i_low = monte_carlo_power(
        seeds=chosen["seeds"],
        baseline_p=REALISTIC_BASELINE_P,
        arm_p=REALISTIC_BASELINE_P,
        tasks=POWER_TASKS,
        trials=POWER_TRIALS,
        seed=POWER_SEED + 1,
        rule="t_test",
    )
    type_i_mid = monte_carlo_power(
        seeds=chosen["seeds"],
        baseline_p=0.30,
        arm_p=0.30,
        tasks=POWER_TASKS,
        trials=POWER_TRIALS,
        seed=POWER_SEED + 2,
        rule="t_test",
    )
    smaller = monte_carlo_power(
        seeds=chosen["seeds"],
        baseline_p=0.05,
        arm_p=0.15,
        tasks=POWER_TASKS,
        trials=POWER_TRIALS,
        seed=POWER_SEED + 3,
        rule="t_test",
    )
    return {
        "analysis_assumption": (
            "Independent Bernoulli task outcomes and independent seeds. "
            "This is not a measured agent success rate."
        ),
        "heldout_tasks": POWER_TASKS,
        "realistic_effect": {
            "arm_success_probability": REALISTIC_ARM_P,
            "baseline_success_probability": REALISTIC_BASELINE_P,
        },
        "trials": POWER_TRIALS,
        "v1": {
            "acceptance": "all three seed differences strictly positive",
            "exact_probability_one_seed_positive_at_realistic_effect": beat,
            "greedy_temperature_0_power": beat,
            "independent_seed_power": v1_independent,
            "independent_seed_power_at_0_to_0.25": quarter_beat**3,
            "independent_seed_power_at_0_to_0.50": large_beat**3,
            "independent_seed_power_at_0.05_to_0.15": small_beat**3,
            "seeds": 3,
            "temperature": 0,
            "detects_realistic_effect": False,
        },
        "v2_proposal": {
            "acceptance": "one-sided level-0.05 t-test on seed-level paired differences",
            "superseded_by": (
                "protocols/mlsys_coding_policy_comparison_v2.json uses a "
                "task-level paired t-test with a Holm correction. This block "
                "is an earlier seed-level calculation and is not that rule. "
                "The draft power table is benchmarks/mlsys_coding_v2/power.py."
            ),
            "critical_value": student_t_critical(chosen["seeds"] - 1, 0.05),
            "empirical_type_i_at_p_0.05": type_i_low,
            "empirical_type_i_at_p_0.30": type_i_mid,
            "power_at_0.05_to_0.15": smaller,
            "power_at_realistic_effect": chosen["power"],
            "seeds": chosen["seeds"],
            "temperature": 1.0,
        },
    }


# Retained CPU measurements from other tasks. Not remeasured on this pack.
RETAINED_CODE_REPAIR_TOKENS_PER_S = 12.29
RETAINED_CODE_REPAIR_PEAK_RSS_BYTES = 3_430_170_624
RETAINED_URLLIB3_GENERATION_S = 56.256
RETAINED_URLLIB3_TOKENS = 411
RETAINED_OPENBOOKQA_TRAIN_PEAK_RSS_MIB = 11924.6


def estimate_protocol_cost(measured: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Wall-clock sketch for one v1 arm. Token rate is from a different task."""
    eval_attempts = 3 * 4
    max_new_tokens = 2048
    train_tokens = 20000
    eval_generation_s = (
        eval_attempts * max_new_tokens / RETAINED_CODE_REPAIR_TOKENS_PER_S
    )
    train_generation_s = train_tokens / RETAINED_CODE_REPAIR_TOKENS_PER_S
    urllib3_rate = RETAINED_URLLIB3_TOKENS / RETAINED_URLLIB3_GENERATION_S
    return {
        "capability_claim": False,
        "eval_attempts_per_arm_v1": eval_attempts,
        "eval_generation_seconds_at_full_2048_token_cap": eval_generation_s,
        "eval_wall_cap_seconds_per_arm": eval_attempts * 600,
        "measured_synthetic_seconds": None if measured is None else dict(measured),
        "no_update_training_seconds": 0.0,
        "note": (
            "Generation time uses the retained 12.29 tokens/s from the CPU "
            "code-repair feasibility run, which solved 0/32 on a different task. "
            "It is not a measurement of this pack. A 0.5B full-parameter CPU "
            "update in the retained OpenBookQA comparison peaked at 11924.6 MiB, "
            "above this protocol's 8 GiB training cap. Inference RSS in the "
            "code-repair run was 3430170624 bytes."
        ),
        "protocol_rss_cap_bytes": 8_589_934_592,
        "retained_code_repair_peak_rss_bytes": RETAINED_CODE_REPAIR_PEAK_RSS_BYTES,
        "retained_code_repair_tokens_per_s": RETAINED_CODE_REPAIR_TOKENS_PER_S,
        "retained_openbookqa_train_peak_rss_mib": RETAINED_OPENBOOKQA_TRAIN_PEAK_RSS_MIB,
        "retained_urllib3_tokens_per_s": urllib3_rate,
        "sft_and_grpo_train_generation_seconds_at_20000_tokens": train_generation_s,
        "sft_and_grpo_train_wall_cap_seconds": 7200,
        "sources": [
            "docs/cpu-code-repair-feasibility-v1-report.md",
            "docs/urllib3-3352-real-model-feasibility-v1-report.md",
            "docs/openbookqa-qwen25-grpo-sft-confirmation-v1-report.md",
        ],
    }


def run_synthetic_dry_run(
    output: Path, *, performance_proxy: bool = True
) -> dict[str, Any]:
    """One CPU pass over train-split fixtures. Not a held-out comparison."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pack = FrozenPack()
    heldout_probe = {"rejected": False, "task_id": pack.heldout_ids[0]}
    try:
        pack.task_directory(pack.heldout_ids[0], role="train")
    except HeldOutAccessError as exc:
        heldout_probe = {
            "rejected": True,
            "task_id": pack.heldout_ids[0],
            "error": str(exc),
        }
    if heldout_probe["rejected"] is not True:
        raise HeldOutAccessError("held-out probe was not rejected")
    smoke = cpu_policy_step()
    grading = real_grading_decision(output / "canary")
    if grading["allowed"]:
        raise RuntimeError("real grading became allowed during a synthetic dry-run")
    arm_seconds: dict[str, float] = {}
    stamp = time.perf_counter()
    baseline = pack.grade_fixture(SYNTHETIC_TASK_ID, oracle_patch=False)
    arm_seconds["no_update"] = time.perf_counter() - stamp
    stamp = time.perf_counter()
    sft = _run_sft(pack, output)
    arm_seconds["sft"] = time.perf_counter() - stamp
    stamp = time.perf_counter()
    grpo = asyncio.run(_run_grpo_group(pack, output))
    arm_seconds["grpo"] = time.perf_counter() - stamp
    performance = None
    if performance_proxy:
        stamp = time.perf_counter()
        performance = pack.grade_fixture(PERFORMANCE_PROXY_TASK_ID, oracle_patch=False)
        performance["seconds"] = time.perf_counter() - stamp
        performance["role"] = "train_split_cost_proxy_not_a_heldout_score"
    summary = {
        "arms": {
            "grpo": grpo,
            "no_update": {
                "fixture_grade": baseline,
                "gradient_updates": 0,
                "split": "train",
                "task_id": SYNTHETIC_TASK_ID,
            },
            "sft": sft,
        },
        "capability_claim": False,
        "cost_estimate": estimate_protocol_cost(arm_seconds),
        "cpu_policy_step": smoke,
        "heldout_probe": heldout_probe,
        "heldout_tasks_read": [],
        "label": "synthetic_mock_policy_train_split_only",
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "peak_rss_bytes": _peak_rss_bytes(),
        "performance_proxy": performance,
        "protocol_budget_consumed": False,
        "protocol_id": "mlsys-coding-policy-comparison-v1",
        "real_grading": grading,
        "split_sha256": pack.split_sha256,
        "status": "synthetic_dry_run",
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def refusal_payload(root: Path = REPO_ROOT) -> dict[str, Any]:
    protocol = json.loads(
        (root / "protocols" / "mlsys_coding_policy_comparison_v1.json").read_text(
            encoding="utf-8"
        )
    )
    reasons = []
    if protocol["authorization"]["status"] != "authorized":
        reasons.append("authorization status is not authorized")
    clearance = root / protocol["authorization"]["required_clearance_file"]
    if not clearance.is_file():
        reasons.append("isolation clearance file is absent")
    if protocol["runner"]["implemented"] is not True:
        reasons.append("comparison runner is not implemented")
    reasons.append(
        "this revision does not load a model, start a container, or apply an update"
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "reasons": reasons,
        "status": "not_run",
    }
