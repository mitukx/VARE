"""Preregistered multi-scenario verifier-drift study with repeated real updates.

This is a bounded systems experiment using TinyStories-1M and the pinned RVL
GRPO trainer. Reward semantics are synthetic; the study does not measure code
repair capability. See benchmarks/verifier_consistent_async_grpo_v7/protocol.json.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.evaluation_evidence import (  # noqa: E402
    EvaluationEvidenceJournal,
    sha256_file as evidence_sha256_file,
    utc_now,
)

PROTOCOL_PATH = ROOT / "benchmarks/verifier_consistent_async_grpo_v7/protocol.json"
RETENTION_MANIFEST_PATH = ROOT / "benchmarks/verifier_consistent_async_grpo_v7/evaluation-retention-correction-v1.json"
GROUP_SIZE = 8
GROUPS_PER_ARM = 5
UPDATES_PER_ARM = 4
ACTIVE_EVALUATION_JOURNALS: dict[str, dict] = {}
INJECTED_FAILURE_POINTS: set[str] = set()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def inject_failure(point: str) -> None:
    """Test-only fault points exercise the real durability boundaries."""
    requested = os.environ.get("VARE_EXPERIMENT_FAILPOINT")
    if requested == point and point not in INJECTED_FAILURE_POINTS:
        INJECTED_FAILURE_POINTS.add(point)
        raise RuntimeError(f"injected failure at {point}")


async def promote_then_inject(promote, candidate_id: str, mark_applied) -> None:
    """Place the promotion fault after policy mutation but before its caller commits."""
    await promote(candidate_id)
    mark_applied()
    inject_failure("during_promotion")


def persist_active_evidence_failures(error: BaseException) -> None:
    """Persist failures for every arm that had not reached its completion boundary."""
    for evidence_root, context in list(ACTIVE_EVALUATION_JOURNALS.items()):
        try:
            context["journal"].record_failure(
                failure_type=type(error).__name__, message=str(error),
                stage=context.get("stage"), run_id=context.get("run_id"),
                scenario_id=context.get("scenario_id"), arm=context.get("arm"))
        except BaseException as evidence_error:
            print(f"failed to persist evaluation failure record at {evidence_root}: "
                  f"{evidence_error}", file=sys.stderr)


def validate_protocol(protocol: dict) -> None:
    required = {
        "arm_orders", "arms", "auditor_sha256", "base_commit", "crash_recovery",
        "group_delivery_order", "model_files", "model_revision", "pilot_gate",
        "primary_metric", "resource_limits", "runner_sha256", "rvl_backend_sha256",
        "rvl_patch_sha256", "rvl_revision", "scenarios", "training_groups",
        "v6_helper_sha256", "v7_support_sha256", "pinned_crash_recovery_files_sha256",
        "vare_source_sha256", "versions", "workload",
    }
    require(required <= protocol.keys(),
            f"protocol is missing top-level keys: {sorted(required - protocol.keys())}")
    workload = protocol["workload"]
    require({"seed", "training_prompts", "evaluation", "group_size",
             "groups_per_arm", "optimizer_updates_per_arm", "rollouts_per_arm",
             "verifier_calls_per_arm", "max_new_tokens"} <= workload.keys(),
            "protocol workload schema is incomplete")
    require(type(workload["seed"]) is int and workload["seed"] >= 0,
            "protocol workload seed must be a nonnegative integer")
    require(workload["group_size"] == GROUP_SIZE
            and workload["groups_per_arm"] == GROUPS_PER_ARM
            and workload["optimizer_updates_per_arm"] == UPDATES_PER_ARM
            and workload["rollouts_per_arm"] == GROUP_SIZE * GROUPS_PER_ARM
            and workload["verifier_calls_per_arm"] == GROUP_SIZE * GROUPS_PER_ARM,
            "protocol workload and runner budgets differ")
    require(isinstance(workload["evaluation"], dict)
            and isinstance(workload["training_prompts"], list)
            and bool(workload["training_prompts"])
            and isinstance(workload["evaluation"].get("prompts"), list)
            and bool(workload["evaluation"]["prompts"]),
            "protocol prompts are missing")
    scenarios = protocol["scenarios"]
    scenario_ids = [row["id"] for row in scenarios]
    require(len(scenario_ids) == len(set(scenario_ids)) and bool(scenario_ids),
            "protocol scenario identifiers are empty or duplicated")
    arm_names = {"sync", "naive_async", "vare_async"}
    require(set(protocol["arm_orders"]) == set(scenario_ids),
            "protocol arm order must specify every and only scenario")
    for key in ("group_delivery_order", "training_groups"):
        require(set(protocol[key]) == arm_names,
                f"protocol {key} must specify every and only registered arm")
    for scenario_id in scenario_ids:
        require(set(protocol["arm_orders"][scenario_id]) == arm_names
                and len(protocol["arm_orders"][scenario_id]) == len(arm_names),
                f"protocol arm order is incomplete for {scenario_id}")
    for arm in arm_names:
        require(set(protocol["group_delivery_order"][arm]) == set(range(GROUPS_PER_ARM)),
                f"protocol delivery schedule is incomplete for {arm}")
        require(len(protocol["training_groups"][arm]) == UPDATES_PER_ARM
                and len(set(protocol["training_groups"][arm])) == UPDATES_PER_ARM
                and set(protocol["training_groups"][arm]) <= set(range(GROUPS_PER_ARM)),
                f"protocol training schedule is invalid for {arm}")


def validate_retention_manifest() -> None:
    manifest = json.loads(RETENTION_MANIFEST_PATH.read_text())
    require(manifest.get("schema") == "vare.evaluation-retention-correction.v1",
            "unknown evaluation-retention correction identity")
    require(manifest.get("historical_v7_3_protocol_sha256") ==
            sha(ROOT / "benchmarks/verifier_consistent_async_grpo_v7/protocol.json"),
            "historical v7.3 protocol identity differs")
    require(manifest.get("historical_v7_3_runner_sha256") ==
            "d4ca48fd95a8c7e08ef88032e1dca63e7632286748ac2055029d75f34313f067",
            "historical v7.3 runner identity differs")
    require(manifest.get("prospective_runner_sha256") == sha(Path(__file__).resolve()),
            "prospective runner identity differs from retention manifest")
    require(manifest.get("evaluation_evidence_module_sha256") ==
            sha(ROOT / "src/vare/evaluation_evidence.py"),
            "evaluation evidence module differs from retention manifest")


def whole_word(text: str, word: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text,
                          flags=re.IGNORECASE))


def score(rule: str, text: str) -> float:
    if rule == "ground_word":
        return float(whole_word(text, "ground"))
    if rule == "punctuation_or_location":
        return float(bool(re.search(r"[.!?]", text)) or any(
            whole_word(text, word) for word in ("ground", "grass", "floor", "sky", "branch")))
    if rule == "exact_ground_period":
        return float(bool(re.fullmatch(r"\s*ground\.\s*", text, flags=re.IGNORECASE)))
    if rule == "not_ground_word":
        return float(not whole_word(text, "ground"))
    raise ValueError(f"unrecognized frozen verifier rule: {rule}")


def scorer_for(scenario: dict, version: int, text: str) -> float:
    versions = scenario["verifiers"]
    key = "v0" if version == 0 else "v1" if version == 1 else None
    if key is None:
        raise ValueError("invalid verifier version")
    return score(versions[key]["rule"], text)


def authoritative_score(scenario: dict, text: str) -> float:
    return score(scenario["verifiers"]["v1"]["rule"], text)


def disagreements(group: dict, scenario: dict) -> dict[str, int]:
    false_accepts = false_rejects = 0
    for member in group["members"]:
        truth = authoritative_score(scenario, member["attempt"].output)
        observed = float(member["score"])
        false_accepts += int(observed == 1.0 and truth == 0.0)
        false_rejects += int(observed == 0.0 and truth == 1.0)
    return {"false_accepts": false_accepts, "false_rejects": false_rejects,
            "total_disagreements": false_accepts + false_rejects}


def ranking_stats(observed: list[float], truth: list[float]) -> dict[str, int]:
    inversions = comparable = 0
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            observed_delta = observed[left] - observed[right]
            truth_delta = truth[left] - truth[right]
            if observed_delta != 0.0 and truth_delta != 0.0:
                comparable += 1
                inversions += int(observed_delta * truth_delta < 0.0)
    return {"pairwise_inversions": inversions, "comparable_pairs": comparable}


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


async def collect_group(helper, hooks, task, scenario: dict, state, *,
                        start_step: int, transition: bool) -> dict:
    """Generate then verify through a bounded one-item queue; no parallel model calls."""
    rows: list[dict | None] = [None] * GROUP_SIZE
    queue_wait = [0.0] * GROUP_SIZE
    started = time.perf_counter()
    active_id, active_version = hooks.active_policy()
    if not transition:
        for member in range(GROUP_SIZE):
            attempt = await hooks.rollout(task, active_id, active_version, start_step + member)
            rows[member] = {
                "attempt": attempt, "verifier_version": state.version,
                "score": scorer_for(scenario, state.version, attempt.output),
                "queue_wait_s": 0.0,
            }
        transition_record = None
    else:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        generated = {"count": 0}
        transition_record = {}

        async def producer() -> None:
            for member in range(GROUP_SIZE):
                attempt = await hooks.rollout(task, active_id, active_version, start_step + member)
                generated["count"] = member + 1
                await queue.put((member, attempt, time.perf_counter()))
                await asyncio.sleep(0)
            await queue.put(None)

        async def verifier() -> None:
            seen = 0
            while True:
                item = await queue.get()
                if item is None:
                    return
                member, attempt, enqueued = item
                queue_wait[member] = time.perf_counter() - enqueued
                version = state.version
                rows[member] = {
                    "attempt": attempt, "verifier_version": version,
                    "score": scorer_for(scenario, version, attempt.output),
                    "queue_wait_s": queue_wait[member],
                }
                seen += 1
                if seen == GROUP_SIZE // 2:
                    state.version = 1
                    transition_record.update({
                        "verified_members": seen,
                        "generated_members": generated["count"],
                        "rollouts_still_active": generated["count"] < GROUP_SIZE,
                    })
                await asyncio.sleep(0)

        await asyncio.gather(producer(), verifier())
        require(transition_record.get("rollouts_still_active") is True,
                "verifier transition did not overlap active rollout collection")

    members = [row for row in rows if row is not None]
    require(len(members) == GROUP_SIZE, "incomplete rollout group")
    return {
        "members": members,
        "versions": [int(member["verifier_version"]) for member in members],
        "behavior_policy": {"policy_id": active_id, "policy_version": active_version},
        "scores": [float(member["score"]) for member in members],
        "wall_s": time.perf_counter() - started,
        "queue_wait_s": queue_wait,
        "transition": transition_record,
        "disagreements": disagreements({"members": members}, scenario),
    }


def make_task(Task, scenario_id: str, group_index: int, prompt: str):
    return Task(f"{scenario_id}-group-{group_index:02d}", prompt, "landing",
                {"learning_rule": "terminal_action_on_policy_grpo_v1"})


def make_eval_tasks(Task, prompts: list[str], scenario_id: str):
    return [Task(f"eval-{scenario_id}-{index:02d}", prompt, "landing", {})
            for index, prompt in enumerate(prompts)]


async def run_naive_update(helper, runtime: dict, group: dict, group_id: str,
                           verifier_shas: dict, signer: dict, journal, promotion_cfg,
                           eval_records: list, arm_out: Path, round_index: int,
                           capture_evaluation):
    from vare.evidence import sha256_file, sha256_json
    from vare.promotion import PromotionGate

    hooks = runtime["hooks"]
    experiences = helper.make_experiences(group, group_id, verifier_shas=verifier_shas)
    incumbent_id = hooks.active_policy()[0]
    candidate_id = await hooks.train_candidate_with_freshness(
        incumbent_id, experiences, hooks._training_freshness_assessor)
    checkpoint_paths = runtime["checkpoint_paths"]
    before_report, before = await capture_evaluation(
        incumbent_id, "incumbent", round_index)
    after_report, after = await capture_evaluation(candidate_id, "candidate", round_index)
    signer_root = signer["root"]
    before_signed = signer_root.attest(
        before_report, checkpoint_sha256=sha256_file(checkpoint_paths[incumbent_id]))
    after_signed = signer_root.attest(
        after_report, checkpoint_sha256=sha256_file(checkpoint_paths[candidate_id]))
    eval_records.append({
        "round": round_index, "incumbent": before, "candidate": after,
        "incumbent_report": asdict(before_signed),
        "candidate_report": asdict(after_signed),
    })
    gate = PromotionGate(promotion_cfg, trust_root=signer_root)
    decision = gate.decide(before_signed, after_signed, expected_verifier_version=1)
    require(decision.accepted, f"independent v1 promotion gate rejected naive update: {decision.reasons}")
    evaluation_hash = sha256_json({"incumbent": asdict(before_signed),
                                   "candidate": asdict(after_signed)})
    transaction_id = f"naive-{round_index:02d}-{os.urandom(8).hex()}"
    journal.prepare(
        tx_id=transaction_id,
        incumbent_id=incumbent_id,
        incumbent_sha256=sha256_file(checkpoint_paths[incumbent_id]),
        candidate_id=candidate_id,
        candidate_sha256=sha256_file(checkpoint_paths[candidate_id]),
        evaluation_sha256=evaluation_hash,
    )
    await hooks.promote(candidate_id)
    journal.commit(transaction_id)
    return {"candidate_id": candidate_id, "accepted": decision.accepted,
            "reasons": list(decision.reasons), "training_group": group_id,
            "optimizer_calls": runtime["optimizer_calls"]["count"],
            "update_replay": runtime["updates"][-1],
            "policy_sha256": sha(checkpoint_paths[candidate_id]),
            "policy_tensor_sha256": __import__("validate_grpo_e2e_cpu").tensor_hash(
                hooks._states[candidate_id].training_state["model"]
                if hasattr(hooks._states[candidate_id], "training_state")
                else hooks._states[candidate_id]["model"]),
            "journal": journal.status()}


async def run_arm(args, helper, protocol: dict, scenario: dict,
                  scenario_index: int, arm: str, arm_position: int,
                  baseline_swap: int | None):
    from dataclasses import asdict
    from vare.attestation import EvidenceTrustRoot
    from vare.config import EngineConfig, LagConfig, PromotionConfig, ReplayConfig
    from vare.engine import CapabilityLoop
    from vare.evidence import sha256_file, sha256_json
    from vare.promotion_journal import PromotionJournal
    from vare.trajectory_admission import TrajectoryRewardRoot
    from vare.trajectory_learner import TrajectoryLearnerAdapter
    from vare.trajectory_store import PolicyCheckpoint, TrajectoryStore
    from vare.telemetry import EventLog

    from vare.types import Task

    scenario_id = scenario["id"]
    workload = protocol["workload"]
    experiment_seed = workload["seed"]
    arm_out = args.output / scenario_id / arm
    arm_out.mkdir(parents=True)
    evidence_journal = EvaluationEvidenceJournal(arm_out / "evaluation_evidence")
    ACTIVE_EVALUATION_JOURNALS[str(arm_out)] = {
        "journal": evidence_journal, "run_id": str(args.output.name),
        "scenario_id": scenario_id, "arm": arm, "stage": "runtime_initialization",
    }
    naive = arm == "naive_async"
    runtime = await helper.build_runtime(
        args, arm_out, int(experiment_seed + scenario_index * 100),
        naive=naive, eval_version=1)
    hooks, trainer, checkpoints = runtime["hooks"], runtime["trainer"], runtime["checkpoint_paths"]
    state = runtime["state"]
    initial_state_hash = runtime["initial_model_tensor_sha256"]
    state.version = 1 if scenario["id"] == "no_drift" else (0 if naive or arm == "vare_async" else 1)
    hooks.eval_tasks = make_eval_tasks(
        Task, workload["evaluation"]["prompts"], scenario_id)

    source_paths = {
        "runner": Path(__file__).resolve(),
        "protocol": args.protocol_path.resolve(),
        "retention_manifest": RETENTION_MANIFEST_PATH.resolve(),
        "rvl_helper": Path(helper.__file__).resolve(),
        "rvl_backend": (args.rvl_source / "src/rvl_systems/hf_backend.py").resolve(),
        "vare_rvl_grpo": (ROOT / "src/vare/integrations/rvl_grpo.py").resolve(),
        "evaluation_evidence": (ROOT / "src/vare/evaluation_evidence.py").resolve(),
    }
    source_hashes = {name: evidence_sha256_file(path)
                     for name, path in source_paths.items()}
    verifier_definition = {
        "scenario": scenario_id,
        "version": 1,
        "meaning": scenario["verifiers"]["v1"]["meaning"],
        "rule": scenario["verifiers"]["v1"]["rule"],
    }
    verifier_identity = hashlib.sha256(json.dumps(
        verifier_definition, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()

    async def capture_evaluation(policy_id: str, role: str, round_index: int | None = None):
        invocation_id = uuid.uuid4().hex
        started_at = utc_now()
        checkpoint_path = checkpoints[policy_id]
        task_prompts = [{"task_id": task.id, "prompt": task.prompt}
                        for task in hooks.eval_tasks]
        common = {
            "invocation_id": invocation_id,
            "role": role,
            "run_id": str(args.output.name),
            "scenario_id": scenario_id,
            "arm": arm,
            "round": round_index,
            "started_at_utc": started_at,
            "model": {"name": "TinyStories-1M", "revision": protocol["model_revision"],
                      "files_sha256": protocol["model_files"]},
            "checkpoint": {"policy_id": policy_id,
                           "file_sha256": evidence_sha256_file(checkpoint_path)},
            "verifier": {"name": "landing", "version": 1,
                         "identity_sha256": verifier_identity,
                         "definition": verifier_definition,
                         "pass_threshold": 0.5},
            "evaluator": {"name": "independent greedy held-out evaluator",
                          "identity_sha256": source_hashes["rvl_helper"],
                          "decode": protocol["workload"]["evaluation"]["decode"]},
            "source_hashes": source_hashes,
        }
        evidence_journal.record_evaluation_started({
            "schema": "vare.evaluation-start.v1", **common,
            "tasks": task_prompts,
        })
        ACTIVE_EVALUATION_JOURNALS[str(arm_out)]["stage"] = (
            f"evaluation:{role}:{invocation_id}")
        report, details = await helper.independent_evaluate(runtime, policy_id, 1)
        outputs = details["outputs"]
        scores = details["scores"]
        require(set(outputs) == {row["task_id"] for row in task_prompts}
                and set(scores) == set(outputs),
                "independent evaluation output IDs differ from frozen task list")
        completed_at = utc_now()
        record = {
            "schema": "vare.evaluation-evidence.v1", **common,
            "completed_at_utc": completed_at,
            "aggregate": {"primary": float(report.primary), "n": int(report.n)},
            "tasks": [{
                "task_id": row["task_id"], "prompt": row["prompt"],
                "output_text": outputs[row["task_id"]]["text"],
                "output_token_ids": outputs[row["task_id"]]["token_ids"],
                "score": float(scores[row["task_id"]]),
                "passed": float(scores[row["task_id"]]) >= 0.5,
            } for row in task_prompts],
            "evaluation_invocation": {
                "id": invocation_id, "started_at_utc": started_at,
                "completed_at_utc": completed_at,
            },
        }
        evidence_journal.record_evaluation(record)
        ACTIVE_EVALUATION_JOURNALS[str(arm_out)]["stage"] = "evaluation_durable"
        inject_failure("after_evaluation")
        return report, details

    verifier_shas = {
        version: sha256_json({"scenario": scenario_id, "version": version,
                              "rule": scenario["verifiers"]["v0" if version == 0 else "v1"]["rule"]})
        for version in (0, 1)
    }
    signer = {"root": None}
    eval_records = []

    async def evaluate(policy_id: str, _runtime=runtime, _signer=signer,
                       _files=checkpoints, _records=eval_records):
        active_id = hooks.active_policy()[0]
        role = "incumbent" if policy_id == active_id else "candidate"
        report, details = await capture_evaluation(policy_id, role, len(_records))
        root = _signer["root"]
        signed = root.attest(report, checkpoint_sha256=sha256_file(_files[policy_id]))
        _records.append({"policy_id": policy_id, "report": asdict(signed), **details})
        return signed

    original_promote = hooks.promote

    async def fault_injected_promote(candidate_id: str):
        await promote_then_inject(
            original_promote, candidate_id,
            lambda: ACTIVE_EVALUATION_JOURNALS[str(arm_out)].__setitem__(
                "stage", "promotion_applied_before_commit"))

    hooks.promote = fault_injected_promote

    # Existing GRPO helper closes over its module-level scorer; binding it here
    # makes both training-time and held-out grading use the frozen scenario.
    helper.versioned_score = lambda version, text: scorer_for(scenario, version, text)

    journal = PromotionJournal(
        arm_out / "promotion.sqlite", initial_policy_id="policy-0",
        initial_checkpoint_sha256=sha256_file(checkpoints["policy-0"]))
    promotion_cfg = PromotionConfig(
        min_primary_gain=-1.0, max_slice_regression=1.0,
        max_verifier_disagreement=1.0, min_eval_examples=len(hooks.eval_tasks),
        max_relative_cost_increase=1e6)

    store = adapter = loop = reward_root = None
    registered_policies = {"policy-0"}
    if arm != "naive_async":
        run_id = f"vare-grpo-v7-{scenario_id}-{arm}-seed{experiment_seed + scenario_index * 100}"
        store = TrajectoryStore.create(arm_out / "trajectories.sqlite",
                                       run_id=run_id, max_inflight=64)

        def current_policy():
            policy_id, policy_version = hooks.active_policy()
            return PolicyCheckpoint(policy_id, policy_version,
                                    sha256_file(checkpoints[policy_id]))

        for version in (0, 1):
            store.register_verifier("landing", version, verifier_shas[version])
        initial_policy = current_policy()
        store.register_policy(initial_policy)
        reward_root = TrajectoryRewardRoot(os.urandom(32), run_id, sha(args.protocol_path))
        initial_verifier_version = state.version
        eval_root = EvidenceTrustRoot(
            os.urandom(32), sha(args.protocol_path), verifier_shas[1], run_id,
            sha256_json([asdict(task) for task in hooks.eval_tasks]),
            f"eval-v7-{scenario_id}-{arm}-nonce-0000", True)
        signer["root"] = eval_root
        adapter = TrajectoryLearnerAdapter(
            store, trust_root=reward_root, current_policy=current_policy,
            verifier_identity=("landing", initial_verifier_version,
                               verifier_shas[initial_verifier_version]),
            reference_families=("landing",), max_groups=2, max_members=GROUP_SIZE)
        hooks.training_evidence_authorizer = adapter.authorizes
        hooks.require_authenticated_training = True
        loop = CapabilityLoop(
            hooks=hooks, verifier=state, require_authenticated_training=True,
            training_evidence_authorizer=adapter.authorizes,
            require_current_evaluation_verifier=True, evidence_trust_root=eval_root,
            promotion_journal=journal,
            checkpoint_hasher=lambda policy_id: sha256_file(checkpoints[policy_id]),
            config=EngineConfig(
                samples_per_task=GROUP_SIZE, replay_batch_size=GROUP_SIZE,
                replay=ReplayConfig(capacity=GROUP_SIZE),
                lag=LagConfig(max_policy_lag=0, max_verifier_lag=0),
                promotion=promotion_cfg),
            event_log=EventLog(arm_out / "events.jsonl"),
            seed=int(experiment_seed + scenario_index * 100))
        hooks.evaluate = evaluate
        loop.evidence_trust_root = eval_root
        hooks.training_evidence_authorizer = adapter.authorizes
    else:
        signer["root"] = EvidenceTrustRoot(
            os.urandom(32), sha(args.protocol_path), verifier_shas[1],
            f"naive-{scenario_id}-seed{experiment_seed + scenario_index * 100}",
            sha256_json([asdict(task) for task in hooks.eval_tasks]),
            f"naive-eval-{scenario_id}-nonce-0000", True)
        # Version-unaware RVL advantage normalization is the deliberate negative control.
        helper.install_naive_rvl_reward_path(hooks)
        from vare.lag import LagController, make_experience_freshness_assessor
        hooks._training_freshness_assessor = make_experience_freshness_assessor(
            LagController(LagConfig(max_policy_lag=1, max_verifier_lag=1)),
            active_policy_version=lambda: hooks.active_policy()[1],
            active_verifier_version=lambda: state.version)

    # Every arm performs five equal 8-completion groups and four commits.
    # Async arms hold group 0 while training fresh group 1 and generating group
    # 2 from the promoted policy. The resulting queue contains old-policy,
    # mixed-verifier group 0 beside new-policy, current-verifier group 2.
    prompt_list = workload["training_prompts"]
    groups_by_index = {}
    updates = []
    audit_records = []
    delivered_groups = 0
    arm_started = time.perf_counter()
    arm_cpu_started = time.process_time()
    baseline_report, baseline_detail = await capture_evaluation("policy-0", "baseline")
    audit_records.append({"kind": "baseline", "report": asdict(baseline_report), **baseline_detail})

    def record_group(group_index: int, group: dict, task) -> None:
        group["group_id"] = f"g-{group_index:02d}-{scenario_id}"
        group["task"] = asdict(task)
        group["policy_versions"] = sorted({
            int(row["attempt"].policy_version) for row in group["members"]})
        group["policy_ids"] = sorted({row["attempt"].policy_id for row in group["members"]})
        require(len(group["policy_versions"]) == 1 and len(group["policy_ids"]) == 1,
                "rollout group mixes behavior policies")
        group["independent_scores"] = [
            authoritative_score(scenario, row["attempt"].output) for row in group["members"]]
        group["ranking"] = ranking_stats(group["scores"], group["independent_scores"])
        group["response_text_sha256"] = hashlib.sha256(json.dumps(
            [row["attempt"].output for row in group["members"]],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        group["_generated_at"] = time.perf_counter()
        groups_by_index[group_index] = group

    def write_rollouts() -> None:
        serialized = []
        for index in sorted(groups_by_index):
            group = groups_by_index[index]
            row = helper.serializable_group(group)
            row.update({
                "behavior_policy": group["behavior_policy"],
                "policy_ids": group["policy_ids"],
                "versions": group["versions"],
                "scores": group["scores"],
                "group_id": group["group_id"],
                "task": group["task"],
                "policy_versions": group["policy_versions"],
                "independent_scores": group["independent_scores"],
                "disagreements": group["disagreements"],
                "ranking": group["ranking"],
                "response_text_sha256": group["response_text_sha256"],
            })
            if "queue_residence_s" in group:
                row["queue_residence_s"] = group["queue_residence_s"]
            if "delivery_lag" in group:
                row["delivery_lag"] = group["delivery_lag"]
            serialized.append(row)
        write(arm_out / "rollouts.json", {
            "scenario": scenario_id, "arm": arm, "groups": serialized})

    async def generate_group(group_index: int) -> dict:
        task = make_task(Task, scenario_id, group_index,
                         prompt_list[group_index % len(prompt_list)])
        group = await collect_group(
            helper, hooks, task, scenario, state,
            start_step=group_index * GROUP_SIZE,
            transition=bool(group_index == 0 and arm != "sync"
                            and scenario_id != "no_drift"))
        record_group(group_index, group, task)
        write_rollouts()
        return group

    async_arm = arm != "sync"
    if async_arm:
        # Produce two groups under policy 0 before learning. Group 0 crosses
        # the verifier transition; group 1 is wholly current and trains first.
        await generate_group(0)
        await generate_group(1)
    delivery_order = protocol["group_delivery_order"][arm]
    persisted_groups: set[int] = set()
    preconsumed_groups: set[int] = set()

    def persist_vare_group(group_index: int) -> None:
        group = groups_by_index[group_index]
        task = Task(**group["task"])
        behavior_policy = group["behavior_policy"]
        behavior_id = behavior_policy["policy_id"]
        policy = PolicyCheckpoint(behavior_id, behavior_policy["policy_version"],
                                  sha256_file(checkpoints[behavior_id]))
        if policy.policy_id not in registered_policies:
            store.register_policy(policy)
            registered_policies.add(policy.policy_id)
        group_id = group["group_id"]
        helper.persist_group(
            store, group_id, group["members"], policy=policy,
            reward_root=reward_root, verifier_shas=verifier_shas, task=task)
        persisted_groups.add(group_index)

    for group_index in delivery_order:
        if group_index not in groups_by_index:
            await generate_group(group_index)
        if async_arm and group_index == 0 and 2 not in groups_by_index:
            # This is the preregistered stale-queue condition: keep group 0
            # while a new policy generates group 2, then deliver group 0.
            await generate_group(2)
            queued = [groups_by_index[0], groups_by_index[2]]
            queued_policy_versions = sorted({
                version for item in queued for version in item["policy_versions"]})
            queued_verifier_versions = sorted({
                version for item in queued for version in item["versions"]})
            require(queued_policy_versions == [0, 1],
                    "delayed queue does not contain two behavior-policy versions")
            if scenario_id != "no_drift":
                require(queued_verifier_versions == [0, 1],
                        "delayed queue does not contain both verifier versions")
            audit_records.append({
                "kind": "buffered_multi_version_queue",
                "delivered_next": groups_by_index[0]["group_id"],
                "also_buffered": groups_by_index[2]["group_id"],
                "policy_versions": queued_policy_versions,
                "verifier_versions": queued_verifier_versions,
                "queue_depth_groups": len(queued),
            })

        group = groups_by_index[group_index]
        group_id = group["group_id"]
        task = Task(**group["task"])
        if (arm == "vare_async"
                and adapter.current_verifier() != ("landing", 1, verifier_shas[1])):
            adapter.activate_verifier(("landing", 1, verifier_shas[1]))
        if "queue_residence_s" not in group:
            group["queue_residence_s"] = time.perf_counter() - group["_generated_at"]
        if "delivery_lag" not in group:
            active_id, active_version = hooks.active_policy()
            group["delivery_lag"] = {
                "active_policy_id": active_id,
                "active_policy_version": active_version,
                "policy_lag": max(0, active_version -
                                  group["behavior_policy"]["policy_version"]),
                "active_verifier_version": state.version,
                "verifier_lags": [max(0, state.version - item["verifier_version"])
                                  for item in group["members"]],
            }
        eligible = group_index in set(protocol["training_groups"][arm])
        stale_vare_quarantine = arm == "vare_async" and group_index == 0
        preconsumed = arm == "vare_async" and group_index in preconsumed_groups
        if eligible and not stale_vare_quarantine:
            require(len(set(group["scores"])) > 1,
                    f"preregistered stop: degenerate GRPO group {scenario_id}/{arm}/{group_index}")
        if stale_vare_quarantine:
            active_policy = hooks.active_policy()
            behavior_policy = group["behavior_policy"]
            require(behavior_policy["policy_version"] < active_policy[1],
                    "stale group was not held across a policy promotion")
            # Refresh verifier lineage before asking VARE to admit delayed data.
            if adapter.current_verifier() != ("landing", 1, verifier_shas[1]):
                adapter.activate_verifier(("landing", 1, verifier_shas[1]))

        update_record = None
        if preconsumed:
            audit_records.append({"kind": "group_consumed_from_buffered_queue",
                                  "group_id": group_id,
                                  "consumer_delivery": "g-00-" + scenario_id})
        elif arm == "naive_async" and eligible:
            signer["root"] = EvidenceTrustRoot(
                os.urandom(32), sha(args.protocol_path), verifier_shas[1],
                    f"naive-{scenario_id}-seed{experiment_seed + scenario_index * 100}",
                sha256_json([asdict(task) for task in hooks.eval_tasks]),
                f"naive-eval-{scenario_id}-nonce-{group_index:04d}", True)
            update_record = await run_naive_update(
                helper, runtime, group, group_id, verifier_shas, signer,
                journal, promotion_cfg, eval_records, arm_out, len(updates),
                capture_evaluation)
        elif arm != "naive_async":
            if group_index not in persisted_groups:
                persist_vare_group(group_index)
            if stale_vare_quarantine:
                require(2 in groups_by_index,
                        "stale VARE delivery requires the fresh buffered policy-1 group")
                require(len(set(groups_by_index[2]["scores"])) > 1,
                        "preregistered stop: degenerate buffered GRPO group")
                if 2 not in persisted_groups:
                    persist_vare_group(2)
                groups_by_index[2]["queue_residence_s"] = (
                    time.perf_counter() - groups_by_index[2]["_generated_at"])
                current_id, current_version = hooks.active_policy()
                groups_by_index[2]["delivery_lag"] = {
                    "active_policy_id": current_id,
                    "active_policy_version": current_version,
                    "policy_lag": max(0, current_version -
                                      groups_by_index[2]["behavior_policy"]["policy_version"]),
                    "active_verifier_version": state.version,
                    "verifier_lags": [max(0, state.version - item["verifier_version"])
                                      for item in groups_by_index[2]["members"]],
                }
                pending_ids = store.verified_groups(limit=16)
                require(pending_ids == ["g-00-" + scenario_id,
                                        "g-02-" + scenario_id],
                        f"trajectory store lacks the frozen two-policy queue: {pending_ids}")
                write_rollouts()
                audit_records.append({
                    "kind": "trajectory_store_multi_version_queue",
                    "group_ids": pending_ids,
                    "policy_versions": [0, 1],
                    "verifier_versions": sorted({
                        version for index in (0, 2)
                        for version in groups_by_index[index]["versions"]}),
                })
            admission_attempt = eligible or (
                stale_vare_quarantine)
            if admission_attempt:
                signer["root"] = EvidenceTrustRoot(
                    os.urandom(32), sha(args.protocol_path), verifier_shas[1],
                    f"vare-{scenario_id}-{arm}-seed{experiment_seed + scenario_index * 100}",
                    sha256_json([asdict(t) for t in hooks.eval_tasks]),
                    f"vare-eval-{scenario_id}-{arm}-nonce-{group_index:04d}", True)
                loop.evidence_trust_root = signer["root"]
                from vare.promotion import PromotionGate
                loop.promotion = PromotionGate(promotion_cfg, trust_root=signer["root"])
                prior_steps = runtime["optimizer_calls"]["count"]
                advanced = await adapter.advance(
                    loop, round_index=(2 if stale_vare_quarantine else group_index))
                if stale_vare_quarantine:
                    require(advanced is not None,
                            "VARE did not skip the stale group and train the queued fresh group")
                    counts = store.counts()
                    require(counts.get("quarantined", 0) >= GROUP_SIZE,
                            "VARE did not quarantine all stale-group members")
                    require(runtime["optimizer_calls"]["count"] == prior_steps + 1,
                            "fresh buffered group did not produce exactly one optimizer step")
                    stale_rows = store.group(group_id)
                    fresh_rows = store.group("g-02-" + scenario_id)
                    require(len(stale_rows) == GROUP_SIZE
                            and all(row["state"] == "quarantined" for row in stale_rows),
                            "stale buffered group was not atomically quarantined")
                    require(len(fresh_rows) == GROUP_SIZE
                            and all(row["state"] == "consumed" for row in fresh_rows),
                            "fresh buffered group was not consumed exactly once")
                    audit_records.append({"kind": "quarantine", "group_id": group_id,
                                          "counts": counts,
                                          "decisions": [row["reason"] for row in stale_rows],
                                          "also_trained_group": "g-02-" + scenario_id})
                    update_record = {
                        "candidate_id": advanced.candidate_id,
                        "promoted_id": advanced.promoted_id,
                        "accepted": advanced.decision.accepted,
                        "training_group": "g-02-" + scenario_id,
                        "optimizer_calls": runtime["optimizer_calls"]["count"],
                        "update_replay": runtime["updates"][-1],
                        "policy_sha256": sha(checkpoints[advanced.promoted_id]),
                        "policy_tensor_sha256": __import__("validate_grpo_e2e_cpu").tensor_hash(
                            hooks._states[advanced.promoted_id].training_state["model"]
                            if hasattr(hooks._states[advanced.promoted_id], "training_state")
                            else hooks._states[advanced.promoted_id]["model"]),
                        "journal": journal.status(),
                    }
                    preconsumed_groups.add(2)
                else:
                    require(advanced is not None,
                            "fresh verifier-consistent group did not produce a round")
                    require(runtime["optimizer_calls"]["count"] == prior_steps + 1,
                            "learner round did not commit exactly one optimizer step")
                    update_record = {
                        "candidate_id": advanced.candidate_id,
                        "promoted_id": advanced.promoted_id,
                        "accepted": advanced.decision.accepted,
                        "training_group": group_id,
                        "optimizer_calls": runtime["optimizer_calls"]["count"],
                        "update_replay": runtime["updates"][-1],
                        "policy_sha256": sha(checkpoints[advanced.promoted_id]),
                        "policy_tensor_sha256": __import__("validate_grpo_e2e_cpu").tensor_hash(
                            hooks._states[advanced.promoted_id].training_state["model"]
                            if hasattr(hooks._states[advanced.promoted_id], "training_state")
                            else hooks._states[advanced.promoted_id]["model"]),
                        "journal": journal.status(),
                    }
            elif group_index == 4:
                audit_records.append({"kind": "budget_matched_unused_group",
                                      "group_id": group_id})
        else:
            if group_index == 4:
                audit_records.append({"kind": "budget_matched_unused_group",
                                      "group_id": group_id})

        if update_record is not None:
            updates.append(update_record)
            audit_records.append({"kind": "update", **update_record})
        write_rollouts()
        write(arm_out / "progress.json", {
            "scenario": scenario_id, "arm": arm,
            "generated_groups": len(groups_by_index),
            "delivered_groups": delivered_groups + 1,
            "optimizer_updates": len(updates),
            "elapsed_s": time.perf_counter() - arm_started,
            "peak_rss_bytes": peak_rss_bytes(),
        })
        limits = protocol["resource_limits"]
        require(time.perf_counter() - arm_started <= limits["max_wall_seconds_per_arm"],
                "per-arm wall-time budget exceeded at group boundary")
        require(time.perf_counter() - args.started <= limits["max_wall_seconds_total"],
                "total wall-time budget exceeded at group boundary")
        require(peak_rss_bytes() <= limits["max_rss_bytes"],
                "peak RSS exceeded budget at group boundary")
        current_swap = helper.swap_used_bytes()
        if baseline_swap is not None and current_swap is not None:
            require(max(0, current_swap - baseline_swap)
                    <= limits["max_positive_swap_growth_bytes"],
                    "positive swap growth exceeded budget at group boundary")
        delivered_groups += 1

    groups = [groups_by_index[index] for index in sorted(groups_by_index)]
    require(len(groups) == GROUPS_PER_ARM, "arm did not generate the complete fixed workload")
    write(arm_out / "evaluations.json", eval_records)
    write(arm_out / "audit_records.json", audit_records)
    total_wall = time.perf_counter() - arm_started
    require(len(updates) == UPDATES_PER_ARM,
            f"{arm} committed {len(updates)} updates, expected {UPDATES_PER_ARM}")
    require(runtime["optimizer_calls"]["count"] == UPDATES_PER_ARM,
            "optimizer call count differs from matched update budget")
    store_counts = store.counts() if store is not None else None
    if store is not None:
        store.backup(arm_out / "snapshot.sqlite")
        from vare.trajectory_store import audit_trajectory_store
        store_audit = audit_trajectory_store(arm_out / "snapshot.sqlite")
        require(store_audit.get("pass") is True,
                "offline trajectory ledger audit failed")
    else:
        store_audit = None
    final_policy = hooks.active_policy()[0]
    baseline_mean = baseline_report.primary
    final_report, final_detail = await capture_evaluation(final_policy, "final")
    ACTIVE_EVALUATION_JOURNALS[str(arm_out)]["stage"] = "before_arm_completion"
    inject_failure("before_arm_completion")
    sample_latencies = [float(member["attempt"].latency_ms) / 1000.0
                        for group in groups for member in group["members"]]
    queue_waits = [float(member["queue_wait_s"])
                   for group in groups for member in group["members"]]
    group_latencies = [float(group["wall_s"]) for group in groups]
    queue_residence = [float(group["queue_residence_s"]) for group in groups]
    generated_tokens = sum(
        int(member["attempt"].metadata["rvl_generation"]["token_count"])
        for group in groups for member in group["members"])
    selected_groups = {row["training_group"] for row in updates}
    groups_by_id = {group["group_id"]: group for group in groups}
    invalid_commits = sum(
        groups_by_id[group_id]["disagreements"]["total_disagreements"] > 0
        for group_id in selected_groups)
    evaluation_scores = list(baseline_detail["scores"].values())
    for record in eval_records:
        evaluation_scores.extend(record.get("scores", {}).values())
        evaluation_scores.extend(record.get("candidate", {}).get("scores", {}).values())
    evaluation_scores.extend(final_detail["scores"].values())
    summary = {
        "scenario": scenario_id, "arm": arm,
        "run_order_position": arm_position,
        "initial_model_tensor_sha256": initial_state_hash,
        "initial_checkpoint_sha256": sha(checkpoints["policy-0"]),
        "final_checkpoint_sha256": sha(checkpoints[final_policy]),
        "final_policy_tensor_sha256": __import__("validate_grpo_e2e_cpu").tensor_hash(
            hooks._states[final_policy].training_state["model"]
            if hasattr(hooks._states[final_policy], "training_state")
            else hooks._states[final_policy]["model"]),
        "groups_generated": GROUPS_PER_ARM,
        "rollouts": GROUP_SIZE * GROUPS_PER_ARM,
        "verifier_calls": GROUP_SIZE * GROUPS_PER_ARM,
        "optimizer_calls": runtime["optimizer_calls"]["count"],
        "committed_updates": len(updates), "updates": updates,
        "groups": [{"group_id": g["group_id"], "versions": g["versions"],
                    "behavior_policy": g["behavior_policy"],
                    "policy_ids": g["policy_ids"],
                    "policy_versions": g["policy_versions"],
                    "scores": g["scores"], "independent_scores": g["independent_scores"],
                    "disagreements": g["disagreements"], "ranking": g["ranking"],
                    "response_text_sha256": g["response_text_sha256"],
                    "transition": g["transition"], "wall_s": g["wall_s"],
                    "queue_wait_s": g["queue_wait_s"],
                    "queue_residence_s": g["queue_residence_s"],
                    "delivery_lag": g["delivery_lag"]} for g in groups],
        "baseline_evaluation": {"primary": baseline_mean, "n": baseline_report.n,
                                "scores": baseline_detail["scores"]},
        "final_evaluation": {"primary": final_report.primary, "n": final_report.n,
                             "scores": final_detail["scores"]},
        "heldout_delta": final_report.primary - baseline_mean,
        "per_task_final_delta": {
            task_id: final_detail["scores"][task_id] - baseline_detail["scores"][task_id]
            for task_id in baseline_detail["scores"]},
        "evaluation_saturated": len(set(evaluation_scores)) <= 1,
        "policy_changed": __import__("validate_grpo_e2e_cpu").tensor_hash(
            hooks._states[final_policy].training_state["model"]
            if hasattr(hooks._states[final_policy], "training_state")
            else hooks._states[final_policy]["model"]) != initial_state_hash,
        "store_counts": store_counts, "store_audit": store_audit,
        "promotion_journal": journal.status(),
        "valid_committed_updates": len(updates) - invalid_commits,
        "invalid_committed_updates": invalid_commits,
        "checkpoint_io": runtime["checkpoint_io"],
        "checkpoint_bytes_written": sum(item["bytes"] for item in runtime["checkpoint_io"]),
        "checkpoint_write_seconds": sum(item["write_s"] for item in runtime["checkpoint_io"]),
        "generated_tokens": generated_tokens,
        "latency_seconds": {
            "sample_p50": helper.percentile(sample_latencies, 0.5),
            "sample_p95": helper.percentile(sample_latencies, 0.95),
            "group_p50": helper.percentile(group_latencies, 0.5),
            "group_p95": helper.percentile(group_latencies, 0.95),
            "queue_wait_p50": helper.percentile(queue_waits, 0.5),
            "queue_wait_p95": helper.percentile(queue_waits, 0.95),
            "group_queue_residence_p50": helper.percentile(queue_residence, 0.5),
            "group_queue_residence_p95": helper.percentile(queue_residence, 0.95),
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "cpu_s": time.process_time() - arm_cpu_started,
        "wall_s": total_wall,
        "rollouts_per_hour": GROUP_SIZE * GROUPS_PER_ARM * 3600 / max(total_wall, 1e-9),
        "committed_updates_per_hour": len(updates) * 3600 / max(total_wall, 1e-9),
        "valid_updates_per_hour": (len(updates) - invalid_commits) * 3600 / max(total_wall, 1e-9),
        "quarantined_valid_trajectories": sum(
            int(float(member["score"]) == float(group["independent_scores"][index]))
            for group in groups if arm == "vare_async" and scenario_id != "no_drift"
            and group["group_id"].endswith("group-00")
            for index, member in enumerate(group["members"])),
    }
    write(arm_out / "summary.json", summary)
    # Keep compact traces and checkpoints locally for independent reproduction.
    files = {str(path.relative_to(arm_out)): sha(path)
             for path in arm_out.rglob("*") if path.is_file()
             and path.name not in {"SHA256.json"}
             and not path.name.endswith(("-wal", "-shm"))}
    write(arm_out / "SHA256.json", files)
    ACTIVE_EVALUATION_JOURNALS.pop(str(arm_out), None)
    del runtime, hooks, trainer, store, adapter, loop
    import gc
    gc.collect()
    return summary


async def run(args) -> None:
    require(sys.flags.no_site == 1, "launch with python -S for import isolation")
    require(not args.output.exists(), "output directory already exists")
    validate_retention_manifest()
    args.protocol_path = args.protocol.resolve()
    protocol = json.loads(args.protocol_path.read_text())
    validate_protocol(protocol)
    require(protocol.get("execution_authorized") is True,
            "protocol has not authorized this new prospective run")
    experiment_seed = int(protocol["workload"]["seed"])
    require(sha(Path(__file__)) == protocol["runner_sha256"], "runner hash differs from protocol")
    require(sha(ROOT / "scripts/audit_verifier_drift_scenarios.py") == protocol["auditor_sha256"],
            "auditor hash differs from protocol")
    require(sha(ROOT / "scripts/benchmark_verifier_consistent_async_grpo.py")
            == protocol["v6_helper_sha256"], "pinned GRPO helper hash differs")
    for relative, digest in protocol["v7_support_sha256"].items():
        require(sha(ROOT / relative) == digest,
                f"v7 support source hash differs: {relative}")
    for relative, digest in protocol["pinned_crash_recovery_files_sha256"].items():
        require(sha(ROOT / relative) == digest,
                f"pinned crash-recovery source hash differs: {relative}")
    source_status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True).splitlines()
    unexpected_status = [line for line in source_status
                         if not line.startswith("?? results/")]
    require(not unexpected_status,
            f"source tree has non-result changes before measured execution: {unexpected_status}")
    source_head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    common_base = subprocess.check_output(
        ["git", "-C", str(ROOT), "merge-base", "HEAD", protocol["base_commit"]], text=True).strip()
    require(common_base == protocol["base_commit"], "source tree is not based on the frozen parent")
    args.output.mkdir(parents=True)
    args.output_created = True
    for filename, digest in protocol["model_files"].items():
        require(sha(args.model_dir / filename) == digest, f"model checksum mismatch: {filename}")
    for relative, digest in protocol["vare_source_sha256"].items():
        require(sha(ROOT / relative) == digest, f"VARE source hash mismatch: {relative}")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "rev-parse", "HEAD"], text=True).strip()
            == protocol["rvl_revision"], "RVL source revision differs")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "status", "--porcelain"], text=True).strip()
            == "M src/rvl_systems/hf_backend.py", "unexpected RVL source modifications")
    require(sha(args.rvl_source / "src/rvl_systems/hf_backend.py")
            == protocol["rvl_backend_sha256"], "RVL backend hash differs")
    require(sha(ROOT / "patches/rvl-contiguous-transition-logprobs-v1.patch")
            == protocol["rvl_patch_sha256"], "RVL patch provenance differs")

    # Load the established runner helpers only after the repository path is set.
    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark_verifier_consistent_async_grpo as helper
    helper.prepare_pinned_imports(args.rvl_source)
    helper.verify_import_origins(args.rvl_source)
    import torch
    import transformers
    import accelerate
    import huggingface_hub
    import safetensors
    import tokenizers
    import platform
    require(platform.python_version() == protocol["versions"]["python"],
            "Python version differs")
    require(torch.__version__ == protocol["versions"]["torch"], "Torch version differs")
    require(transformers.__version__ == protocol["versions"]["transformers"],
            "Transformers version differs")
    for package, module in (("accelerate", accelerate), ("huggingface_hub", huggingface_hub),
                            ("safetensors", safetensors), ("tokenizers", tokenizers)):
        require(module.__version__ == protocol["versions"][package],
                f"{package} version differs")
    require(platform.machine() == "arm64", "run requires the preregistered Apple Silicon host")
    require(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            == 32 * 1024**3, "host memory differs from the 32 GB preflight")
    torch.set_num_threads(1)
    baseline_swap = helper.swap_used_bytes()
    prereg_hash = sha(args.protocol_path)
    result = {
        "status": "RUNNING", "protocol_sha256": prereg_hash,
        "source_commit": source_head, "seed": experiment_seed,
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": "CPU fp32",
                        "threads": 1, "swap_before_bytes": baseline_swap,
                        "model_revision": protocol["model_revision"],
                        "rvl_revision": protocol["rvl_revision"]},
        "scenarios": {},
    }
    write(args.output / "protocol.json", protocol)
    write(args.output / "progress.json", {"status": "RUNNING", "completed_scenarios": []})
    original_scorer = helper.versioned_score
    try:
        for scenario_index, scenario in enumerate(protocol["scenarios"]):
            scenario_id = scenario["id"]
            result["scenarios"][scenario_id] = {}
            arm_order = protocol["arm_orders"][scenario_id]
            for arm_position, arm in enumerate(arm_order):
                result["scenarios"][scenario_id][arm] = await run_arm(
                    args, helper, protocol, scenario, scenario_index, arm, arm_position,
                    baseline_swap)
                arm_summary = result["scenarios"][scenario_id][arm]
                require(arm_summary["wall_s"]
                        <= protocol["resource_limits"]["max_wall_seconds_per_arm"],
                        "arm wall-time budget exceeded")
                require(arm_summary["peak_rss_bytes"] <= protocol["resource_limits"]["max_rss_bytes"],
                        "arm RSS budget exceeded")
                swap_now = helper.swap_used_bytes()
                if baseline_swap is not None and swap_now is not None:
                    require(max(0, swap_now - baseline_swap)
                            <= protocol["resource_limits"]["max_positive_swap_growth_bytes"],
                            "swap growth exceeded predeclared budget")
                write(args.output / "progress.json", {
                    "status": "RUNNING", "completed_scenarios": list(result["scenarios"]),
                    "active_scenario": scenario_id, "active_arm": arm,
                    "arms_complete": len(result["scenarios"][scenario_id]),
                    "elapsed_s": time.perf_counter() - args.started,
                    "swap_usage_bytes": swap_now,
                })
            arm_summaries = result["scenarios"][scenario_id]
            require(len({item["initial_model_tensor_sha256"]
                         for item in arm_summaries.values()}) == 1,
                    "matched arms did not load identical initial model tensors")
            require(len({item["groups"][0]["response_text_sha256"]
                         for item in arm_summaries.values()}) == 1,
                    "matched first-group rollouts differ across arms")
            # Expansion is fixed in advance and contingent only on pilot validity.
            if scenario_index == 0:
                pilot = result["scenarios"][scenario_id]
                naive_group = pilot["naive_async"]["groups"][0]
                vare_group = pilot["vare_async"]["groups"][0]
                gate = (
                    naive_group["disagreements"]["total_disagreements"] > 0
                    and pilot["vare_async"]["store_counts"].get("quarantined", 0) >= GROUP_SIZE
                    and pilot["vare_async"]["committed_updates"] == UPDATES_PER_ARM
                    and len(set(naive_group["scores"])) > 1
                    and all(len(set(update["update_replay"]["advantages"])) > 1
                            for update in pilot["vare_async"]["updates"])
                )
                result["pilot_gate"] = {
                    "passed": gate,
                    "criterion": "nonzero stale label disagreement; complete VARE stale-group quarantine; four actual VARE updates with nonconstant group advantages",
                    "naive_stale_disagreements": naive_group["disagreements"],
                    "vare_stale_group": vare_group["group_id"],
                    "vare_quarantined": pilot["vare_async"]["store_counts"].get("quarantined", 0),
                }
                if not gate:
                    result["status"] = "STOPPED_PILOT_NO_SIGNAL"
                    break

        if result["status"] == "RUNNING":
            result["status"] = "COMPLETE"
        result["wall_s"] = time.perf_counter() - args.started
        require(result["wall_s"] <= protocol["resource_limits"]["max_wall_seconds_total"],
                "total wall-time budget exceeded")
        result["peak_rss_bytes"] = peak_rss_bytes()
        result["swap_after_bytes"] = helper.swap_used_bytes()
        if baseline_swap is not None and result["swap_after_bytes"] is not None:
            result["swap_growth_bytes"] = max(0, result["swap_after_bytes"] - baseline_swap)
            require(result["swap_growth_bytes"]
                    <= protocol["resource_limits"]["max_positive_swap_growth_bytes"],
                    "final swap growth exceeded predeclared budget")
        require(result["peak_rss_bytes"] <= protocol["resource_limits"]["max_rss_bytes"],
                "peak RSS exceeded predeclared budget")
        write(args.output / "summary.json", result)
        write(args.output / "progress.json", {
            "status": result["status"],
            "completed_scenarios": list(result["scenarios"]),
            "elapsed_s": result["wall_s"],
        })
        files = {str(path.relative_to(args.output)): sha(path)
                 for path in args.output.rglob("*") if path.is_file()
                 and path.name != "SHA256.json" and not path.name.endswith(("-wal", "-shm"))}
        write(args.output / "SHA256.json", files)
        print(json.dumps(result, indent=2))
    finally:
        helper.versioned_score = original_scorer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    cli = parser.parse_args()
    cli.output_created = False
    cli.started = time.perf_counter()
    try:
        asyncio.run(asyncio.wait_for(run(cli), timeout=2400))
    except BaseException as error:
        persist_active_evidence_failures(error)
        if cli.output_created and cli.output.exists():
            write(cli.output / "failure.json", {"type": type(error).__name__,
                "message": str(error), "time": time.time()})
            files = {str(path.relative_to(cli.output)): sha(path)
                     for path in cli.output.rglob("*") if path.is_file()
                     and path.name != "SHA256.json" and not path.name.endswith(("-wal", "-shm"))}
            write(cli.output / "SHA256.json", files)
        raise
