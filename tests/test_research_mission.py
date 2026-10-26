from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vare.research_mission import (
    IntegrityError,
    LeaseError,
    MissionError,
    MissionJournal,
    SourceMismatchError,
    classify_failure,
)


H = "a" * 64
P = "b" * 64
M = "c" * 64
E = "d" * 64
RESULT_BYTES = b"mission journal test action result\n"
RESULT = hashlib.sha256(RESULT_BYTES).hexdigest()


@pytest.fixture
def mission(tmp_path: Path) -> tuple[MissionJournal, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "VARE test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "vare-test@example.invalid"],
        check=True,
    )
    (repo / "source.py").write_text("stable\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "source identity"], check=True
    )
    db = tmp_path / "state" / "mission.sqlite3"
    journal = MissionJournal.create(
        db,
        mission_id="mission-test",
        repo_root=repo,
        budget={
            "memory_bytes": 1024,
            "disk_bytes": 2048,
            "model_tokens": 100,
            "wall_seconds": 1000,
            "candidate_attempts": 2,
        },
    )
    return journal, repo, db


def register_dev(
    journal: MissionJournal, *, task_id: str = "dev-1", exposure: str = "unseen"
) -> None:
    journal.register_task(
        "mission-test",
        task_id=task_id,
        family_id=f"family-{task_id}",
        split="development",
        eligibility="eligible" if exposure == "unseen" else "qualified",
        exposure=exposure,
        source_identity="upstream@immutable-rev",
        grader_sha256=H,
        certificate_sha256=P,
        exposure_audit_sha256=M if exposure == "unseen" else None,
    )


def add_probe(
    journal: MissionJournal, action_id: str = "probe", *, request=None, retry_safe=False
):
    return journal.add_action(
        "mission-test",
        action_id=action_id,
        idempotency_key=f"idem-{action_id}",
        kind="candidate_discovery",
        phase="DISCOVER",
        retry_safe=retry_safe,
        request=request or {},
    )


def add_model_attempt(journal: MissionJournal, task_id: str = "dev-1"):
    return journal.add_action(
        "mission-test",
        action_id=f"attempt-{task_id}",
        idempotency_key=f"idemp-attempt-{task_id}",
        kind="model_attempt",
        phase="EXECUTE",
        task_id=task_id,
        protocol_sha256=P,
        provenance_sha256=M,
        request={"memory_bytes": 512, "model_tokens": 80, "wall_seconds": 300},
    )


def complete_action(
    journal: MissionJournal,
    action_id: str,
    *,
    owner: str,
    generation: int,
    now: float,
    result_bytes: bytes = RESULT_BYTES,
    actual_resources: dict[str, int] | None = None,
) -> None:
    digest = hashlib.sha256(result_bytes).hexdigest()
    ref = Path(f"{journal.path}.{digest}.artifact")
    try:
        with ref.open("xb") as handle:
            handle.write(result_bytes)
    except FileExistsError:
        assert ref.read_bytes() == result_bytes
    journal.complete(
        "mission-test",
        action_id,
        owner=owner,
        generation=generation,
        result_sha256=digest,
        result_ref=str(ref),
        actual_resources=actual_resources
        or {
            key: 0
            for key in (
                "memory_bytes",
                "disk_bytes",
                "model_tokens",
                "wall_seconds",
                "candidate_attempts",
            )
        },
        now=now,
    )


def advance_phase(
    journal: MissionJournal, expected_phase: str, next_phase: str
) -> None:
    contents = json.dumps(
        {
            "from": expected_phase,
            "to": next_phase,
            "fixture_decision": "advance_after_completed_action",
        },
        sort_keys=True,
    ).encode("utf-8")
    reference = Path(
        f"{journal.path}.{expected_phase.lower()}-{next_phase.lower()}.decision.json"
    )
    try:
        with reference.open("xb") as handle:
            handle.write(contents)
    except FileExistsError:
        assert reference.read_bytes() == contents
    journal.advance(
        "mission-test",
        expected_phase=expected_phase,
        next_phase=next_phase,
        decision_sha256=hashlib.sha256(contents).hexdigest(),
        decision_ref=str(reference),
    )


def enter_execute(journal: MissionJournal) -> None:
    for phase, next_phase in (
        ("DISCOVER", "SCREEN"),
        ("SCREEN", "QUALIFY"),
        ("QUALIFY", "FREEZE"),
        ("FREEZE", "EXECUTE"),
    ):
        action_id = f"setup-{phase.lower()}"
        journal.add_action(
            "mission-test",
            action_id=action_id,
            idempotency_key=action_id,
            kind="fixture",
            phase=phase,
        )
        action = journal.claim_next(
            "mission-test", owner="setup", now=100, lease_seconds=100
        )
        assert action["action_id"] == action_id
        complete_action(journal, action_id, owner="setup", generation=1, now=101)
        advance_phase(journal, phase, next_phase)


@pytest.mark.parametrize(
    (
        "failure_class",
        "attempt",
        "streak",
        "exposed",
        "retry_safe",
        "status",
        "next_action",
        "consumed",
    ),
    [
        ("A", 1, 1, False, True, "queued", "retry_idempotent_operation", False),
        ("A", 3, 1, False, True, "failed", "escalate_after_retry_exhaustion", False),
        ("A", 2, 3, False, True, "blocked", "redesign_or_change_direction", False),
        (
            "A",
            1,
            1,
            False,
            False,
            "uncertain",
            "reconcile_external_state_before_retry",
            False,
        ),
        ("A", 1, 1, True, True, "consumed", "continue_on_new_task_family", True),
        (
            "B",
            1,
            1,
            False,
            False,
            "superseded",
            "new_implementation_version_and_regressions",
            False,
        ),
        ("C", 1, 1, False, False, "consumed", "new_protocol_and_unconsumed_task", True),
        (
            "D",
            1,
            1,
            False,
            False,
            "consumed",
            "failure_analysis_then_fresh_development_cohort",
            True,
        ),
        (
            "E",
            1,
            1,
            False,
            False,
            "blocked",
            "schedule_independent_admissible_work",
            False,
        ),
    ],
)
def test_bounded_retry_policy_classifies_a_through_e(
    failure_class,
    attempt,
    streak,
    exposed,
    retry_safe,
    status,
    next_action,
    consumed,
):
    decision = classify_failure(
        failure_class,
        attempt_number=attempt,
        same_root_cause_streak=streak,
        model_exposure_started=exposed,
        operation_idempotent=retry_safe,
    )
    assert decision.action_status == status
    assert decision.next_action == next_action
    assert decision.consumed_outcome is consumed


def test_retry_policy_rejects_unknown_or_unbounded_inputs():
    with pytest.raises(MissionError, match="A, B, C, D, or E"):
        classify_failure("F", attempt_number=1)
    with pytest.raises(MissionError, match="1..3"):
        classify_failure("A", attempt_number=1, max_attempts=100)
    with pytest.raises(MissionError, match="positive integer"):
        classify_failure("A", attempt_number=0)


def test_mission_creation_never_overwrites_existing_state(mission):
    journal, repo, db = mission
    before = db.read_bytes()
    with pytest.raises(MissionError, match="already exists"):
        MissionJournal.create(db, mission_id="replacement", repo_root=repo)
    assert db.read_bytes() == before
    assert journal.audit("mission-test")["event_count"] == 1


def test_task_and_action_identity_are_immutable_and_idempotent(mission):
    journal, _, _ = mission
    register_dev(journal)
    row = add_probe(journal)
    duplicate = add_probe(journal)
    assert row["identity_sha256"] == duplicate["identity_sha256"]
    with pytest.raises(MissionError, match="idempotency key was reused"):
        journal.add_action(
            "mission-test",
            action_id="different",
            idempotency_key="idem-probe",
            kind="candidate_discovery",
            phase="DISCOVER",
        )
    with pytest.raises(MissionError, match="task identity is immutable"):
        journal.register_task(
            "mission-test",
            task_id="dev-1",
            family_id="changed",
            split="development",
            eligibility="eligible",
            exposure="unseen",
            source_identity="upstream@immutable-rev",
            grader_sha256=H,
            certificate_sha256=P,
            exposure_audit_sha256=M,
        )


def test_task_provenance_record_is_hash_checked_on_registration_and_audit(mission):
    journal, _, _ = mission
    reference = journal.path.parent / "task-record.json"
    contents = b'{"task":"immutable-source-and-grader"}\n'
    reference.write_bytes(contents)
    digest = hashlib.sha256(contents).hexdigest()
    journal.register_task(
        "mission-test",
        task_id="dev-record",
        family_id="family-dev-record",
        split="development",
        eligibility="candidate",
        exposure="unknown",
        source_identity="upstream@immutable-rev",
        record_ref=str(reference),
        record_sha256=digest,
    )
    journal.audit("mission-test")
    reference.write_bytes(contents + b"tampered")
    with pytest.raises(IntegrityError, match="do not match"):
        journal.audit("mission-test")


def test_training_and_heldout_actions_fail_closed(mission):
    journal, _, _ = mission
    with pytest.raises(MissionError, match="training is disabled"):
        journal.add_action(
            "mission-test",
            action_id="training",
            idempotency_key="training",
            kind="training",
            phase="EXECUTE",
            protocol_sha256=P,
            provenance_sha256=M,
        )
    for index in range(15):
        journal.register_task(
            "mission-test",
            task_id=f"heldout-{index}",
            family_id=f"heldout-family-{index}",
            split="heldout",
            eligibility="eligible",
            exposure="unseen",
            source_identity=f"upstream@rev-{index}",
            grader_sha256=H,
            certificate_sha256=P,
            exposure_audit_sha256=M,
        )
    with pytest.raises(MissionError, match="single-use evaluation exposure adapter"):
        journal.add_action(
            "mission-test",
            action_id="heldout-eval",
            idempotency_key="heldout-eval",
            kind="heldout_evaluation",
            phase="EXECUTE",
            protocol_sha256=P,
            provenance_sha256=M,
        )


def test_model_attempt_must_be_queued_for_execute_phase(mission):
    journal, _, _ = mission
    register_dev(journal)
    with pytest.raises(MissionError, match="EXECUTE phase"):
        journal.add_action(
            "mission-test",
            action_id="wrong-phase",
            idempotency_key="wrong-phase",
            kind="model_attempt",
            phase="DISCOVER",
            task_id="dev-1",
            protocol_sha256=P,
            provenance_sha256=M,
        )


def test_only_eligible_unseen_development_tasks_admit_model_actions(mission):
    journal, _, _ = mission
    journal.register_task(
        "mission-test",
        task_id="heldout-1",
        family_id="heldout-family-1",
        split="heldout",
        eligibility="eligible",
        exposure="unseen",
        source_identity="upstream@rev",
        grader_sha256=H,
        certificate_sha256=P,
        exposure_audit_sha256=M,
    )
    with pytest.raises(MissionError, match="Development task"):
        add_model_attempt(journal, "heldout-1")
    journal.register_task(
        "mission-test",
        task_id="dev-unknown",
        family_id="dev-family-unknown",
        split="development",
        eligibility="qualified",
        exposure="unknown",
        source_identity="upstream@rev",
        grader_sha256=H,
        certificate_sha256=P,
    )
    with pytest.raises(MissionError, match="Development task"):
        add_model_attempt(journal, "dev-unknown")
    with pytest.raises(MissionError, match="15 unseen eligible"):
        journal.add_action(
            "mission-test",
            action_id="heldout-eval",
            idempotency_key="heldout-eval-key",
            kind="heldout_evaluation",
            phase="EXECUTE",
            protocol_sha256=P,
            provenance_sha256=M,
        )


def test_model_exposure_is_durable_and_consumes_family_before_inference(mission):
    journal, _, db = mission
    register_dev(journal)
    add_model_attempt(journal)
    claimed = journal.claim_next(
        "mission-test", owner="agent-A", now=100, lease_seconds=100
    )
    assert (
        claimed is None
    )  # model action is queued in EXECUTE, current phase is DISCOVER
    # The gate cannot be reached without completed evidence in each preceding phase.
    add_probe(journal, "discover", retry_safe=True)
    claimed = journal.claim_next(
        "mission-test", owner="agent-A", now=100, lease_seconds=100
    )
    assert claimed["action_id"] == "discover"
    complete_action(journal, "discover", owner="agent-A", generation=1, now=101)
    advance_phase(journal, "DISCOVER", "SCREEN")
    with pytest.raises(MissionError, match="not in the frozen"):
        advance_phase(journal, "SCREEN", "EXECUTE")
    # Continue evidence-backed phase transitions; no external work is invoked.
    phase = "SCREEN"
    for next_phase in ("QUALIFY", "FREEZE", "EXECUTE"):
        action_id = f"evidence-{phase.lower()}"
        journal.add_action(
            "mission-test",
            action_id=action_id,
            idempotency_key=action_id,
            kind="fixture_audit",
            phase=phase,
        )
        action = journal.claim_next(
            "mission-test", owner="agent-A", now=101, lease_seconds=100
        )
        assert action["action_id"] == action_id
        complete_action(journal, action_id, owner="agent-A", generation=1, now=102)
        advance_phase(journal, phase, next_phase)
        phase = next_phase
    action = journal.claim_next(
        "mission-test", owner="agent-A", now=102, lease_seconds=100
    )
    assert action["action_id"] == "attempt-dev-1"
    journal.start_model_exposure(
        "mission-test", action["action_id"], owner="agent-A", generation=1, now=103
    )
    reopened = MissionJournal(db)
    with pytest.raises(MissionError, match="already has a model attempt"):
        reopened.add_action(
            "mission-test",
            action_id="attempt-again",
            idempotency_key="another-attempt",
            kind="model_attempt",
            phase="EXECUTE",
            task_id="dev-1",
            protocol_sha256=P,
            provenance_sha256=M,
        )
    audit = reopened.audit("mission-test")
    assert audit["spent"]["candidate_attempts"] == 1
    assert audit["task_counts"] == [
        {
            "split": "development",
            "eligibility": "eligible",
            "exposure": "consumed",
            "n": 1,
        }
    ]
    assert audit["event_count"] >= 10


def test_fenced_lease_rejects_stale_completion_after_expiry(mission):
    journal, _, _ = mission
    add_probe(journal, retry_safe=True)
    first = journal.claim_next(
        "mission-test", owner="worker-1", now=100, lease_seconds=5
    )
    assert first["lease_generation"] == 1
    # An expired idempotent action is recovered, but a bounded backoff applies.
    assert (
        journal.claim_next("mission-test", owner="worker-2", now=106, lease_seconds=5)
        is None
    )
    second = journal.claim_next(
        "mission-test", owner="worker-2", now=137, lease_seconds=5
    )
    assert second["lease_generation"] == 2
    with pytest.raises(LeaseError):
        complete_action(journal, "probe", owner="worker-1", generation=1, now=138)
    complete_action(journal, "probe", owner="worker-2", generation=2, now=138)
    complete_action(journal, "probe", owner="worker-2", generation=2, now=139)
    with pytest.raises(LeaseError):
        complete_action(
            journal,
            "probe",
            owner="worker-2",
            generation=2,
            now=139,
            result_bytes=b"conflicting action result",
        )
    assert journal.audit("mission-test")["action_counts"] == {"completed": 1}


def test_completion_requires_matching_artifact_and_explicit_usage(mission):
    journal, _, _ = mission
    add_probe(journal)
    action = journal.claim_next(
        "mission-test", owner="worker", now=100, lease_seconds=30
    )
    with pytest.raises(MissionError, match="explicitly report every resource"):
        journal.complete(
            "mission-test",
            action["action_id"],
            owner="worker",
            generation=1,
            result_sha256=RESULT,
            result_ref=str(Path(f"{journal.path}.{RESULT}.artifact")),
            now=101,
        )
    artifact_ref = Path(f"{journal.path}.{RESULT}.artifact")
    artifact_ref.write_bytes(RESULT_BYTES)
    all_zero = {
        "memory_bytes": 0,
        "disk_bytes": 0,
        "model_tokens": 0,
        "wall_seconds": 0,
        "candidate_attempts": 0,
    }
    with pytest.raises(IntegrityError, match="do not match"):
        journal.complete(
            "mission-test",
            action["action_id"],
            owner="worker",
            generation=1,
            result_sha256="f" * 64,
            result_ref=str(artifact_ref),
            actual_resources=all_zero,
            now=101,
        )
    complete_action(journal, action["action_id"], owner="worker", generation=1, now=101)
    Path(f"{journal.path}.{RESULT}.artifact").write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="artifact bytes"):
        journal.audit("mission-test")


def test_sigkill_after_model_exposure_recovers_as_consumed(mission):
    journal, _, db = mission
    register_dev(journal)
    enter_execute(journal)
    action = add_model_attempt(journal)
    claim = journal.claim_next("mission-test", owner="crash-worker", lease_seconds=120)
    assert claim["action_id"] == action["action_id"]
    script = (
        "import os, signal, sys\n"
        "from vare.research_mission import MissionJournal\n"
        "journal = MissionJournal(sys.argv[1])\n"
        "journal.start_model_exposure('mission-test', sys.argv[2], "
        "owner='crash-worker', generation=1)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    killed = subprocess.run(
        [sys.executable, "-c", script, str(db), action["action_id"]],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert killed.returncode == -signal.SIGKILL
    recovered = MissionJournal(db)
    before_expiry = recovered.audit("mission-test")
    assert before_expiry["spent"]["candidate_attempts"] == 1
    assert before_expiry["task_counts"][0]["exposure"] == "consumed"
    resumed = recovered.resume("mission-test", now=time.time() + 121)
    assert resumed["next_action"] is None
    assert resumed["resources"]["spent"]["model_tokens"] == 80


def test_uncertain_non_idempotent_lease_is_never_blindly_retried(mission):
    journal, _, _ = mission
    add_probe(journal, retry_safe=False)
    journal.claim_next("mission-test", owner="worker-1", now=100, lease_seconds=5)
    assert (
        journal.claim_next("mission-test", owner="worker-2", now=106, lease_seconds=5)
        is None
    )
    resumed = journal.resume("mission-test", now=107)
    assert resumed["next_action"] is None
    assert (
        resumed["recommendations"][0]["recommendation"]
        == "reconcile_external_state_before_retry"
    )
    with pytest.raises(MissionError, match="uncertain"):
        advance_phase(journal, "DISCOVER", "SCREEN")


def test_reward_failure_after_model_start_is_consumed_and_charged(mission):
    journal, _, _ = mission
    register_dev(journal)
    action = add_model_attempt(journal)
    # Queue in the correct phase through audited, completed phase actions.
    for phase, next_phase in (
        ("DISCOVER", "SCREEN"),
        ("SCREEN", "QUALIFY"),
        ("QUALIFY", "FREEZE"),
        ("FREEZE", "EXECUTE"),
    ):
        aid = f"phase-{phase}"
        journal.add_action(
            "mission-test",
            action_id=aid,
            idempotency_key=aid,
            kind="fixture",
            phase=phase,
        )
        assert (
            journal.claim_next(
                "mission-test", owner="setup", now=200, lease_seconds=100
            )["action_id"]
            == aid
        )
        complete_action(journal, aid, owner="setup", generation=1, now=201)
        advance_phase(journal, phase, next_phase)
    assert (
        journal.claim_next("mission-test", owner="model", now=210, lease_seconds=100)[
            "action_id"
        ]
        == action["action_id"]
    )
    journal.start_model_exposure(
        "mission-test", action["action_id"], owner="model", generation=1, now=211
    )
    decision = journal.fail(
        "mission-test",
        action["action_id"],
        owner="model",
        generation=1,
        failure_class="D",
        root_cause="bad_patch",
        detail="model output was incorrect",
        actual_resources={
            "memory_bytes": 0,
            "disk_bytes": 0,
            "model_tokens": 0,
            "wall_seconds": 0,
            "candidate_attempts": 0,
        },
        now=212,
    )
    assert decision.action_status == "consumed"
    assert decision.retry_same_operation is False
    audit = journal.audit("mission-test")
    assert audit["spent"]["candidate_attempts"] == 1
    assert (
        audit["spent"]["model_tokens"] == 80
    )  # unknown partial use is charged at the full reservation
    assert add_model_attempt(journal)["status"] == "consumed"


def test_phase_decision_artifact_is_retained_and_rechecked(mission):
    journal, _, _ = mission
    enter_execute(journal)
    reference = Path(f"{journal.path}.discover-screen.decision.json")
    assert reference.is_file()
    reference.write_bytes(reference.read_bytes() + b"tampered")
    with pytest.raises(IntegrityError, match="do not match"):
        journal.audit("mission-test")


def test_resource_reservations_are_serialized_and_budgeted(mission):
    journal, _, _ = mission
    add_probe(journal, "a", request={"memory_bytes": 700})
    add_probe(journal, "b", request={"memory_bytes": 700})
    first = journal.claim_next(
        "mission-test", owner="worker-a", now=100, lease_seconds=30
    )
    second = journal.claim_next(
        "mission-test", owner="worker-b", now=100, lease_seconds=30
    )
    assert first is not None and second is None
    assert (
        journal.resume("mission-test", now=101)["resources"]["reserved"]["memory_bytes"]
        == 700
    )
    complete_action(
        journal,
        first["action_id"],
        owner="worker-a",
        generation=1,
        now=101,
        actual_resources={
            "memory_bytes": 700,
            "disk_bytes": 0,
            "model_tokens": 0,
            "wall_seconds": 0,
            "candidate_attempts": 0,
        },
    )
    assert (
        journal.claim_next("mission-test", owner="worker-b", now=102, lease_seconds=30)
        is None
    )
    assert journal.audit("mission-test")["spent"]["memory_bytes"] == 700


def test_concurrent_claims_have_one_effective_fenced_owner(mission):
    journal, _, db = mission
    add_probe(journal)
    other = MissionJournal(db)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        calls = [
            pool.submit(
                j.claim_next, "mission-test", owner=owner, now=100, lease_seconds=30
            )
            for j, owner in ((journal, "worker-a"), (other, "worker-b"))
        ]
    results = [future.result() for future in calls]
    claims = [result for result in results if result is not None]
    assert len(claims) == 1
    assert claims[0]["lease_generation"] == 1


def test_state_tampering_and_source_mismatch_fail_closed(mission):
    journal, repo, db = mission
    add_probe(journal)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE actions SET status='completed' WHERE action_id='probe'")
    with pytest.raises(IntegrityError, match="current mission state"):
        journal.audit("mission-test")

    # A separate pristine journal demonstrates source identity failure without altering the first record.
    repo2 = repo.parent / "repo2"
    repo2.mkdir()
    subprocess.run(["git", "init", "-q", str(repo2)], check=True)
    subprocess.run(
        ["git", "-C", str(repo2), "config", "user.name", "VARE test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo2), "config", "user.email", "vare-test@example.invalid"],
        check=True,
    )
    (repo2 / "source.py").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo2), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo2), "commit", "-qm", "source"], check=True)
    second = MissionJournal.create(
        repo.parent / "second.sqlite3", mission_id="second", repo_root=repo2
    )
    (repo2 / "source.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(MissionError, match="checkout must be clean"):
        second.audit("second")
    subprocess.run(["git", "-C", str(repo2), "checkout", "--", "source.py"], check=True)
    (repo2 / "extra.py").write_text("new commit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo2), "add", "extra.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo2), "commit", "-qm", "new source revision"], check=True
    )
    with pytest.raises(SourceMismatchError, match="source identity"):
        second.audit("second")


def test_event_chain_and_database_survive_reopen(mission):
    journal, _, db = mission
    add_probe(journal)
    reopened = MissionJournal(db)
    assert reopened.audit("mission-test")["event_count"] == 2
    assert reopened.resume("mission-test")["next_action"]["action_id"] == "probe"


def test_tampered_event_hash_is_rejected(mission):
    journal, _, db = mission
    add_probe(journal)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE events SET event_sha256=? WHERE sequence=0", ("f" * 64,))
    with pytest.raises(IntegrityError, match="event hash mismatch"):
        journal.audit("mission-test")


def test_entrypoint_exposes_resume_command(monkeypatch, capsys, mission):
    journal, _, db = mission
    add_probe(journal)
    from vare import entrypoint

    monkeypatch.setattr(
        "sys.argv",
        ["vare", "research", "resume", "--db", str(db), "--mission-id", "mission-test"],
    )
    assert entrypoint.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["executor"].startswith("not_configured")
    assert payload["next_action"]["action_id"] == "probe"
