import asyncio

import pytest

from vare.trajectory_dispatch import AuditedDispatcher, TrajectoryActionDenied, egress_host_allowed
from vare.trajectory_safety import (
    TrajectoryPolicy, TrustedTrajectoryRecorder, audit_trajectory,
)


def test_exact_host_policy_rejects_confusables():
    allowed = ("example.org",)
    assert egress_host_allowed("https://example.org/data", allowed)
    for target in (
        "https://example.org.evil.test/api",
        "https://evil.test/@example.org/",
        "https://example.org@evil.test/",
        "http://example.org:12345/",
        "https://example.org/#fragment",
        "https://EXAMPLE.ORG.evil.test/",
        "https://example.org/space here",
    ):
        assert not egress_host_allowed(target, allowed), target


def test_trusted_dispatch_records_allowed_calls(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",), ("example.org",))
    path = tmp_path / "trace.jsonl"
    recorder = TrustedTrajectoryRecorder(path, run_id="r1", policy_id="candidate")
    dispatcher = AuditedDispatcher(recorder=recorder, policy=policy)
    seen = []

    async def call():
        seen.append("executed")
        return "ok"

    async def run():
        assert await dispatcher.execute(
            "browser", call, network_url="https://example.org/report",
            source_origin="retrieved_webpage"
        ) == "ok"

    asyncio.run(run())
    dispatcher.close()
    assert seen == ["executed"]
    audit = audit_trajectory(path, run_id="r1", policy_id="candidate", policy=policy)
    assert audit.status == "pass" and audit.event_count == 3


@pytest.mark.parametrize("tool,url", [
    ("unknown_tool", None),
    ("browser", "https://not-allowed.invalid/private"),
])
def test_denied_action_recorded_without_execution(tmp_path, tool, url):
    policy = TrajectoryPolicy("v1", ("browser",), ("example.org",))
    path = tmp_path / "trace.jsonl"
    dispatcher = AuditedDispatcher(
        recorder=TrustedTrajectoryRecorder(path, run_id="r1", policy_id="candidate"),
        policy=policy,
    )
    seen = []

    async def call():
        seen.append(1)

    async def run():
        with pytest.raises(TrajectoryActionDenied):
            await dispatcher.execute(tool, call, network_url=url)

    asyncio.run(run())
    dispatcher.close()
    assert seen == []
    audit = audit_trajectory(path, run_id="r1", policy_id="candidate", policy=policy)
    assert audit.status == "violation"


def test_failed_tool_still_records_result(tmp_path):
    policy = TrajectoryPolicy("v1", ("browser",))
    path = tmp_path / "trace.jsonl"
    dispatcher = AuditedDispatcher(
        recorder=TrustedTrajectoryRecorder(path, run_id="r1", policy_id="candidate"),
        policy=policy,
    )

    async def fail():
        raise ValueError("fixture failure")

    async def run():
        with pytest.raises(ValueError, match="fixture failure"):
            await dispatcher.execute("browser", fail)

    asyncio.run(run())
    dispatcher.close()
    assert audit_trajectory(path, run_id="r1", policy_id="candidate", policy=policy).status == "pass"
