# Owned coding-agent subprocess boundary (v1)

## Problem and scope

The former `CommandWorkspaceAgent` launched a blocking `subprocess.run` in
`asyncio.to_thread`. Cancelling the awaiting asyncio task did not terminate
the external coding agent. The agent could continue mutating a candidate
workspace after an executor lease expired or after the independent verifier
started. This is a correctness gap, independent of model quality.

## Implementation

`_run_owned_command` (in `src/vare/environments/campaign.py`) reuses VARE's existing `_grader_guard.py` ownership
process, with its dedicated process group and parent-pipe EOF signal. The
existing command adapter keeps its argument placeholder and environment
allowlist semantics. Child output is drained concurrently, total output is
limited to 262,144 bytes, and retained stdout/stderr are capped at 100,000
bytes each. Completion reports the child's status rather than the guard's.

The async owner closes its pipe on completion, timeout or cancellation; the
guard kills ordinary descendants. If the guard does not exit, a three-second
cleanup bound falls back to process-group SIGKILL. The implementation awaits
cleanup even during repeated cancellation. It does not claim exactly-once
external side effects, and a child escaping the process group remains outside
the trust boundary. POSIX is required.

## Focused regressions

`python -m pytest tests/test_command_agent_ownership.py tests/test_environment_campaign.py`

Tests cover cancellation, timeout, descendant liveness, output overflow and
status capture. The two descendant tests use a deliberately delayed file write
as an observable side effect: it must not appear after termination.

## Evidence and limitations

No model run, task success, local CPU benchmark, or performance improvement is
claimed by this PR. Existing synthetic campaign results remain synthetic.
Agent/tool event-by-event recovery using `TrajectoryStore`, real model
inference, and independently graded successful coding trajectories remain
outside this bounded correctness change. Full acceptance of Verified MLSys
Agent Baseline v1 therefore remains **NO-GO** pending these integrations.

This PR changes process ownership, not the evaluator trust model. Same-UID
processes, session escape, malicious network activity and container isolation
must be addressed separately before running untrusted agent code.
