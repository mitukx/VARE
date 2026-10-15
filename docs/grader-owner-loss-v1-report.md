# Grader owner loss: containment and durable resume

The bounded runner now stops an ordinary grader process group when its owning
evaluation worker disappears, including SIGKILL and `os._exit`. Previously,
only the worker's Python `finally` block killed that group. An uncatchable
worker loss left the grader and its child executing after the durable lease
expired. Fencing stopped stale publication but did not stop that execution.

This is a fresh source-defect regression with real CPU processes, not an agent
task-success, training, or model-capability experiment. Existing retired model
cohorts and confirmation tasks were not used.

## Implementation and ownership

[`execute`](../src/vare/runner.py) launches a small standard-library
[`guard`](../src/vare/_grader_guard.py) as a new POSIX session leader. The
worker owns the write end of its stdin pipe. The guard watches EOF and kills
its own still-owned group; its grader and ordinary descendants share that
group. The grader receives `/dev/null` stdin and does not inherit the private
status descriptor. A separate four-byte pipe preserves the grader's actual
exit status without mixing control data into stdout/stderr.

After reporting grader exit, the guard remains alive until worker cleanup
closes the ownership pipe. It closes its capture descriptors so the existing
EOF requirement still rejects an inherited pipe held by a lingering child.
Normal completion, timeout, output overflow and cancellation all close the
same ownership pipe. Launch and cleanup are drained across cancellation.
The public CLI, accepted grader schema, output budget, lease/token state
machine and existing audit format remain unchanged. New campaign and durable
snapshots include the helper source.

## Frozen comparison

The [protocol](../protocols/grader_owner_loss_v1.lock.json) pins main
`f54c488e272b7dbe513558a2e3489700f5814d35` and hashes one protected
[test file](../tests/test_grader_owner_loss.py). The defect was already observed
before the paired measurement; this is development regression evidence.

Each fresh fixture launches a real durable worker, grader and child with a
heartbeat file. It waits for readiness, kills the worker while grading, checks
both remaining processes stop within two seconds, checks the heartbeat stays
unchanged, resumes the original SQLite store, and independently replays the
export. The replacement must commit token 2 after one expired attempt; the
event sequence must be enrolled/claimed/expired/claimed/committed.

| Worker fault | Pinned main | Guard candidate | Observed stop | Resume wall time |
| --- | --- | --- | --- | --- |
| SIGKILL | grader/child remain running | stopped; token 2 passes | 7.09 ms | 1.057 s |
| SIGTERM | grader/child remain running | stopped; token 2 passes | 7.29 ms | 1.043 s |
| immediate exit 73 | grader/child remain running | stopped; token 2 passes | 7.19 ms | 1.057 s |

All three baseline assertions fail; all three candidate cases pass. Resume
includes waiting for the one-second lease to expire. These three observations
are not a recovery-latency distribution or a service-level guarantee.

Six additional lifecycle tests cover cancellation during launch, repeated
cancellation during cleanup, missing private exit status, and spawn failure
descriptor cleanup, a launch coroutine that cancels itself, and a broken
status pipe after grader exit. Existing runner faults still cover rejection, malformed
output, provenance errors, protected/source changes, timeout, output overflow,
ordinary child cleanup and inherited pipes. Fourteen artifact mutations are
rejected in normal and optimized Python, including correctly rehashed false
stop results, missing fault injection, excessive stop latency, wrong group
ownership and inconsistent cost summaries.

## Descriptive cost, including the regression

Five alternating baseline/candidate pairs each evaluate 16 fresh synthetic
valid jobs with four workers and a 20 ms grader delay. Both arms run the real
validation, subprocess, capture, revalidation and evidence-write path. The
raw measurement wall time includes initial snapshot creation; the nested
campaign's existing wall metric excludes it. Grader outcomes are controls,
not independent coding-task or model scores.

| Median across five runs | Pinned main | Guard candidate |
| --- | ---: | ---: |
| 16-job wall time | 1.0117 s | 1.0977 s |
| throughput | 15.816 jobs/s | 14.576 jobs/s |
| per-job execution p50 | 210.69 ms | 227.65 ms |
| per-job execution p95 | 229.73 ms | 262.13 ms |
| offline campaign audit | 16.65 ms | 16.51 ms |
| controller peak traced allocation | 1,013,552 B | 1,099,034 B |
| controller peak RSS | 30,867,456 B | 29,409,280 B |
| largest reaped child peak RSS | 17,678,336 B | 17,678,336 B |

The final observed wall cost increases **8.50%**. The initial implementation's
five pairs showed **10.07%** more wall time; the cancellation-corrected version
showed **7.13%**. Both remain retained separately. Later runs followed
cancellation and status-pipe correctness fixes under the same frozen
protocol; no timing result was used to select or tune the implementation.
This is a correctness improvement
with overhead, not a speedup. Each active job adds one small guard process.
RSS is a process peak, not simultaneous aggregate memory; child RSS is the
maximum reported by `RUSAGE_CHILDREN`, not a sum. Process startup, Git,
validation, tracing and filesystem costs dominate this short fixture. Five
local pairs do not isolate those costs or establish population intervals.
No accelerator or paid API was used.

## Reproduction and retained failures

```bash
python -m pytest tests/test_grader_owner_loss.py tests/test_grader_guard.py tests/test_runner.py tests/test_durable.py
python scripts/reproduce_grader_owner_loss_v1.py --output /tmp/vare-owner-loss-new
python scripts/audit_grader_owner_loss_v1.py /tmp/vare-owner-loss-new
python -O scripts/audit_grader_owner_loss_v1.py /tmp/vare-owner-loss-new
python -m pytest tests/test_grader_owner_loss_audit.py
```

Run from the checkout with Python >=3.11, Git, POSIX and pytest. Source fetch
is needed only if the pinned main commit is missing. The grader/cost workload
itself is offline. The replay script uses explicit source paths and a temporary
working directory, avoiding the root bootstrap accidentally selecting the
candidate for the baseline. A fresh output directory is required.

The final [comparison archive](../results/grader-owner-loss-v1/comparison-v3.zip) and
[hash/size index](../results/grader-owner-loss-v1/comparison-v3.index.json) retain
both source snapshots, all test records, real process receipts, the three
successful recovery exports, all ten cost campaigns, raw records and hashes.
The [auditor](../scripts/audit_grader_owner_loss_v1.py) does not import the
reproducer or launch graders: it checks exact inventory, pinned baseline
source, protected test hash, fault outcomes and group identities, replays
existing durable/campaign contracts and recomputes latency/throughput/medians.
The archive preserves all 847 original files byte-for-byte in 592,157 bytes;
the auditor accepts a directory or ZIP and rejects traversal, symlinks and
duplicate archive members before extraction. To audit the committed record:
`python scripts/audit_grader_owner_loss_v1.py results/grader-owner-loss-v1/comparison-v3.zip`.
Hashes check consistency relative to reviewed source, not authenticity of an
operator able to rewrite the entire bundle.

The initial baseline failures remain in `development-baseline`. The first
candidate run hit a test-helper error: `ps` returns 1 for an already-gone PID,
which the helper incorrectly treated as an execution failure. That log remains
in `development-candidate/initial-tests.txt`; the helper was corrected before
freezing the protected comparison. Subsequent lifecycle and full-suite logs
are retained alongside the raw comparison. A later injected self-cancellation
of the launch coroutine exposed a busy loop in the initial candidate's shield
retry logic. Its one-second external timeout and the source before correction
remain in `development-candidate`; the fix propagates an already-cancelled
owned task. The [initial comparison](../results/grader-owner-loss-v1/comparison-v1.zip)
and [cancellation-corrected comparison](../results/grader-owner-loss-v1/comparison-v2.zip)
remain unchanged and replayable. A final controlled status-reader closure
reproduced a `BrokenPipeError` that let the first guard exit with a live child;
the guard now kills its group in its own `finally` block on Python errors.
That counterexample and source are retained in `development-candidate` too.
Costs were not used to tune a gate.

The final local source passes **808 default CPU tests** (14 optional-dependency
skips), the separate **14 mathematical grader tests**, Ruff and mypy. The
final comparison replays in normal and optimized Python. Raw test logs/XML,
source hashes and the M1 Pro/32 GB runtime receipt are in the results directory.

## Remaining limits

- This is cooperative POSIX process ownership, not a hostile-code sandbox.
  A descendant that changes session/group escapes this guard.
- Uncatchable death or indefinite suspension of the guard itself is not
  contained. The missing-status check fails closed but does not repair an
  externally killed guard's descendants.
- Unrelated forked processes must not inherit and keep the ownership writer
  open. The built-in durable launcher uses `spawn`; operator-created raw forks
  during an active evaluation are outside the tested ownership contract.
- Process stopping does not undo effects already performed. Evaluation remains
  at least once, with fenced terminal publication; no exactly-once effects are
  claimed. Kernel scheduling, machine power loss and process escape are untested.
- Keep the enrolled runner revision installed when resuming a private live
  database. Source assets are captured at enrollment; this change does not add
  runtime upgrade/migration or attest code changes during an existing job.
- This does not connect durable grading to every agent campaign or establish
  independently improved model task success. Those remain separate gaps.
