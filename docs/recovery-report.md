# Transactional CPU evaluation recovery: retained evidence

## Falsifiable question

Can a local evaluation coordinator recover from process loss without allowing a superseded worker or changed inputs to authorize a current successful decision? The runner uses SQLite claims/leases and reuses locked grader execution. Source files and graders remain outside live state; their enrollment hashes are persisted. This question concerns correctness under explicit failure injection, not training or model capability.

The [requirements matrix](requirements.md) maps the invariants to implementation/tests. The [recovery contract](recovery.md) explains transaction boundaries, freshness, commands and limitations.

## Frozen measurement

Merge commit `af3c0fa` integrated the recovered control-plane source at `f5c92cf`, moved implementation to canonical `src/vare`, and committed [cpu-recovery-v1](../protocols/cpu_recovery_v1.json) before measurement. It freezes eight cases, three replications, four workers, a three-attempt budget, a 10-second normal lease and a 0.3-second fault lease. The primary criterion requires every declared invariant to hold and every export to replay. Failure would abort with retained artifacts; no protocol retuning or rerun was used for the reported outcome.

The experiment ran on local macOS arm64 / Python 3.9.6 / SQLite 3.54.0, using standard-library execution. A separate Python 3.12.14/pytest suite verifies the restored Python >=3.11 control-plane modules. Neither experiment invokes a model, GPU or paid API. Public source fetch is required for the historical integration check.

## Result

**24/24 replicated fault cases matched the frozen criteria.** The exported synthetic scenarios contain **117 jobs and 126 attempts**. Multiple attempts are expected in expiry/recovery cases; they are retained rather than collapsed into a false exactly-once execution claim.

| Case | Replications matched | Evidence observed |
| --- | ---: | --- |
| Concurrent claims | 3/3 | Four processes finish 32 jobs per replication; each job has token 1 and one committed attempt. |
| Claimant process death | 3/3 | Exit 23 after claim; expired token 1 is retained and token 2 publishes the current pass. |
| Death in a write transaction | 3/3 | Exit 24 before commit; state remains pending and recovery commits token 1. |
| Lost acknowledgement | 3/3 | Exit 25 after commit; restart keeps the original token 1 decision. |
| Late old completion | 3/3 | Token 2 commits; token 1's late outcome is retained as fenced and cannot replace it. |
| Completed result becomes stale | 3/3 | Export disables current success after source mutation while preserving the old passing record. |
| Pending input changes | 3/3 | Inputs are invalidated without launching an attempt. |
| Retry budget exhaustion | 3/3 | Three expired claims produce `retry_exhausted`, with no current candidate success. |

[Raw summary](../results/cpu-recovery-v1/summary.json), [protocol/script snapshots](../results/cpu-recovery-v1/snapshot/) and [first replication](../results/cpu-recovery-v1/replicate-0/) retain events, protected assets, current states, raw streams and attempt records.

The historical integration enrolled the pre-fix/fixed revisions for both calibrated RL tasks, killed one claimant before grading, and resumed with four workers. **All four candidate outcomes matched:** two pre-fix rejections and two fixed acceptances. Five attempts are retained: one expired claim and four completed grades; exactly one job advances to token 2. See [integration summary](../results/cpu-recovery-v1/historical-summary.json) and [raw state/events](../results/cpu-recovery-v1/historical/state.json).

Observed experiment wall time was approximately 60.57 seconds including synthetic cases, process startup and historical source preparation. Wall time is diagnostic, not a throughput or recovery-latency comparison. This protocol has no baseline for a performance-improvement claim.

## Audit and reproduction

Offline reconstruction:

```bash
python3 scripts/audit_recovery.py results/cpu-recovery-v1
python3 -m vare durable-audit results/cpu-recovery-v1/historical
```

The first command checks the experiment manifest, replays every bundle, recalculates case criteria, binds historical revisions/outcomes and checks the headline summary. It should report 24 matched fault cases, 117 synthetic jobs, 126 synthetic attempts, four historical jobs and five historical attempts. This is consistency validation against reviewed artifacts, not independent third-party reproduction or cryptographic authentication.

Full reproduction into a fresh directory:

```bash
python3 scripts/measure_recovery.py --output /tmp/vare-recovery-reproduction
python3 scripts/audit_recovery.py /tmp/vare-recovery-reproduction
```

Use POSIX, Git and Python >=3.9 for the standalone recovery/evaluation path. Source preparation uses network; no model weights are fetched. Timing varies by machine. The full restored package tests require Python >=3.11 and the optional pytest development dependency.

A separate read-only review exposed five counterexamples during development: trusting caller-mutated enrollment, exporting a stale pass without refresh, non-idempotent retransmission after invalidation, an impossible terminal replay transition and a linked database path. They were fixed with narrow regressions before the frozen measurement. After measurement, offline auditing was additionally tightened to reject invalidation without an enrollment event; this does not change the measured worker execution path and is covered by a regression. Measurement snapshots preserve the exact source used.

## Claim boundary

This is same-host transactional coordination and trusted CPU grader fault injection. Leases can cause repeat execution; only current terminal publication is fenced. Owner labels are not authentication. Before/after file hashing does not freeze a malicious writer. Clock movement, network partitions, distributed fleets, machine power loss and crash-time child containment were not tested. Injected process exits occur before grader launch, inside a state transaction or after a completed commit; the study does not claim to kill an untrusted grader safely after a worker's uncatchable death.

The recovered old L0 archive failed hash/decompression checks and is explicitly excluded from verified historical accuracy claims. That negative result and its original bytes are retained separately in the [import audit](../results/restored-evidence-audit-v1/summary.json). Current evidence consists of reproducible grader, scheduling and recovery results with the limits above.

## CI portability regression

The first integrated GitHub CI run failed during test collection because the `pytest` console entry did not include the repository root for benchmark/script namespace imports. The failure log is retained in [`ci-import-regression-v1`](../results/ci-import-regression-v1/). The pytest configuration now includes both canonical `src` and the repository root, and CI invokes `python -m pytest`. Both local invocation styles pass all 65 tests; GitHub run history retains the failed run rather than hiding it. This import-path repair does not change the measured recovery protocol or worker algorithm.
