# Durable evaluator source pinning

Durable workers now refuse a declared executor source bundle that differs from
the bytes enrolled with their live database. Previously, a worker could resume
with changed code, commit a result, and export the original executor snapshot
as if it described the execution. The existing consistency auditor accepted
that export. This is a source-provenance defect in trusted local operation,
not a model-capability finding.

## Existing boundary and correction

[`Store.create`](../src/vare/durable.py) already captures five executor files:
`__init__.py`, `__main__.py`, `runner.py`, `durable.py`, and `_grader_guard.py`.
The correction reuses those assets and the runner's existing bounded file-hash
check. No new database schema, migration system, worker pool or grading loop
was added.

The source check runs before worker refresh, inside the claim transaction,
before grader dispatch, and before completion admission. Missing/extra source
assets, malformed stored bytes, unreadable source or a hash mismatch raise
`RuntimeMismatchError`. Claim/attempt/event state stays unchanged on rejection;
no enrolled snapshot is silently replaced. Operators can still inspect/export
historical state using a newer installation. Resume requires the enrolled
revision, or explicit enrollment in a new state file for a reviewed new runtime.

The completion check also protects direct `Store.complete` callers. It cannot
undo grading or side effects already performed before a mismatch is observed.
Existing lease expiry, generation fencing, idempotent completion and protected
grader/source checks remain in force.

## Frozen source comparison

The [protocol](../protocols/durable_runtime_source_v1.lock.json) pins merged main
`58ef14e7a6ffc9a763de14e95cfb7527fcb3ece4` and the SHA-256 of the protected
[regression file](../tests/test_durable_runtime_source.py). This is development
regression of an already observed defect, frozen before paired costs. It does
not consume a model confirmation cohort.

| Cases | Pinned main | Correction |
| --- | ---: | ---: |
| Each of five enrolled executor assets is corrupted to SQL text, at claim and completion | 10 failures | 10 passes |
| Missing/extra executor asset | 2 failures | 2 passes |
| Worker rejects before refresh/grading | 1 failure | 1 pass |
| Actual process crash, changed-code resume, original-code restoration | 1 failure | 1 pass |
| Actual process crash, unchanged runtime control | 1 pass | 1 pass |
| Total | 14 failures, 1 pass | 15 passes |

The process case uses separate source directories and real Python processes.
Runtime A enrolls the job and exits 42 after claiming it. Runtime B differs by
an executable diagnostic assignment in `runner.py`; its output confirms the
changed module loaded. On main, B grades successfully at token 2 while its
export contains A's runner bytes. The two runner hashes disagree, even though
the original auditor reports a consistent passing export.

With the correction, B is refused at token 1 with only enrolled/claimed events.
Restarting A then expires the first attempt, executes the real protected grader
at token 2 and commits one result. The retained used/exported runner hashes now
agree. The unchanged control also resumes to token 2. SQLite and the original
grader/auditor execute throughout; these are synthetic correctness controls,
not a learned agent discovering a patch or a model solving a new coding task.

The frozen corruption helper uses SQLite concatenation, which produces TEXT
rather than BLOB values. Those ten cases exercise malformed-asset refusal.
Five supplemental [BLOB-preserving checks](../tests/test_durable_runtime_source_hashes.py)
exercise a genuine differing source hash for every declared file and require
the specific mismatch diagnostic. The actual changed-runtime process case
also uses well-formed original BLOB enrollment. The frozen test/protocol and
first measurement remain unchanged.

## CPU costs

Five alternating pairs run 16 fresh valid fixture jobs through the existing
four-process `run_workers` entry point, with 20 ms grader delay and 10-second
leases. Wall time covers worker startup, initial freshness scans, claims,
source checks, grading, capture, revalidation and publication. Enrollment,
export and offline replay are measured separately. Each job commits once.

| Median across five runs | Main | Correction |
| --- | ---: | ---: |
| 16-job worker wall time | 1.9490 s | 1.9675 s |
| throughput | 8.209 jobs/s | 8.132 jobs/s |
| grader execution p50 | 198.73 ms | 199.29 ms |
| grader execution p95 | 207.45 ms | 213.30 ms |
| export | 530.93 ms | 570.37 ms |
| offline audit | 20.37 ms | 20.63 ms |
| controller traced allocation peak | 310,834 B | 319,423 B |
| controller RSS peak | 32,079,872 B | 29,835,264 B |
| largest reaped child RSS peak | 33,341,440 B | 31,457,280 B |

Observed wall time increases **0.95%**. This is consistency enforcement with
overhead, not a throughput or memory improvement. Grader execution latency
excludes checks performed outside `evaluate`; total worker wall time includes
them. Traced allocation covers the controller, not worker heaps. RSS values
are individual process peaks rather than concurrent aggregate memory. The
export difference and RSS variation do not isolate effects of this patch.
These short local M1 Pro/32 GB samples have no population interval. No GPU,
model download, model inference, training or paid API was used.

## Reproduction and retained evidence

```bash
python -m pytest tests/test_durable_runtime_source.py tests/test_durable.py
python scripts/reproduce_durable_runtime_source_v1.py --output /tmp/vare-runtime-pin-new
python scripts/audit_durable_runtime_source_v1.py /tmp/vare-runtime-pin-new
python -O scripts/audit_durable_runtime_source_v1.py /tmp/vare-runtime-pin-new
python -m pytest tests/test_durable_runtime_source_audit.py
```

Run from the checkout with Python >=3.11, pytest, Git and POSIX. A fresh output
directory is required. Fetching an absent pinned baseline needs Git access;
the evaluator workload and retained artifact replay are offline.

The [archive](../results/durable-runtime-source-v1/comparison-v1.zip) and
[hash/size index](../results/durable-runtime-source-v1/comparison-v1.index.json)
retain 1,035 raw files in 810,728 bytes. Every member was compared byte-for-byte
with the raw directory before packaging. Both source bundles, all tests and
process outputs, rejected/incomplete and restored exports, and all ten cost
exports are retained. Initial failing tests remain in `development-baseline`.
The [auditor](../scripts/audit_durable_runtime_source_v1.py) does not import the
reproducer or execute graders. It pins all five baseline source hashes without
Git history, replays the existing durable audit, checks actual process outputs
and used/exported identities, and recomputes latency, throughput and medians.

Eighteen fault cases (16 with recomputed manifests, two inventory faults)
reject forged refusal/success, wrong used or
restored source, changed baseline source, wrong cost-runtime identity, Boolean
tokens, duplicate JSON fields and inconsistent summaries/inventory. Two
positive checks hide Git/history entirely. All run in normal and optimized
Python. Hashes establish consistency relative to reviewed assets; they do not
authenticate an operator able to rewrite the database and reviewed source.

## Limits

This pins the five **declared source files** under stable trusted installation
and database ownership. It does not attest loaded bytecode, monkeypatches,
transient changes restored between checks, `_public.py`/other unlisted imports,
Python/native dependency versions, or the truthfulness of an arbitrary worker.
It is not a hostile-code sandbox or automatic runtime-upgrade protocol. Old
exports remain auditable; older live stores require their original runtime.
No model-learning or task-success gap is closed by these fixtures.

Final local validation: **888 CPU tests pass** (14 optional-dependency skips),
the separate **14 mathematical grader tests pass**, and Ruff/mypy pass. Normal
and optimized artifact replay both pass. Logs/XML, source hashes and initial
failures are retained in the results directory; the 235 previously allowlisted
large blobs remain unchanged.
