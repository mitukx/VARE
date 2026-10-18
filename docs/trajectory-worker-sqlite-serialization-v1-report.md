# Same-process trajectory worker SQLite contention

## Finding

On the pinned M1 Pro host, the existing independent execute/verify/consume
stages test hung on all three baseline attempts when several
`asyncio.to_thread` calls opened and closed connections through one
`TrajectoryStore`. A per-instance reentrant lock now covers each SQLite
connection's full context lifetime. The same test completed in all three
candidate attempts.

The change is scoped to one `TrajectoryStore` object in one process. It does
not coordinate separate store objects or processes. The lock serializes DB
access for that object, so concurrency and throughput tradeoffs under larger
workloads remain unmeasured. The existing runtime-source pin also means an
active store enrolled under the prior `trajectory_store.py` bytes will reject
normal resume under this revision; no cross-revision store migration is
implemented here. Read-only evidence audit remains separate from resume.

## Locked comparison

The v2 lock pins baseline `40db644c7eae4d830dc721e1d8185665229cfbd6`, candidate
`3f546253b7df26799ce65098f4c62f7d6132d263`, the regression test, and runner by
content hash. It was frozen after an initial diagnostic reproduction, so this
is a repeatable regression comparison, not a preregistered research study.
Both arms ran three times with an external 15-second process timeout on macOS
27.0.1 / Apple M1 Pro / 32 GiB / Python 3.12.12.

| Arm | Outcomes | Measured wall time |
| --- | --- | --- |
| Baseline | 3/3 externally timed out | 15.0087–15.0119 s per attempt |
| Candidate | 3/3 passed | 1.14696–1.16540 s per attempt; median 1.15006 s |

The baseline times are right-censored at the timeout and cannot be compared as
throughput or speedup. Candidate time includes Python/pytest startup and the
test's intentional 0.9-second slow step. No memory, CPU utilization, or
larger-workload overhead was measured.

The focused trajectory store/admission/worker/benchmark suite passed (64
tests). The full local suite completed with **1081 passed, 78 skipped, 1
failed**. Its failure was
`test_cancellation_drains_owned_launch_and_cleanup[cleanup]` in
`tests/test_grader_guard.py`, outside the modified files. That case and its
whole test file passed when rerun alone on both baseline and candidate; the
suite-level failure remains disclosed rather than called a full-suite pass.

## Reproduction

Run from the candidate commit with Python 3.12 and the project test
dependencies installed:

```bash
SCRATCH=$(mktemp -d /tmp/vare-worker-repro.XXXXXX)
BASE="$SCRATCH/baseline"
CANDIDATE="$SCRATCH/candidate"
BASE_OUT="$SCRATCH/baseline-run"
CANDIDATE_OUT="$SCRATCH/candidate-run"
git worktree add --detach "$BASE" 40db644c7eae4d830dc721e1d8185665229cfbd6
git worktree add --detach "$CANDIDATE" 3f546253b7df26799ce65098f4c62f7d6132d263
python scripts/reproduce_trajectory_worker_sqlite.py \
  --arm baseline --checkout "$BASE" \
  --output "$BASE_OUT" \
  --protocol protocols/trajectory_worker_sqlite_serialization_v2.lock.json
python scripts/reproduce_trajectory_worker_sqlite.py \
  --arm candidate --checkout "$CANDIDATE" \
  --output "$CANDIDATE_OUT" \
  --protocol protocols/trajectory_worker_sqlite_serialization_v2.lock.json
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_trajectory_store.py tests/test_trajectory_admission.py \
  tests/test_trajectory_workers.py tests/test_trajectory_benchmark.py
```

Raw outputs, including the suite-level non-pass, are in
[`results/trajectory-worker-sqlite-serialization-v1/`](../results/trajectory-worker-sqlite-serialization-v1/).
`manifest.json` contains the committed file hashes. This establishes one
bounded same-host regression fix; it does not establish multi-process SQLite
safety, production-scale stage throughput, or any model-learning result.
