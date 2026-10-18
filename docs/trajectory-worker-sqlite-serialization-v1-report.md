# Same-process trajectory worker SQLite contention

## Finding

Several concurrent `asyncio.to_thread` workers opening and closing SQLite
connections through one `TrajectoryStore` reproduced a hang in the bounded
execute/verify/consume pipeline. Serializing each store connection lifetime
with an instance `RLock` fixed that observed failure, but review found that
`iter_steps()` yielded rows while still holding the lock. A paused consumer
could therefore block a lease heartbeat until the worker lease expired.

`iter_steps()` now reads at most 16 rows per page, closes the connection, and
only then yields to the consumer. The RLock remains limited to one store object
in one process; separate objects and processes do not coordinate through it.
Because the runtime source is enrolled by hash, active stores enrolled under
older `trajectory_store.py` bytes still reject normal resume. No migration is
implemented here.

## Measurements

All runs used macOS 27.0.1 / Apple M1 Pro / 32 GiB / Python 3.12.12. These are
correctness regressions, not model-learning or performance claims.

The v1/v2 pinned comparison had the existing bounded-backlog worker test time
out on baseline in 3/3 runs and pass on the initial candidate in 3/3 runs. A
later v3 repetition exposed that this baseline hang is intermittent: 2/3 runs
timed out at 15.077 and 15.088 seconds, while one completed in 4.505 seconds.
The candidate passed in 3/3 runs at 1.198–1.376 seconds. Since the baseline
did not meet the v3 lock's expected outcome, v3 is a **non-pass** and is
retained as such. The original test is not a deterministic oracle for this
race and its timings do not establish a speedup.

The review-specific v4 test pauses `iter_steps()` after its first row, then
requires a lease heartbeat to finish within 0.5 seconds. On the code before
the paged-reader fix, it failed in 3/3 runs (process wall time 0.822–0.929
seconds). On the fix, it passed in 3/3 runs (0.247–0.280 seconds). Wall time
includes Python/pytest startup. This is direct evidence that the reported
heartbeat starvation is fixed for the tested same-store path.

| Check | Pre-fix | Fixed candidate |
| --- | --- | --- |
| Paused-reader heartbeat, v4 | 3/3 fail; 0.822–0.929 s | 3/3 pass; 0.247–0.280 s |
| Existing async pipeline test, v3 | 2/3 timeout, 1/3 pass | 3/3 pass; 1.198–1.376 s |
| Targeted trajectory tests | — | 67 passed in 9.58 s |
| Broad local suite, excluding the two standalone Math500 grader modules | — | 1084 passed, 78 skipped in 166.01 s |

An earlier broad suite run had one grader-cleanup failure; its isolated test
and test file passed on baseline and candidate. The later broad run passed.
Both raw logs remain in the result bundle. The suite excludes
`tests/test_math500_grader_v1.py` and `tests/test_math500_grader_v2.py`; CI
continues to run its dedicated grader jobs.

No CPU utilization, peak memory, multi-process coordination, storage scaling,
or workload throughput measurement was made. The per-store lock serializes
database access and may limit concurrency under other workloads.

## Reproduction

Use Python 3.12.12 and the repository test dependencies. The v4 lock pins the
pre-fix and fixed source revisions, source hashes, the regression test hash,
runner hash, three repetitions, and expected exit codes:

```bash
SCRATCH=$(mktemp -d /tmp/vare-worker-repro.XXXXXX)
BASE="$SCRATCH/baseline"
CANDIDATE="$SCRATCH/candidate"
git worktree add --detach "$BASE" 3f546253b7df26799ce65098f4c62f7d6132d263
git worktree add --detach "$CANDIDATE" e3a9e1908052ea14f6126386e303939393914fc4
python scripts/reproduce_trajectory_worker_sqlite.py \
  --arm baseline --checkout "$BASE" \
  --output "$SCRATCH/baseline-run" \
  --protocol protocols/trajectory_worker_paused_reader_heartbeat_v4.lock.json
python scripts/reproduce_trajectory_worker_sqlite.py \
  --arm candidate --checkout "$CANDIDATE" \
  --output "$SCRATCH/candidate-run" \
  --protocol protocols/trajectory_worker_paused_reader_heartbeat_v4.lock.json
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts= -q \
  tests/test_trajectory_store.py tests/test_trajectory_admission.py \
  tests/test_trajectory_workers.py tests/test_trajectory_benchmark.py \
  tests/test_trajectory_worker_lease_heartbeat.py \
  tests/test_trajectory_worker_sqlite_reproduction.py
```

The runner places the locked test file in a temporary isolated directory so
the pre-fix checkout remains clean while the test imports that checkout's
`src`. Raw v1–v4 logs and summaries are in
[`results/trajectory-worker-sqlite-serialization-v1/`](../results/trajectory-worker-sqlite-serialization-v1/);
`manifest.json` records content hashes. This establishes one bounded,
single-process storage correctness fix. It does not establish multi-process
SQLite safety, production-scale throughput, or any model-learning result.
