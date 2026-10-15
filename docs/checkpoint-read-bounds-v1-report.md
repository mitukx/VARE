# Bounded checkpoint IO and evaluator recovery

`TrustedCheckpointStore` could block before rejecting a FIFO, and a small
registered checkpoint replaced by a larger regular file could allocate far
beyond the evaluator's read ceiling. The correction rejects nonregular inputs
without waiting for a FIFO writer and bounds reads by the registered size and
the caller's ceiling, including growth after descriptor inspection.

This is CPU systems correctness/resource evidence on VARE's actual checkpoint
code. The end-to-end exercise uses the existing linear-classifier protocol
fixture, signed evaluator subprocess, `CapabilityLoop`, and `PromotionJournal`.
Its candidate is a known saved checkpoint selected by a scripted hook; no
optimizer or coding-model inference runs. Its 0.5 -> 1.0 fixture score is a
promotion control, **not a learning or model-capability gain**.

## Defect and correction

On pinned main `e912938125b8a135347961801e84f624c2b833eb`:

1. `os.open(O_RDONLY)` on a FIFO waits for a writer before `fstat` can reject
   the nonregular descriptor. Registration, digest verification and in-memory
   evaluation all share this path.
2. `read_verified(max_read_bytes=64)` checked the registry's small size, then
   loaded using the store-wide limit (4 GiB by default). A replaced or growing
   object was eventually rejected by its hash, after unnecessary allocation.
3. Fixed 1 MiB chunks could overshoot even a much smaller streaming budget.

The existing reader now uses nonblocking open before its regular-file check,
unbuffered bounded reads with at most one sentinel byte beyond the ceiling,
and never yields the excess block. Loaded reads use the minimum of store,
caller and registered-size bounds; digest verification uses the registered
size too. Empty objects, exact-limit objects, hashes and policy bindings retain
their contracts. Public method signatures and the control-plane architecture
are unchanged.

## Comparison

The [locked comparison](../protocols/checkpoint_read_bounds_v1.lock.json) follows
an observed development reproduction, not preregistered defect discovery.
Five alternating-order pairs run each of three workloads in separate Python
processes. Input allocation precedes tracing. Times include store lookup and
verification, exclude process launch, and run under `tracemalloc`; peak RSS is
whole-process high-water memory. These are local cached-file measurements,
not storage hardware throughput or population estimates.

Final [raw bundle](../results/checkpoint-read-bounds-v1/comparison-v5/manifest.json):
Python 3.12.12, macOS arm64, Apple M1 Pro / 32 GiB, CPU only, zero API spend and no GPU use.

| Workload | Main median | Corrected median |
| --- | ---: | ---: |
| 32 MiB replacement, traced peak allocation | 67,114,578 B | 4,398 B |
| Replacement rejection latency | 10.843 ms | 0.312 ms |
| Replacement process peak RSS | 90,816,512 B | 22,462,464 B |
| 8 MiB growth after `fstat`, traced peak allocation | 16,780,733 B | 8,420 B |
| Growth rejection latency, including injected write | 3.588 ms | 1.760 ms |
| 64 verified reads of 1 MiB, elapsed | 51.764 ms | 52.717 ms |
| Valid-read throughput, descriptive | 1,236.39 MiB/s | 1,214.04 MiB/s |

All invalid reads reject in both revisions; the measured benefit is bounded
work before rejection. The final normal-read median is 1.84% slower; earlier
measurements varied in direction. No normal-throughput speedup is claimed. Registered size limits payload IO, not total Python RSS: accepted reads
can retain chunks and the joined result simultaneously.

The 15 independently asserted regression cases give **main: 7 fail / 8 pass;
corrected: 15 pass**. Three direct FIFO operations hit the external 2 s
watchdog on main. In the separate-process promotion exercise, main's FIFO
evaluation also requires that watchdog; the corrected worker rejects itself.
Both faulty candidates are discarded without promotion. After trusted
restoration of the exact bytes and terminal worker completion, the next round
promotes; a reopened SQLite journal verifies the new checkpoint and no pending
intent or round owner. This reopens the journal, not an interrupted model
optimizer or a process killed halfway through publication.

## Retained failures and protocol changes

- The first 13-case development run had six failures on unchanged main.
- `comparison-v1` stopped before evaluations because `git archive` in a
  partial clone tried to hydrate unrelated blobs. The runner now reads only
  pinned `src/` paths with `git show`.
- The initial completed `comparison-v2` passed its audit but pytest embedded
  a 1 MiB bytes parameter in test IDs. Oversized raw records remain outside Git,
  indexed by [size and SHA-256](../results/checkpoint-read-bounds-v1/initial-retention-index.json).
  Explicit short IDs preserve the same inputs and assertions.
- `comparison-v3` passed the regression/resource gates but failed the separate
  evidence audit: cancellation of a test hook around `to_thread(subprocess.run)`
  could abandon a live worker and omit its result. The corrected test hook
  drains its bounded worker before propagating cancellation. The failed
  [audit record](../results/checkpoint-read-bounds-v1/comparison-v3-audit-failure.json)
  and complete small bundle remain retained. The production IO patch and all
  acceptance limits were unchanged.
- `comparison-v4` passes the comparison and separate stdlib auditor, which
  reconstructs 30 measurements, 15 cases per arm and four recovery records.
  Changed measurement and missing-worker-file probes fail closed.
- `comparison-v5` repeats the unchanged IO/task protocol after integration with
  main `7801f24`. Reproducer and auditor gates now use explicit exceptions so
  Python `-O` cannot disable them; six artifact-fault regressions cover normal
  and optimized audit execution. Both audit modes pass. The earlier runner
  source is retained with its exact v4 hash under `source-snapshots/`.
- Early suite attempts failed on sparse-checkout omissions (a helper module,
  then three retained-record tests); their logs are preserved. They were fixed
  by fetching existing assets, not by changing those tests. The all-tests
  launcher also initially selected OS Python 3.9 in a child command; activating
  the required Python 3.12 environment corrected it. Integration with newer
  main required one additional frozen-result directory for six existing tests.
  These failed attempts are retained separately from the final complete run.

No retired model confirmation task was rerun. New IO faults are development
regressions, reusable as such; the six new linear inputs are protocol controls,
not a protected or untouched model confirmation cohort.

## Reproduce

From a checkout with Python >=3.11 and the development dependency installed:

```bash
python scripts/reproduce_checkpoint_read_bounds_v1.py --output /tmp/vare-checkpoint-io-new
python scripts/audit_checkpoint_read_bounds_v1.py /tmp/vare-checkpoint-io-new
python -m pytest tests/test_checkpoint_read_bounds.py tests/test_checkpoint_io_recovery.py
```

Use a new output path. The comparison requires POSIX FIFOs and Git access to
the pinned revision; only stdlib and pytest are needed. CI reproduces it on
Python 3.12 and uploads its artifacts. Read the retained validation record for
local suite/lint/type results; CI status is a separate external result.

## Limits and next gap

Nonblocking open does not provide an IO deadline for regular files on failing
or remote filesystems. There is no hostile same-UID isolation, parent-directory
race protection, power-loss guarantee, distributed recovery, or proof of full
optimizer/RNG restoration. HMAC keys in public fixtures are consistency checks,
not secrets. The artifact auditor is a separate same-host implementation, not
outside reproduction or external authentication.

The higher-level gap remains independently graded task improvement after a
real agent/model intervention. This change makes that existing checkpoint and
evaluation boundary harder to stall or exhaust; it does not fill that gap or
authorize a new learning study.
