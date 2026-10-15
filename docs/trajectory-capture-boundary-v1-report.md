# Trajectory policy and capture completion

The opt-in trajectory gate could accept a closed capture that omitted a live
tool's result. The offline network auditor also accepted seven constructed
URL targets that the runtime dispatcher rejected under the same frozen policy.
This change aligns those boundaries using the existing recorder, dispatcher,
signed evaluation report and promotion gate.

## Reproduction and fix

Pinned main: `f54c488e272b7dbe513558a2e3489700f5814d35`.

- The dispatcher rejected nonstandard or invalid ports, trailing-dot hosts,
  fragments, spaces and control characters. The auditor ignored some of those
  checks and stripped trailing dots. One shared URL predicate now supplies
  both boundaries; its existing `trajectory_dispatch.egress_host_allowed`
  import remains available. Wrong hosts retain their existing reason; rejected
  URLs on an allowed host add `egress_target_not_allowed`.
- `close(expected_event_count=1)` could succeed after the call event but before
  the awaited tool completed. The call subsequently performed its side effect,
  then failed to record its result because the recorder was already closed.
  The one-event trace still audited as a complete pass. The dispatcher now
  refuses closure until all its `execute()` coroutines are terminal. A rejected
  early close leaves the capture open so the owner can drain work and retry.
- If result persistence failed, a caller could close using the smaller event
  count. Capture persistence errors now permanently fence new calls and close
  on that dispatcher; retain the partial evidence and use a fresh run after
  reconciling the actual side effects. There is no automatic reset or rollback.

Public signatures and event schema stay unchanged. General recorder traces
still describe recorded authorization, not a universal call/result protocol.
The completion guard belongs to the cooperative dispatcher. The independent
runtime event counter is still required; its value is never inferred from the
dispatcher's own count.

## Results

The [frozen comparison](../protocols/trajectory_capture_boundary_v1.lock.json)
was defined after development reproduced the defect, not before discovery.
The identical protected test file grades both source trees in separate pytest
processes. **Main: 11 failures / 5 controls pass; corrected: 16 pass.** These
are engineered correctness cases, not attack or false-positive rate estimates.

The integration case launches a real child Python process, holds it until the
parent signals completion, and independently reads its output file. Main
closes early, loses the result event, and the signed promotion gate accepts the
incomplete one-event capture after task output succeeds. The correction rejects
early closure/auditing, records the terminal result, then allows the qualified
two-event capture through the existing signed gate. The raw traces, output
bytes and full reports are [retained](../results/trajectory-capture-boundary-v1/comparison-v2/manifest.json).
Its incumbent 0.5 and candidate 1.0 are scripted gate controls. There is no
model inference, optimizer update or learning claim.

Five alternating-order CPU pairs each execute 32 no-op tools, close the 64
events and audit them. `tracemalloc` covers dispatch and auditing; startup and
artifact export are excluded. Whole-process RSS is a high-water measurement.
Python 3.12.12, macOS arm64, Apple M1 Pro / 32 GiB; no API or GPU spend.

| Metric, median | Main | Corrected |
| --- | ---: | ---: |
| Dispatch + close | 148.772 ms | 152.983 ms |
| Tools per second | 215.095 | 209.174 |
| Offline audit | 5.752 ms | 5.573 ms |
| Peak traced allocation | 177,999 B | 184,735 B |
| Process peak RSS | 27,901,952 B | 27,164,672 B |

The corrected dispatch median is **2.83% slower**, and traced peak allocation
is 6,736 B higher in this small local sample. No throughput or memory speedup
is claimed. The first retained run had nearly equal dispatch medians; local
fsync/scheduling variation limits interpretation. No performance optimization
was attempted.

The separate auditor imports neither the comparison runner nor VARE's
trajectory parser. It reconstructs hashes, parent/result relationships, event
counts, measurement coverage and medians from ten timing traces and four
command traces, and checks four reports using the existing HMAC contract.
Normal and Python `-O` execution both pass; artifact checks use explicit
exceptions. Public fixture keys establish consistency, not authentication
against a hostile signer or an independently administered evaluator.

## Reproduce and retained iterations

With an activated Python >=3.11 environment and pytest, from the repository:

```bash
python scripts/reproduce_trajectory_capture_boundary_v1.py --output /tmp/vare-capture-new
python scripts/audit_trajectory_capture_boundary_v1.py /tmp/vare-capture-new
python -O scripts/audit_trajectory_capture_boundary_v1.py /tmp/vare-capture-new
python -m pytest tests/test_trajectory_capture_boundary.py tests/test_trajectory_capture_audit.py
```

Use a new output directory. The runner fetches only pinned source paths, not
unrelated large artifacts. CI runs this comparison on Python 3.12 and retains
the full bundle. Local validation is in `results/trajectory-capture-boundary-v1/validation.json`.

The first development reproduction and comparison-v1 remain preserved. The
first comparison retained summaries and report metadata but not raw command
and timing traces; it is not claimed as independently replayable capture
evidence. Evidence revision 2 exports those traces and full reports, and binds
fixture receipts to actual protocol/test hashes. Task inputs, all assertions,
acceptance counts and resource budgets were unchanged. No failed run was
replaced, and no retired model confirmation cohort was used.

## Scope

This does not intercept arbitrary sockets, follow redirects, contain hostile
code, authenticate dispatch counters, detect unrecorded effects, or stop work
detached by a tool callback. Trusted callbacks must own and drain their actual
workers before returning, including cancellation cleanup. Single event-loop
use and privileged recorder ownership remain required. A process crash leaves
an open, unpromotable capture; it does not resume that tool or undo its effects.
The guard prevents these captures through the dispatcher; arbitrary legacy v1
traces have no declared tool-lifetime contract, so their missing terminal events
cannot be inferred universally. Independent real-model task improvement remains
an unresolved separate gap.
