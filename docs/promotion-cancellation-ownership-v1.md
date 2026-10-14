# Accepted-promotion cancellation ownership

## Defect and correction

When a caller cancelled `CapabilityLoop.run_round()` during `hooks.promote()`,
the cancellation was delivered into the hook coroutine. A hook that caught
`CancelledError`, finished installing the candidate, and returned normally
could make the engine report an ordinary successful `RoundResult`. A hook could
also suppress its own cancellation without clearing its task cancellation
state and be accepted as successful.

The engine now runs accepted publication in one owned task and retains the
round lock until that task is terminal and the runtime acknowledgment is
checked. For journaled promotion, caller cancellations are shielded through
policy and checkpoint acknowledgment, SQLite commit, decision recording, and
round release. A successful result includes `deferred_cancellation_count`, and
the event log records one `promotion_cancellation_deferred` event with the
number of cancellation requests. Repeated `Task.cancel()` requests are counted
individually, including requests coalesced into one `CancelledError` delivery.
Publication is never retried.

For unjournaled promotion, the existing API contract remains: caller
cancellation is delivered once to the owned hook, the engine drains hook
cleanup while holding round ownership, then propagates `CancelledError`. The
loop is fenced because runtime state may have changed. A hook that suppresses
its own cancellation but leaves `Task.cancelling()` nonzero is treated as an
unverified publication and fenced. A hook or journal failure takes precedence
over deferred caller cancellation and retains the prepared recovery state.

## Frozen CPU comparison

The [protocol](../protocols/promotion_cancellation_ownership_v1.lock.json),
seven focused cases, and runner were added before the matched run. The test
imports the existing publication fixtures; it does not modify the frozen v1
publication test or its acceptance criteria.

| Observation | Pinned main `4ab4a6a` | Candidate |
| --- | ---: | ---: |
| Cancellation ownership and output-compatibility cases | 7 failures | 7 passes |
| Setup errors / skips | 0 / 0 | 0 / 0 |
| Repeated caller cancellation during durable publication | Cancels the hook task | Hook remains uncancelled; commit completes and the result records 2 deferred cancellations |
| Caller cancellation swallowed by unjournaled hook | Returns a normal result | Drains the hook, propagates cancellation, and fences reentry |
| Publication failure after caller cancellation | Recovery failure retained | Recovery failure retained; no promotion event and prepared journal state remains |

Run the comparison with:

```bash
python -m pip install -e . pytest==8.4.2
python scripts/reproduce_promotion_cancellation_ownership_v1.py --output /tmp/vare-promotion-cancellation-review
```

The runner extracts `src/` from the pinned baseline commit and runs the same
test file against both source trees. It checks the expected baseline failures,
candidate passes, case count, interpreter, source digests, and artifact hashes.
The workflow reruns the comparison on Python 3.12 and uploads its raw logs and
manifest.

`RoundResult` gains a defaulted `deferred_cancellation_count` field. Because
`vare demo --output` serializes each result with `dataclasses.asdict`, this adds
`"deferred_cancellation_count": 0` to every ordinary result row as an additive
JSON-schema change; positional construction of the existing fields remains
valid because the new field has a default.

## Scope and limits

This is CPU-only async control-plane evidence. It adds no training run, model
capability, RL algorithm, or RSI claim. The user-provided [QwenGyre paper](https://arxiv.org/abs/2609.33848)
centers on GPU allocation and long-horizon trajectory systems; this correction
addresses a local Python task-ownership boundary and does not reproduce that
work.

Evidence covers one event loop, one process, local SQLite, and cooperative
hooks. It provides no multi-host atomicity, hard cancellation deadline, or way
to terminate detached workers. Cancellation after durable publication is
dispatched cannot undo an already committed runtime change; the returned result
and event make that deferred cancellation visible. A durable journal remains
optional, and unjournaled uncertain publication requires external recovery.
