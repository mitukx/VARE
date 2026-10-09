# Concurrent round transaction v1

## Question

Can two concurrent `CapabilityLoop.run_round` calls on one loop promote candidates from the same stale incumbent, allowing the later completion to overwrite an earlier successful update?

## Frozen protocol and baseline

The [protocol lock](../protocols/engine_concurrent_round_transactions_v1.lock.json) and [deterministic async regression](../tests/test_engine_concurrent_round_transactions.py) were frozen before baseline measurement. Two rounds start from `p0`. The fixture makes candidate `c1` promote first and delays `c2` evaluation until that promotion. Each update adds 0.1 to its parent's state. The acceptance condition requires `c2` to branch from promoted `c1`, ending at 0.2.

At baseline revision `0995f7fc3a92b317c1b8dce7e13b87c583cf42ec`, both rounds observed `p0`; the later candidate also branched from `p0` and overwrote `c1` with a stale branch at 0.1. The frozen test failed on `parents["c2"] == "c1"`.

## Fix and result

`CapabilityLoop` now holds a per-instance `asyncio.Lock` across the full `run_round` transaction. This serializes calls on one loop while preserving the configured concurrency among rollouts inside each round. After `c1` is promoted, the next round observes it as the incumbent and builds `c2` from that state.

The frozen regression passed after the fix; the concurrent-round, prompt-identity, replay-group, freshness, and RVL-hook suite passed 21 tests. The full repository pytest run exited successfully. GitHub Actions run [37865040331](https://github.com/mitukx/VARE/actions/runs/37865040331) passed 196 tests with 12 skipped, recomputed the frozen GRPO audit studies, and completed the demo. Raw logs, result JSON, and SHA-256 manifest are retained in [`results/engine-concurrent-round-transactions-v1/`](../results/engine-concurrent-round-transactions-v1/). Runtime: Python 3.12.12, pytest 8.4.2, CPU.

## Limitations and decision

This demonstrates one in-process lost-update schedule and its prevention on a single `CapabilityLoop`. It does not cover two loop instances sharing hooks, cancellation/process loss mid-round, distributed locks, or model quality. This closes a specific transaction-integrity gap; it is not a post-training capability result. Keep the fix and preserve external reproduction and independent task-success improvement as open milestones.
