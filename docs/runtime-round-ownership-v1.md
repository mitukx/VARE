# Whole-round runtime ownership and operator recovery

## Defect and correction

Pinned VARE main `377d8894d3d20dd4d37677e223371ef11eadd354` serialized a round and fenced failed publication/discard only within one `CapabilityLoop`. A sibling loop could use the same mutable hooks concurrently or bypass the first loop's unknown-state fence. A promotion journal did not reserve the runtime during training/discard; process death before a promotion intent, or a commit with lost acknowledgment, could leave no pending intent to stop a fresh loop.

Loops using the same hooks object now share a `RuntimeCoordinator` by object identity, including unhashable hooks. Distinct proxies for the same runtime must explicitly share `runtime_coordinator`, or use the same `PromotionJournal` database. The lock covers the full round, before policy lookup. Unknown training, discard, publication, or completion state quarantines the coordinator. Dropping a loop does not clear its fence while the hooks object remains alive. Weak-referenceable hooks are not deliberately kept alive by the registry; non-weak-referenceable hooks are retained for the process lifetime. Coordination assumes one asyncio event loop, not concurrent threads/event loops.

A configured promotion journal additionally acquires a unique full-round reservation before any hook access. A separate connection/loop is refused while the reservation exists. Reservations never expire automatically. On success, or an evaluation failure followed by owned successful discard, the engine verifies the runtime against the durable pointer before releasing its reservation. Unknown state leaves it intact, including after a commit whose acknowledgment was lost. A failed training hook without a candidate handle is quarantined rather than automatically discarded.

## Frozen comparison

The [protocol](../protocols/runtime_round_ownership_v1.lock.json) and [ten-case test](../tests/test_runtime_round_ownership.py) were committed locally as `9eb5117` before the first baseline was run. The protocol fixes the test SHA-256 and baseline revision. The [first baseline](../results/runtime-round-ownership-v1/first-baseline/) and [matched replay](../results/runtime-round-ownership-v1/run-1/manifest.json) are retained.

| Arm | Semantic failures | Controls passing | Errors / skips |
| --- | --- | --- | --- |
| Pinned main | 8 | 2 | 0 / 0 |
| Correction | 0 | 2 (all 10 cases pass) | 0 / 0 |

The eight faults cover sibling overlap; shared discard, publication, and training uncertainty; fresh journal/hook reconstruction after discard failure and lost commit acknowledgment; distinct hooks sharing a journal during training; and a real spawned process exiting during training before promotion intent. The controls preserve normal publication with and without SQLite. Baseline failures must have the specific overlap/hook-access assertion; setup failures do not count as evidence.

```bash
python -m pip install -e '.[dev]'
python scripts/reproduce_runtime_round_ownership_v1.py --output /tmp/vare-runtime-round-review
python -m pytest tests/test_runtime_round_ownership.py tests/test_runtime_round_recovery.py
```

Use a new output directory. Git must have, or be able to fetch, the pinned baseline. The first matched candidate was an uncommitted working tree based on the protocol commit; its recorded file hashes, not that HEAD alone, identify the tested source. Later CI replays identify the submitted head and retain their own logs/hashes. The local chronology is not externally preregistered. Exact first-run source bytes are retained under `run-1/snapshot/` and covered by a separate retention manifest. The comparison uses CPU fixtures, local SQLite, and a spawned process; no model downloads or confirmation prompts.

Maintained checks outside that frozen comparison cover wrong owner/hash/policy recovery, pending intent rejection, nonce ownership, registry lifetime, explicit proxy coordination, successful rollback, and cancelled waiters. Additional maintained checks cover two real process claimants (only one acquires ownership), additive schema initialization without changing old promotion records, and lost release acknowledgment. Two further spawned-process checks die immediately after publication and after the actual SQLite commit; both require explicit recovery before a subsequent successful round. These checkpoint fixtures represent identities/digests, not actual tensor or optimizer restoration. The first version of these two checks failed because its reconstructed fixture restarted the candidate serial at zero and proposed the incumbent again. Setting the fixture serial to the observed version corrected the test; the frozen ten-case test was not changed.

The final local CPU suite passed **639 tests**, with **14** mathematical grader tests run separately. Ruff 0.15.12, mypy 1.19.1, navigation-link checks, and the 235-blob retention budget passed. [Validation logs](../results/runtime-round-ownership-v1/validation/) retain the full-suite and maintained checks. A deliberately injected failure exercised the multiprocess diagnostic path without changing its strict acceptance criteria.

## Operator recovery

Protect the SQLite file and checkpoint store from candidate writes. If `status()['round_owner_id']` is present, first stop and join **all** old owners and their workers; do not use a timeout or matching ID as proof of termination. Independently restore/verify the model, optimizer, RNG, and any other mutable runtime state. A checkpoint digest alone covers only what the trusted hasher actually measures.

For a pending promotion, use the existing `reconcile()` with independently observed checkpoint identity/hash first. Then release the exact round reservation:

```python
status = journal.status()
# Old processes/workers are already terminal and the runtime was verified.
if status['pending_tx_id'] is not None:
    journal.reconcile(
        tx_id=status['pending_tx_id'],
        observed_policy_id=verified_policy_id,
        observed_checkpoint_sha256=verified_checkpoint_sha256,
    )
journal.recover_round(
    owner_id=status['round_owner_id'],
    observed_policy_id=verified_policy_id,
    observed_checkpoint_sha256=verified_checkpoint_sha256,
)
# If retaining the same hooks/coordinator in this process:
loop.runtime_coordinator.recover()
# Construct a fresh loop with fresh per-round evidence/nonce.
```

An old loop's local publication/discard fence is intentionally not reset by coordinator recovery. Construct a fresh loop. Use a new trusted promotion nonce for each independently attested candidate comparison. Wrong reservation owners, a pending intent, or a checkpoint differing from the durable pointer cannot release the reservation. `assert_clean()` still checks only promotion intent/pointer agreement; it is not an ownership check. Engine entry always calls `acquire_round()` first.

## Diagnostics and limits

The existing multiprocess evaluator test keeps its original 12-job, four-worker, two-second-lease acceptance condition. On failure it now captures read-only jobs, attempts (stdout/stderr capped at 4096 bytes each), events, runtime version, and elapsed time. With `VARE_TEST_DIAGNOSTICS_DIR` set, CI retains this JSON as an artifact. It does not refresh inputs, extend leases, retry failures, or establish the cause of the earlier transient CI failure.

This is single-host cooperative runtime ownership, not a new distributed commit algorithm. SQLite exclusion requires every controller to use this reservation-aware version and the same local database; older versions ignore the added reservation table and must not run concurrently; separate databases and uncoordinated distinct proxies do not fence each other. It does not terminate detached workers, make publication cancellation-safe, authenticate tensor state without a trusted hasher, prove operator observations, or establish hostile-code isolation. No power-loss, network-filesystem, distributed/GPU behavior, throughput advantage, independent human reproduction, or model-quality improvement is claimed.
