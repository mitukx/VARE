# Runtime promotion acknowledgment and recovery fence

## Concrete defects

Main `f4b8d11177f2fa0d75eb1ff980c5b2436561b4d7` reports an accepted candidate as promoted when `hooks.promote()` returns normally, even if the default, unjournaled runtime still reports the incumbent or a different policy. Its checkpoint-bound journal path checks identity, but the default path does not.

Publication can also change the runtime before raising or being cancelled. The old engine reports that failure but permits another round when no journal is configured. A real SQLite journal blocks generation while a transaction remains prepared, but the next round still calls the potentially inconsistent runtime's `active_policy()` first. If SQLite commits and its acknowledgment is lost, the journal is already clean and more rollout/training work can start before the consumed-nonce check eventually stops another promotion.

These are outer-loop publication-contract defects. No model learning or serving implementation is replaced.

## Correction and compatibility

After every successful publish-hook return, the engine now requires `active_policy()[0] == candidate_id` before emitting promotion success. The strict journal path additionally retains its independently signed checkpoint-digest comparison and durable commit. Policy identity alone does not authenticate model weights; the existing trusted checkpoint hasher and strict attestation are still required for that stronger contract.

An exception from the publish hook, identity acknowledgment, checkpoint check, or journal commit permanently fences that `CapabilityLoop` instance. The next round raises `PromotionRecoveryRequired` before any hook call or rollout-step increment. There is no automatic reset, candidate discard, or rollback after a possibly partial publication. An operator must independently inspect/restore the actual runtime, verify durable state if configured, and construct a fresh loop after recovery.

Unjournaled caller cancellation continues to propagate `CancelledError`; the local recovery fence is still set first. Journaled publication failures continue to require durable reconciliation. The existing discard-failure fence remains separate. Normal publication with and without a journal still succeeds. The operator-interlock test fixture now installs and acknowledges its candidate instead of merely recording a call; this changes no frozen experiment or acceptance rule.

## Frozen CPU comparison

The [protocol](../protocols/promotion_publication_recovery_v1.lock.json) and test were committed locally before the first baseline run. Its ten fault cases cover exceptions/cancellation before and after runtime mutation, no-op and foreign-policy acknowledgment, checkpoint mismatch, SQLite commit failure before commit, lost acknowledgment after a real commit, and actual caller cancellation. Two successful-publication controls cover journaled and unjournaled operation.

| Observation | Pinned main | Correction |
| --- | --- | --- |
| Fault cases | 10 failures | 10 passes |
| Successful-publication controls | 2 passes | 2 passes |
| Setup errors / skipped cases | 0 / 0 | 0 / 0 |
| No-op or wrong-policy publication | Returns a false promoted result | Raises recovery-required error; no success event |
| Round after a failed publication | Accesses runtime hooks; in some cases also rolls out/trains | Stops before any hook or provenance step |
| Lost SQLite commit acknowledgment | Pointer is committed; further work can start | Committed evidence retained; same loop stays fenced |

The [first baseline](../results/promotion-publication-recovery-v1/initial-baseline/) and [matched replay](../results/promotion-publication-recovery-v1/run-1/) are retained. The replay checks the exact expected failure assertion for each baseline fault, so unrelated failures cannot satisfy the comparison. Manifests bind the protocol, test, engine, interface, runner, Python/pytest versions, case outcomes, and artifact hashes. Local validation commit identifiers are not published Git objects; retained source hashes identify the tested implementation independently of connector-generated commit metadata.

```bash
python -m pip install -e . pytest==8.4.2
python scripts/reproduce_promotion_publication_recovery_v1.py --output /tmp/vare-publication-replay
```

The focused publication, discard, checkpoint, interlock, evaluation-lifetime and overlapping-round selection passes 55 tests. The default full suite with the optional CPU trainer dependencies passes 619 tests; the separate mathematical graders pass 14 tests. Ruff 0.15.12 and package-wide mypy 1.19.1 pass. [Validation logs and hashes](../results/promotion-publication-recovery-v1/validation/) are retained. GitHub CI reruns the frozen 12-case comparison on Python 3.12 and uploads the records alongside the existing CPU reproductions.

## Primary-source design check

Primary sources checked on 2026-10-14: [Python 3.12 task cancellation](https://docs.python.org/3.12/library/asyncio-task.html#task-cancellation) supports preserving cancellation after cleanup; [vLLM weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/) already owns transport and staged weight installation; [vLLM async RL](https://docs.vllm.ai/en/latest/training/async_rl/) exposes committed version metadata and notes that version labels can be changed without changing weights. VARE therefore adds an outer-loop completion/recovery check, not another transport, trainer, or proof that a version label authenticates tensors. The earlier [RLVR/OSS scope review](evaluation-transaction-integration-v1.md#literature-and-oss-decision) still applies. No novel distributed-commit or RL algorithm is claimed.

## Limits and next research decision

Evidence covers one loop in one process with trusted, cooperative fixtures and actual local SQLite transactions. It does not implement multi-host atomicity, durable quarantine across process restart, worker termination, cancellation-safe publication ownership, or independent verification of external recovery. An unreported tensor mutation can still evade an identity-only acknowledgment. Post-publication telemetry/decision-ledger failures, multiple loops sharing a runtime, and interrupted noncooperative workers remain separate boundaries.

No consumed task cohort, model checkpoint, reward threshold, historical result, or frozen protocol was changed. Independently confirmed task success after a real model update remains the main research gap; this correctness result does not authorize reopening retired learner studies.
