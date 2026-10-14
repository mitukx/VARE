# Candidate discard ownership and failed-restore quarantine

## Defect and correction

After [evaluation transaction integration](evaluation-transaction-integration-v1.md) merged in [PR #42](https://github.com/mitukx/VARE/pull/42), main `b05e0ef` still directly awaited candidate discard on ordinary rejection and pre-promotion journal/ledger errors. Caller cancellation could interrupt asynchronous restoration and release the round lock. Separately, even the evaluation-error path reported failed discard as unknown state but allowed the same loop to start another round.

All four engine discard sites now share one ownership helper. It creates a separate discard task, shields it from caller cancellation, drains it to termination, retrieves its result, and keeps the round lock until that point. Successful restoration propagates the caller's first cancellation on ordinary rejection; when handling an existing transaction error, successful restoration preserves that earlier error. Failed discard instead reports unknown policy state with the actual hook exception as its cause, including an internally cancelled discard hook.

A failed discard permanently marks **that `CapabilityLoop` instance** as requiring recovery. Subsequent `run_round` calls fail before reading policy identity, invoking hooks, or advancing rollout provenance. A hook ceasing to throw is not evidence that partially restored state is correct. There is deliberately no automatic reset. An operator must stop dependent work, independently restore/verify the runtime, and only then construct a fresh loop; existing durable promotion-journal reconciliation still applies.

## Frozen CPU evidence

The [protocol](../protocols/candidate_discard_recovery_v1.lock.json) predates the first test outcome. Its seven cases cover ordinary rejection and a failed promotion-intent ledger append, three caller cancellations during successful/failing asynchronous discard, and failed-discard reentry after rejection, evaluation failure, or ledger failure.

| Check | Pinned main `b05e0ef` | Corrected implementation |
| --- | --- | --- |
| Frozen cases | 7 failures, no errors/skips | 7 passes, no errors/skips |
| Cancellation during rejection/ledger rollback | Round exits before restore is terminal | Lock and candidate retained until terminal restore |
| Error after successful ledger rollback | Caller cancellation can obscure prior ledger error | Original ledger error object preserved |
| Failed-discard reentry | Further hook work can begin | Rejected before any hook or provenance step |

The [first baseline](../results/candidate-discard-recovery-v1/initial-baseline/) is retained unchanged. The [matched baseline/candidate replay](../results/candidate-discard-recovery-v1/run-1/) records exact protocol, source, test, runtime, log and JUnit hashes. Its recorded local candidate commit is not published as a remote Git object; source hashes identify the implementation used during validation. This is seven deterministic scheduling fixtures, not an incidence estimate or a performance benchmark.

```bash
python -m pip install -e . pytest==8.4.2
python scripts/reproduce_candidate_discard_recovery_v1.py --output /tmp/vare-discard-replay
```

The maintained nearby engine, operator-interlock and checkpoint-bound-promotion selection passes 43 tests. The default full CPU suite passes 607 tests; Ruff 0.15.12 and package-wide mypy 1.19.1 pass. [Validation logs and hashes](../results/candidate-discard-recovery-v1/validation/) are retained. CI repeats the frozen comparison on Python 3.12 and uploads fresh records, alongside the existing inference/evaluation reproductions and Python 3.11–3.13 suite.

## Scope and remaining work

This extends the same VARE outer-loop transaction contract, not training or serving machinery. The [primary-source literature/OSS comparison](evaluation-transaction-integration-v1.md#literature-and-oss-decision) remains the applicable research decision. No model capability, new learning algorithm, throughput gain, or recursive improvement is claimed.

Quarantine is local to one loop in one process. It is not a shared lock, durable supervisor, OS sandbox, distributed fence, or verification of external recovery. A detached or noncooperative hook can still require external intervention; no hard cleanup deadline is promised. The gate activates on observed discard failure only, not arbitrary unreported trainer mutation, process loss, or an interrupted promotion hook. Those remain separate contracts. No consumed confirmation prompt, historical protocol/result, reward threshold, or checkpoint was changed.
