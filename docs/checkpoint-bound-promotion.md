# Checkpoint-bound evidence and fail-closed promotion recovery (experimental)

VARE's original HMAC attestation protects evaluation report bytes but does not prove that the grader evaluated the model weights later promoted. Similarly, a `promotion_intent` hash-chain entry alone cannot atomically update a model/checkpoint store. This extension makes a specific CPU-only failure mode testable, **without claiming an OS-level security boundary or distributed atomicity**.

## Opt-in strict attestation

`EvidenceTrustRoot(..., evaluation_set_sha256=..., promotion_nonce=..., strict_checkpoint_binding=True)` enables a new signed receipt domain (`vare.promotion-evidence.checkpoint.v2`). The independent grader must call `attest(report, checkpoint_sha256=...)` using the SHA-256 of the **actual bytes of the model checkpoint loaded for that evaluation**, computed inside a trusted privilege domain. The receipt binds that digest, the operator-pinned held-out dataset digest, and a 16+-character single-round intent nonce, as well as the existing protocol, evaluator, run ID and all score fields.

This **does not automatically hash model weights**: the caller is responsible for obtaining the digest independently. The candidate must not control the checkpoint hashing service or the signing key. `checkpoint_digest(report)` returns a digest only after verifying the receipt. A new round needs a new nonce and fresh root; the durable journal enforces uniqueness.

## Local durable promotion journal

`PromotionJournal(db_path, initial_policy_id=..., initial_checkpoint_sha256=...)` stores an operator-controlled authoritative checkpoint pointer and a promotion transaction register in SQLite (`synchronous=FULL`, local filesystem). Transactions transition:

```text
     prepared (durably recorded BEFORE invoking hook.promote)
       |                              |
       | verified candidate live      | independently verified incumbent live
       v                              v
     committed                      aborted
```

Any restart with a `prepared` transaction is **fenced** by `assert_clean`. The trusted recovery operator must retrieve the *actual persisted active checkpoint* and call `reconcile(tx_id=..., observed_policy_id=..., observed_checkpoint_sha256=...)`: only an exact incumbent match aborts, and only an exact candidate match commits; everything else remains blocked. This does not silently repair a corrupted checkpoint or recreate deleted model/optimizer state.

## CapabilityLoop integration

Pass `promotion_journal=...` and `checkpoint_hasher=trusted_snapshot_digest`, alongside the strict `evidence_trust_root`. VARE then:
1. checks the existing durable pointer against the trusted live checkpoint before any rollout;
2. verifies the signed checkpoint digests against independently hashed incumbent and candidate checkpoints after evaluation;
3. prepares a durable transaction with the unique intent nonce and evidence digest;
4. appends the existing `promotion_intent` audit event before mutating live policy state;
5. calls the downstream promote hook and confirms the active policy and candidate digest;
6. atomically advances the journal pointer and marks the transaction committed.

If an audit precommit fails before the promote hook, the candidate is discarded and the verified incumbent is used to abort the journal intent. A partially failing promote hook or a crash before SQLite commit **leaves a prepared transaction**; subsequent rounds cannot run until independently reconciled.

The `checkpoint_hasher` must hash the immutable persisted artifact (including all model weights and any promotion-relevant state) by policy identity, not a candidate-supplied identifier or `repr` of an object. On real systems, signing, hashing, checkpoint storage, SQLite state, and operator recovery belong in trusted separate processes. Promoted runtime state and SQLite state cannot be committed in one atomic transaction merely by calling two APIs.

## Boundaries and limitations

- The durable journal is an *uncertainty fence with recoverable state*, not a distributed two-phase commit or a guarantee that the external hook restored full optimizer/RNG/module-mode state. Separate durable checkpoint/snapshot transports, an actual atomic checkpoint pointer swap, and startup reconciliation are needed for strong end-to-end guarantees.
- A MAC cannot certify an untrusted grader. The signer must be isolated from the candidate. No protection if the secret or artifact store is compromised.
- The journal provides local SQLite crash-consistency, not adversarial-host tamper resistance, network replication, or a cryptographic log anchor.
- The existing legacy attestation and `CapabilityLoop` calls are unchanged unless strict bindings and journal are explicitly supplied; production security mode should enable them.
- Tests use simulated CPU states and injected exceptions. There is no real-model improvement, production deployment evidence, distributed run, or measured attacker resistance.

## Reproduce

```bash
python -m pytest -q tests/test_attestation.py tests/test_checkpoint_bound_promotion.py tests/test_safety_interlock.py tests/test_rvl_evaluation_promotion_e2e.py
```
