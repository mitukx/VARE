# Trusted checkpoint bytes and separate-process evaluation (E0 protocol demonstration)

This extension addresses a gap in the checkpoint-bound promotion path: `CapabilityLoop` accepted an abstract `checkpoint_hasher`, and the test signer previously accepted caller-provided hashes. The new path **copies an actual checkpoint file into operator-controlled content-addressed storage, re-hashes its bytes, and signs a report only after a separate evaluator process loads and scores those bytes**.

## Components and contracts

- `vare.checkpoint_artifacts.TrustedCheckpointStore`: `register_file(policy_id, checkpoint_path)` streams a snapshot into `objects/<sha256>`, publishes via atomic hard link, and durably records the immutable policy-ID-to-hash mapping in local SQLite. `digest(policy_id)` re-reads the complete artifact bytes rather than trusting the declared hash. `read_verified` checks bytes and digest using one file descriptor for small CPU fixtures.
- `python -m vare.trusted_evaluation`: separate-process reference grader which loads a *small, JSON-encoded linear CPU model* from this store, scores a pinned JSON held-out set, and issues a strict checkpoint-bound HMAC attestation. The scoring input is computed by the grader, **not** obtained from an agent-reported metric.
- A `CapabilityLoop` can use `checkpoint_hasher=store.digest` and `promotion_journal=PromotionJournal(...)` together with strict evidence receipts. The integration test exercises the complete store -> subprocess scorer -> signed report -> promotion -> durable journal pointer path.

The reference grader intentionally supports only `{"schema_version":1,"weight":...,"bias":...}` checkpoints and held-out tasks `{"schema_version":1,"tasks":[{"id":...,"x":...,"label":0|1,"family":...}]}`. It is **not** a general ML evaluator and must not be described as proof of post-training capability, agent containment, or secure model serving.

## Trusted owner example

After a trusted backend has exported a complete snapshot to a file (safetensors, archive, or versioned checkpoint), the operator registers it:

```python
from vare.checkpoint_artifacts import TrustedCheckpointStore

store = TrustedCheckpointStore("/operator-only/checkpoints")
digest = store.register_file("policy-0", "/trusted-export/policy-0.snapshot")
assert store.digest("policy-0") == digest
# In the control plane, pass checkpoint_hasher=store.digest
```

For the CPU protocol fixture, invoke the evaluator as an **operator-managed separate process** with these flags:

```bash
python -m vare.trusted_evaluation \
  --checkpoint-store /operator-only/checkpoints \
  --policy-id policy-0 \
  --heldout-set /operator-only/heldout.json \
  --heldout-sha256 <operator-pinned-64hex> \
  --key-file /operator-only/hmac-key-32-bytes \
  --protocol-sha256 <operator-pinned-64hex> \
  --run-id fresh-study-run-001 \
  --promotion-nonce fresh-round-nonce-001
```

The worker computes its own `evaluator_sha256` from the exact module file loaded. The trusted controller pins that module SHA-256, along with the other identifiers in `EvidenceTrustRoot`, before accepting reports. Do **not** let the candidate choose the paths, key, evaluation set, run identity or nonce.

## Security boundaries

A subprocess is **not automatically independent in a security sense**. For untrusted agent-generated code, deploy this grader under a separate OS identity/container/VM with the checkpoint store, HMAC key, scoring code, and held-out tasks inaccessible to the candidate. The demo's same-UID test process demonstrates data flow and protocol correctness only. Protect the interpreter, imported packages and `PYTHONPATH` against candidate modification; avoid `pickle` and dynamic imports of candidate code in the evaluator.

The caller exporting a real model must include the entire promotion-relevant model state (and optimizer/RNG states if rollback requires them) in a deterministic, trusted artifact format, and the actual inference runtime must load **those bytes**. Merely registering a copy while evaluating a different in-memory model would defeat the binding. Large-model loaders may use `artifact_path()`, then independently hash the loaded file before/after usage; that helper does not guard against changes during loading if the filesystem is attacker-writable.

A local SQLite manifest and content-hash file do not protect against a malicious process running with the same filesystem privileges. No remote object store, asymmetric attestation, external KMS, actual constrained model process, or distributed atomic commit is implemented here. This is a CPU E0 end-to-end contract for a real file; stronger deployment isolation and source-specific RVL checkpoint exporters remain separate work.

## Reproduce

```bash
python -m pytest -q tests/test_trusted_checkpoint_evaluator.py \
  tests/test_checkpoint_bound_promotion.py tests/test_attestation.py
```

The tests include: source-file mutation after import, tampering with registered bytes, symlinks, attempting to rebind a policy ID, changed held-out content, signed report tampering, and promotion of the exact checkpoint evaluated by the separate-process reference scorer.
