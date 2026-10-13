# Attested promotion evidence (opt-in)

The promotion gate normally verifies numerical/paired-evidence consistency, not evaluator ownership. A candidate-controlled hook can fabricate internally consistent reports. The optional `EvidenceTrustRoot` closes that **report-authentication** gap only when the grader is an independent trust domain.

## Boundary and deployment

1. Freeze the evaluation protocol and grader. The operator pins their SHA-256 values and issues a unique unpredictable run ID per study, with a random 32-byte-or-longer HMAC key.
2. Run the trusted grader outside the candidate workspace/process/privilege domain, with protected tasks/results and the secret unavailable to any trainable agent. The trusted service computes reports and calls `trust.attest(report)`. Do **not** let candidate hooks sign their own reports.
3. Construct `CapabilityLoop(..., evidence_trust_root=verification_root)` in the controlling service. It fails closed on missing, changed, foreign-run, foreign-grader, or foreign-protocol receipts. Reject results are discarded through the existing transactional path. Use unique policy IDs bound to immutable checkpoint hashes.
4. Do not copy the signing secret into Git, logs, model prompts, subprocess environment, candidate checkout, or the public dataset. Consider separate asymmetric signatures/KMS and nonce/issuance registries for a multi-host production deployment. The current HMAC design presumes that both trusted signer and trusted verifier can safely hold the same key.

## Limitations

This is a **CPU-tested optional contract**, not hostile-code containment. If the caller, grader host, HMAC secret, or operator-pinned identities are compromised, a malicious result can still be signed. Content hashes without trusted external anchors do not authenticate the grader. Signed results can also be repeated *within the same run and policy identity*; the per-run issuance/nonce registry and commit preauthorization are separate future controls. There is no claim of a complete autonomous-agent security architecture, independent grader service, or real-model improvement.

The legacy gate remains permissive unless `evidence_trust_root` is supplied. For security-sensitive deployment, configure this gate explicitly and keep any candidate execution inside an independently enforced VM/container sandbox with network and filesystem restrictions.
