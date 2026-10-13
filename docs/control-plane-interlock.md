# Operator-authorized control-plane interlock (experimental)

This is an **opt-in, CPU-testable outer-loop safety control**, not a sandbox or proof of isolation. It complements [attested promotion evidence](attested-promotion.md); both must be configured separately when applicable. The repository's existing protected evaluator/runner is explicitly *not* a hostile-code security boundary.

## Concrete threat and enforcement

An agent or hook with uncontrolled authority could continue training after an operator revokes authorization, mutate a local grader, pass fabricated promotion metrics, or launch an agent subprocess with inherited controller credentials.

- `SafetyInterlock.require(action, incumbent_id, candidate_id)` checks the signed operator lease *fresh* before every rollout, candidate training, evaluation (before and after), and candidate promotion. Expiration, mismatched run/incumbent, missing or invalid signed lease, and unapproved candidate all fail closed.
- The operator may pin absolute, trusted evaluator asset paths to SHA-256 values. Their bytes must match each check. This does not itself enforce filesystem access control or guard against a TOCTOU attack.
- The capability loop discards the candidate if evaluation fails or if the final promotion permit is denied. An optional decision ledger now writes `promotion_intent` before invoking the promote hook; a failed intent append discards the candidate without promotion.
- `CommandWorkspaceAgent` now defaults to passing only selected non-secret environment variables to the child. Callers can opt in to more names via `environment_allowlist`. This does *not* prevent filesystem, process, or network access by an untrusted same-user process.

## Operator-controlled setup

The operator must generate a random 32+ byte signing key, issue a unique run ID, and store the signed authorization outside the candidate's accessible filesystem namespace. Use a trusted service/host/UID to sign and distribute leases. The operator can revoke future authorized phases by deleting the lease from the control-plane-only location or by allowing the short lease to expire.

Example **trusted-control-plane-side only**:

```python
import hashlib
from pathlib import Path
import secrets
import time
from vare.safety import SafetyInterlock, sign_control_lease

key = secrets.token_bytes(32)  # persist securely in trusted key management
run_id = "study-001"            # must be uniquely allocated by the operator
control_path = Path("/operator-only/current-lease.json")
grader_path = Path("/operator-only/grade.py")
grader_hash = hashlib.sha256(grader_path.read_bytes()).hexdigest()

control_path.write_text(sign_control_lease(
    key=key, run_id=run_id, incumbent_id="policy-0",
    expires_at=int(time.time()) + 300,
    allowed_actions=("rollout", "train", "evaluate"),
))
interlock = SafetyInterlock(
    control_path, key, run_id,
    protected_assets={grader_path: grader_hash},
)
# Pass `safety_interlock=interlock` to `CapabilityLoop`.
```

A lease with actions only `rollout/train/evaluate` deliberately **cannot promote**. After independent evaluation, a trusted operator/service must issue a fresh lease with `promote` and `approved_candidates=(candidate_id,)`. If the candidate is not approved, the engine rejects the promotion and discards it. Do not sign promotion authority from candidate-controlled reports without independent verification.

## Limits and required external components

- **External kill switch:** removing the lease blocks the *next* guarded phase, not a rollout/tool process already running. Immediate termination requires a separate OS/container/VM supervisor that can terminate the process tree and revoke credentials.
- **Capability firewall:** arbitrary code run as the same user can still access resources beyond its stated command vector and potentially read the controller's process memory, files, or credentials. Enforce filesystem mounts, UID separation, denied network egress, process isolation, and resource budgets in an independent supervisor.
- **Tamperproof evaluator:** an on-disk hash without a separately controlled allowlist and OS-enforced read-only mount is not a security boundary. For hostile workloads run the grader separately from candidate code with protected inputs/outputs and immutable build identities.
- **Revocation replay:** if candidate code can replace the control file with a previously signed unexpired lease, revocation is not reliable. That is why the control path and its entire parent directory must be outside its write domain. A multi-host service should use a monotonic issuance counter or an online authorization service.
- **Logging:** the local hash-chain ledger detects inadvertent edits but is not independently anchored. `promotion_intent` before mutation narrows the audit gap, but does not create a distributed/atomic two-phase commit.
- **Testing:** the CPU tests simulate denials, revocation, mutable grading assets, and ledger failure. They do not establish a protected operating-system boundary or post-training capability improvement.

## Validation

```bash
python -m pytest -q tests/test_attestation.py tests/test_safety_interlock.py \
    tests/test_environment_campaign.py tests/test_rvl_evaluation_promotion_e2e.py
```

The interlock is deliberately optional for backwards compatibility. Production operators should explicitly enable it, pin their trusted evaluator, protect the signing key, and run model code only in a separately sandboxed execution environment.
