# Trusted evaluator isolation: Docker-backed candidate and grader roles (E0)

This implementation adds an **opt-in** container boundary to VARE's existing experimental environment campaign and checkpoint-bound promotion path. The candidate receives only a writable workspace; the evaluator receives only read-only checkpoint blobs, a frozen held-out set, and a signing key. The two roles use separately pinned images. The trust boundary is a **separately administered Docker daemon or VM**, not the Python subprocess wrapper. Do not run candidate agents with access to the host Docker socket, daemon credentials, operator key, or scoring environment.

## Requirements

- Docker Engine / Docker Desktop with a trusted, operator-controlled daemon and its standard seccomp filtering enabled. Use a rootless daemon or dedicated VM where possible; do not assume a privileged Docker daemon is a robust hostile-code sandbox.
- Images are **already built and loaded**, pinned as `image-name@sha256:<64 lowercase hex>` (no mutable tags and no runtime pulls).
- The scorer image contains the installed `vare` Python package and no candidate-supplied code. Pin the worker source digest to the prebuilt image's actual module bytes; the receipt must match the independently configured `EvidenceTrustRoot`.
- Operator-owned directories and a signing key are inaccessible to the candidate via any other host path. Container UID/GID must have *read access* to trusted evaluator artifacts. The candidate's one workspace must be writable by the configured UID/GID.
- Trusted operator is responsible for building images, configuring Docker security policies, mounting artifacts, signing leases, granting approvals, killing processes, and reconciling journal state after a crash.

## Candidate agent: one writable mount only

Use the existing environment campaign with an opt-in pinned agent image:

```bash
vare env-campaign \
  --catalog-root benchmarks/smoke \
  --run-root artifacts/isolated-campaign \
  --agent-argv-json '["python","/app/agent.py","{prompt}","{workspace}","{step}"]' \
  --isolated-agent-image 'local/vare-agent@sha256:<actual-64hex-image-digest>' \
  --agent-timeout-s 60 --isolated-memory-mb 512
```

These arguments are evaluated **inside** the candidate container, unlike the legacy `CommandWorkspaceAgent` where they are host commands. The runner uses `--pull=never`, `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges:true`, numeric unprivileged UID/GID, a private process namespace, memory/PID/CPU quotas, and bounded stdout/stderr; it mounts only the selected workspace at `/workspace`. A unique per-run container name enables forced removal on exit, timeout, output overflow, and cancellation. No secret or Docker socket is mounted.

This mode changes **agent execution only**. The current `EnvironmentCampaignRunner` still evaluates repository tasks with its existing trusted local verifier; that verifier must also be isolated in a stronger deployment. Do not interpret `--isolated-agent-image` as hardening the entire campaign.

## Separate trusted promotion evaluator

Construct `IsolatedEvaluationHooks` around existing `LoopHooks`. It replaces only `evaluate(policy_id)` with a Docker-run, signed, checkpoint-bound reference evaluator. The trusted eval container receives:

- checkpoint registry and content-addressed blobs at `/operator/checkpoints` — bind mount **readonly**;
- pinned held-out JSON at `/operator/heldout.json` — **readonly**;
- exactly 32-byte signing key at `/operator/key` — **readonly**.

It receives **no candidate workspace mount**. The trusted worker now opens the checkpoint registry SQLite connection in query-only `mode=ro`; it runs no schema initialization while evaluating. `DockerIsolationRunner.run_evaluator` fails closed if its process fails, receipt is absent/forged, policy ID differs, or evaluation output has an unexpected shape.

Example control-plane wiring (operator-only pseudo-configuration; substitute real pinned digests and trusted paths):

```python
from pathlib import Path
from vare.attestation import EvidenceTrustRoot
from vare.checkpoint_artifacts import TrustedCheckpointStore
from vare.container_isolation import DockerIsolationRunner, IsolationLimits
from vare.isolated_hooks import IsolatedEvaluationHooks
from vare.promotion_journal import PromotionJournal
from vare.engine import CapabilityLoop

# Construct the operator's EvidenceTrustRoot with pinned evaluator/protocol/
# held-out SHA-256, unique run ID and promotion nonce, strict_checkpoint_binding=True.
# The signed receipt is issued INSIDE the evaluator container, never by learner code.
isolated_hooks = IsolatedEvaluationHooks(
    underlying=learner_hooks,  # existing LoopHooks
    runner=DockerIsolationRunner(),
    evaluator_limits=IsolationLimits(
        image="local/vare-grader@sha256:<actual-64hex-image-digest>"
    ),
    trust=trust_root,
    checkpoint_store=Path("/operator-only/checkpoints"),
    heldout_set=Path("/operator-only/heldout.json"),
    key_file=Path("/operator-only/hmac-key"),
)
checkpoint_store = TrustedCheckpointStore("/operator-only/checkpoints", read_only=True)
loop = CapabilityLoop(
    hooks=isolated_hooks,
    verifier=training_verifier,
    evidence_trust_root=trust_root,
    checkpoint_hasher=checkpoint_store.digest,
    promotion_journal=PromotionJournal(
        "/operator-only/promotions.sqlite",
        initial_policy_id=initial_id,
        initial_checkpoint_sha256=checkpoint_store.digest(initial_id),
    ),
)
```

This API is a production-oriented boundary **contract** only. The reference scorer in `vare.trusted_evaluation` handles a toy JSON linear model, not RVL/LLM checkpoints. To deploy with an RVL learner, export real immutable checkpoint bytes, implement a matching trusted *model-aware* grader in the pinned evaluator image, and verify actual loaded bytes. The checkpoint manifest and trained model must not diverge.

## External stop conditions

The supervisor calls `docker rm -f <random-run-name>` even when a Docker client fails, times out or is cancelled. A missing container is treated as successful removal; a failure to confirm removal is fatal. This cleanup is **not** an externally durable kill service: if the Python controller, host or daemon crashes, a container might remain. A separately hosted watchdog/controller with persistent inventory and remote daemon access must enforce lease expiration, kill orphan containers, and revoke credentials. A daemon outage may make cleanup impossible.

## Verification and limitations

```bash
python -m pytest tests/test_container_isolation.py tests/test_trusted_checkpoint_evaluator.py -q
```

The CI tests inspect the exact Docker command construction, evaluate signed-response contracts, reject symlinks/unpinned images, and simulate timeouts/output overflow/forced cleanup with a fake Docker executable. **They do not execute real Docker containers, demonstrate kernel-level isolation, measure attacker escape resistance, run a real RL learner, or establish GPU improvement.**

Before real agent deployment, conduct a real Docker/VM smoke test with an unprivileged container verifying that the model cannot read operator secrets, reach the network/daemon, modify the grader or leave orphan processes; independently audit UID mappings, mounts, daemon security settings and image supply chain. Use the external supervisor to test immediate termination, not merely the phase-boundary safety interlock. Any unavailability of Docker or isolation prerequisites must fail closed.
