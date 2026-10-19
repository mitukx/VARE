# Sandboxed repository evaluation: no host-side candidate test execution (E0)

The existing environment campaign could isolate an agent in Docker while
`CatalogEnvironmentVerifier` still called `ExecutableEvaluator` on the
controller host. Since `run_command` runs a task's test argv under the
environment of its process, this is unacceptable as an isolation boundary
for untrusted candidate code.

This change supplies `DockerRepositoryEvaluator`, an isolated backend for the
same catalog/evaluation contract. The **paired-container CLI mode refuses to
launch unless both the agent image and the repository evaluator image are
configured**. Legacy `env-campaign` and `env-verify` still exist for
trusted local development; they are explicitly not hostile-code safe.

## Execution flow

1. The trusted host creates an ephemeral candidate workspace. The *candidate
   agent* runs in an isolated image with just that writable workspace.
2. The trusted host computes a canonical task-spec hash and launches the
   *repository grader* in a separately pinned image via
   `DockerIsolationRunner`. Only the candidate workspace and the task-spec
   root are mounted into this container, both **readonly**.
3. The grader verifies the task-spec SHA-256 and makes a private copy of the
   candidate checkout under a container-local `/scratch` tmpfs. It copies
   symlinks *as links* to avoid dereferencing adversarial links while copying.
4. The grader executes the existing `ExecutableEvaluator` entirely inside
   the constrained grader container, recomputes protected-workspace hashes,
   and serializes `EnvironmentResult`.
5. The controller verifies the result's schema, finite bounded score,
   task identity, protocol hash, policy ID/version, and command success
   consistency. Nonzero status, invalid output, missing Docker runtime,
   timeout, output overflow, or uncertain cleanup all fail closed.

Both containers are run with no network, readonly rootfs, no-new-privileges,
all caps dropped, numeric non-root identity, resource/PID/time quotas and
bounded output. The Docker daemon/VM and pinned images must be operator-managed.
The repository grader has no mounted signing key or checkpoint registry.

## Paired Docker usage

```bash
vare env-campaign \
  --catalog-root benchmarks/smoke \
  --run-root artifacts/isolated-campaign \
  --agent-argv-json '["python","/app/agent.py","{prompt}","{workspace}","{step}"]' \
  --isolated-agent-image 'local/agent@sha256:<actual-image-digest>' \
  --isolated-evaluator-image 'local/repository-grader@sha256:<actual-image-digest>' \
  --agent-timeout-s 60 \
  --isolated-evaluator-timeout-s 120
```

The evaluator image must contain a trusted installed VARE package, a Python
interpreter, `git`, and any *frozen* grader dependencies. The held-out task pack
must be readable by the configured UID; the candidate checkpoint must be
readable by that UID. No image is pulled at runtime, and the Docker socket
must not be made available inside either role.

`/scratch` is an ephemeral 256 MiB tmpfs (in addition to `/tmp`).
Large repositories and tests that need more space fail closed by default;
extend through reviewed, bounded resource configuration rather than mounting
host writable directories. Candidate workspaces are mounted readonly in
the grader to prevent grader-driven host changes. The evaluated copy preserves
a git working tree for the existing diff and regression checks.

## Known limitations (important)

**This does not keep hidden grader files confidential from untrusted code
running inside the grader container.** The container mounts the task-spec
directory readonly, and tests may import or execute candidate code within that
container. Malicious candidate code could inspect other readonly files in that
same namespace, influence tests, and potentially encode data into its outputs.
The principal achieved boundary is **preventing direct execution of those
tests on the controller host**. Preventing evaluation-data leakage or score
manipulation requires further separation of hidden oracle assets from candidate
execution (e.g. externally controlled test oracle/service, independent scoring
of structured artifacts, or a separate VM with narrow protocol), plus signed
evaluation results.

The trusted grader image must not bundle operator signing keys or secret
held-out answers not intended for the candidate process's filesystem namespace.
The task spec currently remains visible; code-based correctness tests inherently
execute candidate-written code in their container. Docker alone cannot
guarantee containment against kernel escape vulnerabilities. Use an
independently controlled VM/daemon and repeat real-container boundary tests.

The grader does not apply a signature to its repository evaluation payload;
integrity rests on the trusted grader image, Docker isolation, and controller's
provenance/schema validation. The result is E0 control-plane correctness, not
a security penetration-test result or a real-model performance result.
An abrupt **host or daemon crash** can still strand containers; a persistent
external inventory/watchdog is future work.

## Limited trusted JSON-I/O scoring boundary

`DockerRepositoryEvaluator` remains unsuitable for confidential held-out tests:
it runs candidate code in the same container namespace as the repository tests
and task data. Do not use it for an untrusted candidate or claim hidden-test
integrity. A separate, narrower path is available through
`vare.trusted_scoring.TrustedJSONScorer` for tasks that can be expressed as
bounded JSON requests and exact JSON responses. The operator supplies the fixed
candidate command and holds the expected responses; each request runs in a new
container with only the candidate workspace mounted readonly. The scorer sends
the request over bounded stdin and compares the response itself. Candidate
claims such as `{"passed": true}` have no scoring meaning. Candidate output,
including invalid or failing output, is summarized by a digest in the report.

This is a limited interface boundary, not general sandbox certification. It
does not run repository test suites, prevent inference from public requests, or
establish model coding ability. Docker/Colima, its kernel boundary, operator
command, and host remain trusted. Live untrusted repository execution stays
blocked. The E0 synthetic protocol below checks only that this narrow path
cannot read an operator-side canary through a mounted path and rejects a
forged score response.

Reproduce the frozen synthetic probe and audit its evidence with:

```bash
python scripts/verify_trusted_scoring_boundary_v1.py \
  --output results/trusted-scoring-boundary-v1/run-N-colima-m1
python scripts/audit_trusted_scoring_boundary_v1.py \
  results/trusted-scoring-boundary-v1/run-N-colima-m1
```

The runner refuses to overwrite an existing evidence directory and verifies
the source/script hashes in the protocol lock before it starts. A passing E0
result does not authorize `DockerRepositoryEvaluator` for hidden repository
tests or change the live-execution NO-GO status.

The first attempted run is retained at
`results/trusted-scoring-boundary-v1/run-1-colima-m1/`; it records a runner
startup failure caused by the source tree not being on `sys.path`. The first
passing smoke at `run-2-colima-m1/` was superseded when runtime/limit pins were
made executable checks; `run-3-colima-m1/` was then superseded to use a random
canary and omit its plaintext from the evidence bundle; `run-4-colima-m1/` was
superseded after adding cleanup for failed probes; `run-5-colima-m1/` was
superseded after preserving raw stdout digests before UTF-8 decoding. The
current successful same-host smoke is `run-6-colima-m1/`; its independently
recomputed decision is
`PASS_E0_LIMITED_JSON_IO_BOUNDARY`. The two positive cases passed, the
forged-score response scored 0/1, the synthetic canary was unreadable, and no
managed containers remained. This records a software boundary check only. It
does not show an agent solving an engineering task.
See [the run report](trusted-scoring-boundary-v1-report.md) and its locked
[protocol](../protocols/trusted_scoring_boundary_v1.lock.json).

## Test contract

```bash
python -m pytest -q tests/test_sandboxed_repository.py \
  tests/test_container_isolation.py tests/test_environment_campaign.py \
  tests/test_environment_catalog.py
```

CI constructs and validates the proposed command/mounts with no real Docker
daemon. A small trusted `true` fixture exercises the worker protocol locally;
**untrusted code is not launched on the host in tests**. Real Docker
integration including the secret-probe, network denial and orphan-cleanup
tests remains required before hostile agents are admitted.
