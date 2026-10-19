# Docker isolation boundary verification

## Result

This is an E0 runtime-boundary experiment using synthetic probes in a local Colima VM. It does not execute a language model, coding agent, real repository task, or hidden-test grader. It is not a sandbox security certification.

| Measurement | Initial observation (run 1) | After cleanup-race fix (run 2; run 3 repeats the locked protocol) |
|---|---:|---:|
| Candidate host-canary denial and egress denial | 3/3 | 3/3 |
| Effective container configuration and resource limits | pass | pass |
| Timeout cleanup | 3/3 | 3/3 |
| Controller-cancellation cleanup | 3/3 | 3/3 |
| Output-overflow cleanup | 0/3 (cleanup race reported as failure) | 3/3 |
| Synthetic grader-canary confidentiality | **FAIL: visible 3/3** | **FAIL: visible 3/3** |
| Live candidate execution | **BLOCKED** | **BLOCKED** |

The first run exposed a race between `docker run --rm` and the supervisor's forced removal. The Docker daemon returned “removal ... is already in progress”; the supervisor treated that transient response as cleanup failure even though removal was underway. The bounded retry now retries only that daemon response and treats a confirmed “No such container” as successful cleanup. Regression tests cover this race, ordinary cleanup, timeout, and cancellation.

The candidate container had no network, a read-only root filesystem, all Linux capabilities dropped, `no-new-privileges`, UID/GID 65534, one writable synthetic workspace mount, and observed memory/CPU/PID limits of 256 MiB, 0.5 CPU, and 12 processes. The process-limit probe reached `EAGAIN` at the configured limit. These observations do not establish resistance to kernel or runtime escapes.

The grader probe deliberately used `DockerRepositoryEvaluator`'s mount and resource-flag builder and replaced only the worker entrypoint with a read of a random synthetic canary. The canary was readable in every replicate. Therefore the existing grader boundary exposes the mounted grader-side file to code in that container namespace. This is a concrete confidentiality defect; the experiment does not attempt to hide it or claim hidden-test integrity. Do not run candidate code against confidential evaluation data until the grader boundary is redesigned and independently re-verified.

## Reproduction

Requires the named local Colima Docker context, Docker's built-in seccomp profile, Python 3.12, pytest, and the already cached official Python image pinned by digest. The measured environment was Python 3.12.12, pytest 8.4.2, Colima 0.10.3, Docker client/server 29.9.1/29.5.2, Colima on macOS Virtualization.Framework with Docker runtime, and image `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`. No network pull, paid compute, external credentials, real task, or real secret is used.

From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3.12 -m pytest -q tests/test_container_isolation.py
PYTHONPATH=src DOCKER_CONTEXT=colima python3.12 scripts/verify_docker_isolation_boundary_v1.py \
  --protocol protocols/isolation_boundary_verification_v3.lock.json \
  --output results/isolation-boundary-verification-v1/run-3-colima-m1
python3.12 scripts/audit_docker_isolation_boundary_v1.py \
  results/isolation-boundary-verification-v1/run-3-colima-m1
```

The run directory is write-once. `protocol.lock.json` pins source hashes, runtime settings, probes, repetitions, and decision rules. `manifest.json` hashes the observations and protocol and records the implementation hashes and whether the checkout was dirty. The auditor recomputes the boundary, lifecycle, and grader-canary outcomes from the raw observations. Runs 1 and 2 are retained as negative and corrected observations, respectively.

## What this does not demonstrate

- No real-model inference or repository-level agent task ran; task success, patch validity, tokens, tool calls, wall time, memory use for inference, and cost were not measured.
- No interruption/resume of an agent trajectory was measured.
- The grader canary failure means confidential or held-out tests must not be mounted into this current grader boundary for candidate execution.
- Colima and Docker controls provide defense in depth, not proof against hostile kernel/runtime code or all host access.
- The run is local to this Apple Silicon machine; broader platform behavior is untested.
