# Same-interpreter grader audit follow-up

Date: 2026-11-01. This is a synthetic-fixture calibration and a threat-model finding; it is not
production-repository validation or evidence of an arbitrary-Python trust boundary.

## Environment and identity

- Host: Apple M1 Pro, arm64, 32 GiB unified memory, macOS 27.0.1
- Python: 3.12.12
- Colima: 0.10.3, aarch64 VM, macOS Virtualization Framework, virtiofs
- Docker CLI: 29.9.1; Engine: 29.5.2
- Calibration image: `localhost:5000/vare-lp-grader@sha256:6ff5cba42240c57340c215e78d28f8f94624ed57cc0d5b92b548b9dfb32796da`
- Sandbox config digest: `c43361706944378d961b2de03d67653d5cf17dcf31e95de496bab6a557f558ff`
- Runner SHA-256 before/after: `9b2821668d5b08507d110253030b35a2db79a734a6a2c5ab8353eb8766ae5b6b`
- Verifier-source bundle SHA-256 before/after: `d6440bdc1653035b12bf20b8c674a547a99e4243e87da6f214267f77ff7356e2`
- `source_identity_stable`: true; `sandbox_configuration_stable`: true

The verifier-source bundle includes the trusted judge and runner, the backend implementation,
`Dockerfile`, the calibration harness and the frozen v1 `RepoEnv`. Each grade checks those pins
before launch and again on return. A final calibration-level check also refuses `all_verified` if
source bytes or the sandbox configuration change during the full A3 run.

## Findings

The live Docker/Colima calibration verified A3.1–A3.10, completed in 28.19 seconds, and left zero
workspaces. The report records successful attestation: uid 65534, two visible processes, writable
task workspace, and no observed host canary, environment canary, evidence directory, interpreter
write access, root write access, network route, non-loopback interface, host loopback connection,
or Docker socket.

Three same-interpreter residual candidates pass when static scanning is disabled:

- `runner_key_exfil_and_frame_forgery`
- `runner_closure_patch_via_gc`
- `always_equal_return_value`

The current scan quarantines these exact fixtures. This shows only that those literal fixtures match
the current scan patterns. Candidate Python shares an interpreter with the runner and can reach
the verdict key; obfuscated variants are not ruled out. Do not use this mechanism to claim security
against arbitrary Python or to support confirmatory capability claims until candidate execution
and trusted test/verdict logic are separated by an appropriate black-box contract.

The calibrated image contains `pytest==8.4.2`. The focused host suite ran with the image digest
above and reported 60 passed, 10 skipped. `test_live_docker_a3_calibration` passed; the ten skips
are Linux bubblewrap user-namespace probes, unavailable on this macOS host.

## Reproduction

From the repository root with the recorded image available in the Colima Docker context:

```sh
IMAGE='localhost:5000/vare-lp-grader@sha256:6ff5cba42240c57340c215e78d28f8f94624ed57cc0d5b92b548b9dfb32796da'
DOCKER_CONTEXT=colima PYTHONPATH=src python3.12 scripts/learning_pipeline_grader_calibration.py \
  --backend docker --image "$IMAGE" --work-root "$HOME/.cache/vare-grader-calibration-replay" \
  --timeout 120 --out /tmp/vare-grader-calibration-replay.json
DOCKER_CONTEXT=colima VARE_GRADER_IMAGE="$IMAGE" VARE_GRADER_WORK_ROOT="$HOME/.cache" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3.12 -m pytest -q -rA \
  tests/test_learning_pipeline_isolated_grading.py tests/test_learning_pipeline_trusted_grading.py
```

Use a new report output path for each replay; the calibration command refuses to overwrite an
existing report. Rebuilding the image creates a new identity: push it to a loopback registry, use
its digest, and retain the newly emitted digest and config record.

`SHA256SUMS` covers the retained JSON and pytest log in this directory.
