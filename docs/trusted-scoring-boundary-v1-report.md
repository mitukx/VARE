# Trusted scoring boundary v1

## Result

The synthetic E0 probe passed on 2026-10-19 using Colima's Docker runtime on
Apple Silicon. The new `TrustedJSONScorer` keeps expected JSON outputs in the
operator process and launches each candidate request in a fresh container with
only a readonly workspace mount. The runner and protocol are pinned in
[`trusted_scoring_boundary_v1.lock.json`](../protocols/trusted_scoring_boundary_v1.lock.json).

| Check | Outcome |
| --- | ---: |
| Known-good candidate | 2/2 cases passed |
| Candidate returning forged `passed`/`score` fields | 0/1; rejected as wrong output |
| Candidate reading operator-side synthetic grader canary | No; candidate observed `null` |
| Candidate mounts | Exactly one readonly `/workspace` bind mount |
| Managed containers remaining after run | 0 |
| Overall audited decision | `PASS_E0_LIMITED_JSON_IO_BOUNDARY` |

The measured per-container wall times were 0.262s and 0.228s for the positive
cases, and 0.183s for the forged-score control. The canary was generated from
256 random bits for this run; only its SHA-256 is retained, and its plaintext
was deleted after the probe. The report does not treat these
synthetic timings as task throughput. The environment was Python 3.12.12,
Docker client/server 29.9.1/29.5.2, Colima using macOS Virtualization.Framework,
and the locally present image
`python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`.
The scoped tests used pytest 8.4.2 and Ruff 0.15.12.
Container limits were 256 MiB, 0.5 CPU, 12 PIDs, 15 seconds, and 4,096 output
bytes.

## Reproduction

From the repository root, choose a new output directory because the runner
preserves and refuses to overwrite prior attempts:

```bash
python scripts/verify_trusted_scoring_boundary_v1.py \
  --output results/trusted-scoring-boundary-v1/run-N-colima-m1
python scripts/audit_trusted_scoring_boundary_v1.py \
  results/trusted-scoring-boundary-v1/run-N-colima-m1 \
  > results/trusted-scoring-boundary-v1/run-N-colima-m1/audit.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_container_isolation.py tests/test_trusted_scoring.py \
  tests/test_sandboxed_repository.py tests/test_environment_campaign.py \
  tests/test_environment_catalog.py
python -m ruff check src/vare/container_isolation.py src/vare/trusted_scoring.py \
  tests/test_container_isolation.py tests/test_trusted_scoring.py \
  scripts/verify_trusted_scoring_boundary_v1.py \
  scripts/audit_trusted_scoring_boundary_v1.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed in the recorded environment
because an unrelated installed pytest plugin imports incompatible ANTLR
versions. It does not disable project tests. The first attempted smoke was
preserved in `results/trusted-scoring-boundary-v1/run-1-colima-m1/`; it failed
before launching a container because the runner had not added the repository's
`src/` directory to Python's import path. The runner was corrected before the
passing run, and both attempts remain available for review.

Evidence for the passing run is retained in
[`run-5-colima-m1`](../results/trusted-scoring-boundary-v1/run-5-colima-m1/):
the raw observations, lock copy, manifest, and independently recomputed audit.
The manifest and auditor verify source and script hashes. The audited result is
same-host reproduction, not external human review.

`run-2-colima-m1/` is retained as an intermediate pass. It predates the
runner's explicit checks that locked runtime pins, limits, and acceptance
criteria match the executable configuration. `run-3-colima-m1/` used a
fixed, publicly visible canary; run 4 superseded it using an ephemeral canary.
Run 5 also removes the canary if the probe fails, so use it as the current
protocol-valid measurement. The earlier import-path startup failure is
retained in run 1.

## Limits

This proves only the bounded JSON-I/O path against a synthetic canary and
synthetic correct/incorrect controls. It does not establish general sandbox
security, solve a repository-level RL/LLM engineering task, test hidden
repository suites, measure a model, or qualify VARE live candidate execution.
The existing `DockerRepositoryEvaluator` still runs candidate code in the same
container namespace as repository tests and task data; it remains ineligible
for confidential held-out tasks. Live untrusted repository execution remains
**BLOCKED** pending a materially stronger trusted execution and scoring
boundary.
