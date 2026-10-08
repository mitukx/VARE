# Executable engineering environment factory

VARE can treat repository-level engineering work as a verifiable environment without owning a large training cluster. A task pack specifies a source repository, immutable base revision, natural-language objective, trusted evaluator commands, optional performance metrics, protected evaluator files, and resource limits. The model/agent edits only an isolated workspace; the evaluator owns pass/fail and reward.

```text
failure cluster
  -> task catalog / curriculum
  -> isolated repository workspace
  -> coding or research agent
  -> trusted executable tests + metrics
  -> immutable diff + provenance
  -> verified trajectory
  -> replay / candidate training
```

## Why this layer exists

The expensive learner is intentionally outside this package. Environment development, verifier hardening, trajectory analysis, task generation, failure mining, and most debugging are CPU-oriented. Large-model rollouts can be supplied by a remote endpoint, while local deterministic fixtures validate the control plane.

## Task-pack contract

A `task.json` contains:

- a local source directory or Git clone URL plus base revision;
- one or more evaluator-owned command vectors (never model-generated shell text);
- optional JSON-emitting performance metrics with direction and regression gates;
- SHA-256 protected paths for evaluator assets that live inside the candidate workspace;
- wall-time, CPU-time, output-size, and best-effort memory ceilings;
- an explicit correctness/performance reward weight.

For public upstream tasks, prefer a Git revision reference plus independent evaluator assets rather than copying source into VARE. Historical tasks should pin immutable revisions. Results should include the candidate Git diff and task-spec SHA.

The command output limit is enforced while stdout/stderr are drained, with a distinct output-overflow result. The limit applies separately to each captured stream. This bounds retained output buffers; it does not impose a hard process RSS, CPU, disk, or network quota. See the [output budget regression report](environment-output-budget-report.md).

## Current evidence boundary

`benchmarks/smoke/stable_logsumexp` is intentionally synthetic. It exists only to test the harness: the broken baseline fails, an oracle patch passes, evaluator tampering fails closed, and the candidate diff is retained. It is **not** evidence of coding-agent capability.

The next evidence level is a task pack made from real repository histories with evaluator-owned regression tests. Only after that should model comparisons or RL claims be reported.

## Smoke run

```bash
vare env-smoke \
  --spec benchmarks/smoke/stable_logsumexp/task.json \
  --oracle-patch tests/fixtures/oracle/fix_stable_logsumexp.py \
  --output-dir artifacts/env-smoke
```

For an already edited workspace:

```bash
vare env-verify --spec path/to/task.json --workspace path/to/workspace --output result.json
```

## Security boundary

The current CPU runner is a research harness, not a hostile-code sandbox. It uses subprocess time/resource limits and protected-file hashes, but malicious candidate code can still attack its host process or exploit language/runtime behavior. Untrusted open-ended agents should run inside an external container/VM sandbox with evaluator assets mounted read-only. This boundary is explicit rather than hidden behind a strong "sandbox" claim.

## First historical task seed

`benchmarks/seeds/rvl_static_parity` references an immutable pre-fix revision of a real RL systems repository and keeps its regression evaluator outside the candidate checkout. The task targets a previously observed sampler/learner behavior-policy mismatch. It is committed as an **E1 task seed**, not as measured agent evidence: the current evaluator is a narrow external regression contract, and the stronger runtime numerical calibration is now retained under `benchmarks/historical/` and described in `docs/evidence.md`.
