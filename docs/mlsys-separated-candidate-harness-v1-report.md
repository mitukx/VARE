# MLSys separated candidate harness v1

## Result

The frozen MLSys coding adapter now resolves to VARE's proposed
`vare.separated_candidate_harness.grade_function` implementation. Candidate
functions run through the existing digest-pinned Docker isolated-call runner;
the task descriptor is selected from a canonical allowlist, the frozen pack
manifest is verified before and after call materialization, and hidden expected
outputs remain in the parent process. Unknown task IDs and descriptor or pack
integrity mismatches fail closed. The old host-process shim is no longer used
when the proposed implementation is available.

The consumed global-gradient-clipping task was used only as a harness
calibration. Its frozen nested-`zip` checker accepts an empty result. A
reproducible negative control confirms this behavior; the isolated grader
rejects the same candidate, and the frozen oracle passes. Raw outputs, source
hashes, runtime metadata, and an inventory manifest are retained in
[`results/mlsys-separated-candidate-harness-v1/run-1/`](../results/mlsys-separated-candidate-harness-v1/run-1/).

This is **not** a model update, agent rollout, new task evaluation, or held-out
capability result. Consumed v1 task graders and oracle fixtures are used only
to check that the isolated adapter can preserve their grading semantics. The
dataset-standardization timing adapter additionally checks candidate input
immutability inside the isolated call process.

## Verification

Reproduce the calibration on Apple Silicon with Docker available:

```bash
DOCKER_CONTEXT=colima PYTHONPATH=src python \
  scripts/verify_mlsys_separated_candidate_harness_v1.py \
  --output results/mlsys-separated-candidate-harness-v1/run-1
PYTHONPATH=src python scripts/audit_mlsys_separated_candidate_harness_v1.py \
  results/mlsys-separated-candidate-harness-v1/run-1
```

The output directory must not already exist. The verifier runs the negative
and oracle controls in Docker and records its runtime version and elapsed time.
The auditor checks every artifact hash, relevant source hash, claim label, and
source commit identity.

The regression suite also exercises the frozen v1 benchmark's baseline and
oracle fixtures through the new adapter. No claim is made about an LLM, policy
optimization, held-out task success, or training cost.
