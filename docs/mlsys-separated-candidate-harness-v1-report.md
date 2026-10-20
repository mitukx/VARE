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

The consumed row-log-softmax task is used only as an adapter calibration. The
frozen checker rejects a malformed empty result, the isolated grader rejects
the same candidate, and the frozen oracle control passes. This verifies the
adapter path without claiming a checker defect or new task result. The final
raw outputs, source hashes, runtime metadata, and inventory manifest are
retained in [`results/mlsys-separated-candidate-harness-v1/run-3/`](../results/mlsys-separated-candidate-harness-v1/run-3/).

This is **not** a model update, agent rollout, new task evaluation, or held-out
capability result. Consumed task graders and oracle fixtures are used only to
check the isolated adapter boundary. Dataset-standardization, gradient
accumulation, global-gradient clipping, and range-sums intentionally remain
unsupported and fail closed: the existing isolated call protocol cannot
report whether candidate code mutated its arguments, while their frozen
checkers depend on that contract. The shared worker remains unchanged to
preserve source inventories pinned by continuation evidence. V3 copies of
shared v1 task IDs are routed using their canonical descriptor path.

## Verification

Reproduce the calibration on Apple Silicon with Docker available:

```bash
DOCKER_CONTEXT=colima PYTHONPATH=src python \
  scripts/verify_mlsys_separated_candidate_harness_v1.py \
  --output results/mlsys-separated-candidate-harness-v1/run-3
PYTHONPATH=src python scripts/audit_mlsys_separated_candidate_harness_v1.py \
  results/mlsys-separated-candidate-harness-v1/run-3
```

The output directory must not already exist. The verifier runs the negative
and oracle controls in Docker and records its runtime version and elapsed time.
The auditor checks every artifact hash, relevant source hash, claim label, and
source commit identity.

The regression suite exercises the isolated adapter for v1 tasks with supported
call semantics and v3 routing/oracle controls; it explicitly checks that all
four mutation-sensitive tasks fail closed. No claim is made about an LLM, policy
optimization, held-out task success, or training cost.
