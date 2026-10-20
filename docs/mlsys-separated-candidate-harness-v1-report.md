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

The consumed causal-attention task is used only as an adapter calibration. The
frozen checker rejects a malformed empty result, the isolated grader rejects
the same candidate, and the frozen oracle control passes. This verifies the
adapter path without claiming a checker defect or new task result. The final
raw outputs, source hashes, runtime metadata, and inventory manifest are
retained in [`results/mlsys-separated-candidate-harness-v1/run-6/`](../results/mlsys-separated-candidate-harness-v1/run-6/).

This is **not** a model update, agent rollout, new task evaluation, or held-out
capability result. Consumed task graders and oracle fixtures are used only to
check the isolated adapter boundary. Dataset-standardization, aging scheduler,
gradient accumulation, global-gradient clipping, paired shuffle, range-sums,
row-log-softmax, and label-smoothing remain unsupported for isolated grading
and fail closed. The existing JSON-call protocol cannot faithfully express
all required evaluator contracts: some checkers inspect input mutation, some
need non-finite values or Python return types that JSON loses, and performance
checkers require additional parent-side behavior. Direct frozen fixture
materialization for several unsupported tasks remains available to existing
calibration tests; it does not make those tasks routable through the isolated
grader. The shared worker remains unchanged to preserve source inventories
pinned by continuation evidence. V3 copies of shared v1 task IDs are routed
using their canonical descriptor path.

## Verification

Reproduce the calibration on Apple Silicon with Docker available:

```bash
DOCKER_CONTEXT=colima PYTHONPATH=src python \
  scripts/verify_mlsys_separated_candidate_harness_v1.py \
  --output /tmp/vare-separated-harness-replay/run-1
PYTHONPATH=src python scripts/audit_mlsys_separated_candidate_harness_v1.py \
  /tmp/vare-separated-harness-replay/run-1
```

The output directory must not already exist. Choose a fresh path for repeated
runs; the committed run-6 bundle remains immutable. The verifier runs the
negative and oracle controls in Docker and records its runtime version and elapsed time.
The auditor checks every artifact hash, relevant source hash, claim label, and
source commit identity.

The regression suite exercises the isolated adapter for the three tasks whose
frozen correctness semantics fit the existing JSON-call protocol: causal
attention, warmup cosine, and v3 EMA bias. It checks that mutation-sensitive
and type/non-JSON-sensitive tasks fail closed, and that a v3 warmup copy routes
from its canonical path. No claim is made about an LLM, policy
optimization, held-out task success, or training cost.
