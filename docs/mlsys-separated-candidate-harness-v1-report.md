# MLSys separated candidate harness

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
retained in [`results/mlsys-separated-candidate-harness-v1/run-10/`](../results/mlsys-separated-candidate-harness-v1/run-10/).

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
grader. Routed correctness cases now share one isolated candidate process, so
module-level state persists across the ordered hidden cases. A counter-state
regression passes the expected `1, 2` sequence and rejects `1, 1`. Explicit
import-start and ready markers give import and each function call their own
timeout window; a 0.25-second import plus a 0.25-second function under a
0.4-second per-phase limit passes, while a single combined deadline would
reject it. Frozen ValueError-prefix behavior is preserved, and causal-attention
uses the frozen checker's numeric bool behavior. v3 task limits are retained:
the EMA task receives 20 seconds. The v2 protocol matches routed tasks' 2 GiB
memory limit, one CPU, 64 PID limit, and digest-pinned images; v1's 512 MiB
protocol and run-7/run-8 evidence remain frozen history. The shared task runner remains unchanged,
preserving source inventories pinned by continuation evidence. The v3 warmup
copy is resolved through its verified v1 adapter path; label smoothing and
row-log-softmax remain non-routable because the JSON call format cannot retain
their return-type/non-finite input semantics.

## Verification

Reproduce the calibration on Apple Silicon with Docker available:

```bash
DOCKER_CONTEXT=colima PYTHONPATH=src python \
  scripts/verify_mlsys_separated_candidate_harness_v2.py \
  --output /tmp/vare-separated-harness-replay/run-1
PYTHONPATH=src python scripts/audit_mlsys_separated_candidate_harness_v2.py \
  /tmp/vare-separated-harness-replay/run-1
```

The output directory must not already exist. Choose a fresh path for repeated
runs; the committed run-10 bundle remains immutable. The verifier runs the
negative and oracle controls in Docker and records its runtime version and elapsed time.
The auditor checks every artifact hash, relevant source hash, claim label, and
source commit identifier. Source hashes bind the inspected files directly,
including when the PR's commit history is later squashed.

The regression suite exercises the isolated adapter for the three tasks whose
frozen correctness semantics fit the existing JSON-call protocol: causal
attention, warmup cosine, and v3 EMA bias. It checks that mutation-sensitive
and type/non-JSON-sensitive tasks fail closed, and that a v3 warmup copy routes
from its canonical path. No claim is made about an LLM, policy
optimization, held-out task success, or training cost.
