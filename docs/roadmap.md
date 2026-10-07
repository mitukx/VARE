# Roadmap

VARE should expand its claims only when the next evidence tier can be reproduced within the declared resource budget. The current scope includes independently graded historical source changes and measured local CPU evaluation execution.

## E0 — Harness integrity

Both task graders reject isolated mutations of the task descriptor, task brief, and evaluator against the unchanged protocol lock. A retained calibration has two untampered controls and six tamper cases; all controls reach the candidate-file validation point and all six mutations are rejected. See [`protocol-integrity/cpu-calibration-v1`](../results/protocol-integrity/cpu-calibration-v1/summary.json).

The trust anchor is the version-controlled protocol lock. This calibration does not authenticate a coordinated edit to the lock and does not sandbox hostile candidate code; those limits are explicit in the [evidence notes](evidence.md).

## E1 — Historical task calibration

Two pinned tasks now have calibrated pre-fix and fixed revisions:

1. [HF behavior-policy parity](../benchmarks/historical/rvl_behavior_policy_parity/TASK.md) checks rollout and learner probability parity with deterministic CPU fixtures.
2. [TRL accumulation-window normalization](../benchmarks/historical/trl_grpo_accumulation_scale/TASK.md) checks extracted production loss-normalization branches with deterministic scalar fixtures.

Both graders reject the pre-fix revision and accept the known fixed revision. Their raw results, protocol snapshots, and hashes are retained in [`results/`](../results/) and summarized in [`evidence.md`](evidence.md). These outcomes calibrate the tasks and graders; they do not show that an agent can discover the fix or that a model improves.

## E2 — Agent trajectories

Run a genuinely local, freely available coding agent repeatedly on the same immutable task set with a fixed wall-time and tool budget. Record the agent/configuration, task revision, repetition, wall time, patch hash, grader output, failure category, and transcript hash when retention is permitted. Do not compare runs after changing the task or grader without versioning both. No E2 result is currently retained.

## E3 — Curriculum intervention

After comparable E2 trajectories exist, cluster observed failures and compare a transparent failure-driven task selector against a fixed selector at equal rollout and wall-time budgets. Freeze the task split, primary metric, and acceptance rule before measuring the comparison.

## E4 — Small-model learning

Consider training only if E1–E3 identify a causal question that cheaper CPU experiments cannot answer and free compute is available. Preregister the held-out metric, seeds, budget, stopping rule, and promotion threshold. Retain null and negative runs.

## E5 — Systems impact

The original evaluation runner now has a narrow local scheduling measurement: five paired one-worker/four-worker campaigns, 80 historical-task grades with consistent outcomes, and a median paired speedup of 3.4857×. Sixteen synthetic reliability cases match their declared classifications. See the [report](scheduler-report.md), [implementation contract](execution.md) and [raw evidence](../results/cpu-scheduler-v1/summary.json).

This completes a local systems measurement, not distributed execution or inference performance. The next systems questions are interrupted-run recovery, larger/heterogeneous CPU workloads and resource isolation. An upstream contribution would be a separate result and has not been made here.

## E6 — Repeated improvement

Study repeated system-proposed interventions only after independent downstream effects can be measured across tasks and verifier drift and compute cost are accounted for.

## Current stop point

The isolated-input E0 calibration, execution failure-injection checks and two E1 task/grader pairs are complete. A narrow E5 local scheduling result is retained. E2–E4, distributed systems evidence and E6 remain outstanding. No agent trajectory, model learning, generalization, or capability gain is claimed.
