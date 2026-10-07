# Roadmap

VARE should expand its claims only when the next evidence tier can be reproduced within the declared resource budget. The current scope includes independently graded historical source changes and measured local CPU evaluation execution.

## E0 — Harness integrity

Both current task graders reject isolated mutations of the task descriptor, task brief, and evaluator against the unchanged protocol lock. The latest calibration has two untampered controls and six tamper cases; all controls reach candidate-file validation and all six mutations are rejected. See [`protocol-integrity/cpu-calibration-v4`](../results/protocol-integrity/cpu-calibration-v4/summary.json).

The trust anchor is the version-controlled protocol lock. This calibration does not authenticate a coordinated edit to the lock and does not sandbox hostile candidate code; those limits are explicit in the [evidence notes](evidence.md).

A separate [promotion-gate regression](promotion-gate-report.md) found that seven non-finite report inputs were accepted before a fail-closed validation change. The frozen CPU grader rejects those inputs and preserves a finite positive control. This is one implementation contract, not an estimate of gate error rates.

A separate [replay group regression](replay-group-freshness-report.md) found that freshness filtering returned a partial group when one member was stale. The fix drops that group while preserving independent fresh groups and per-item mode. This is one replay invariant test, not a learning or performance result.

## E1 — Historical task calibration

Two pinned tasks now have calibrated pre-fix and fixed revisions:

1. [HF behavior-policy parity](../benchmarks/historical/rvl_behavior_policy_parity/TASK.md) checks rollout and learner probability parity with deterministic CPU fixtures. Protocol v3 adds a retained false-acceptance regression for inherited non-neutral `typical_p`; v2 accepted that candidate, while v3 rejects all six rollout cases.
2. [TRL accumulation-window normalization](../benchmarks/historical/trl_grpo_accumulation_scale/TASK.md) checks extracted production loss-normalization branches with deterministic scalar fixtures, normalizer and post-division loss-write guards, and direct denominator matching. A frozen audit confirmed a v3 false acceptance when `loss` was doubled after its checked division; protocol v4 rejects that mutation in both branches while retaining fixed-source acceptance. The audit tests one mutation, not a general error rate.

Both graders reject the pre-fix revision and accept the known fixed revision. Their raw results, protocol snapshots, and hashes are retained in [`results/`](../results/) and summarized in [`evidence.md`](evidence.md). These outcomes calibrate the tasks and graders; they do not show that an agent can discover the fix or that a model improves.

## E2 — Agent trajectories

The corrected [local CPU pilot](local-agent-pilot.md) ran one small model on one immutable task for three formal seeds. All attempts read source but produced no accepted edit; the grader rejected all three unchanged candidates. The original v1 cohort is invalidated due to a tool-history serialization bug. This is a retained negative tool-loop result, not successful task-solving evidence. Next, test a materially improved and frozen interaction protocol on a broader immutable task pack, while keeping task and model changes versioned. Do not compare it directly to this pilot unless the task, model and tool changes are versioned.

## E3 — Curriculum intervention

After comparable E2 trajectories exist, cluster observed failures and compare a transparent failure-driven task selector against a fixed selector at equal rollout and wall-time budgets. Freeze the task split, primary metric, and acceptance rule before measuring the comparison.

## E4 — Small-model learning

Consider training only if E1–E3 identify a causal question that cheaper CPU experiments cannot answer and free compute is available. Preregister the held-out metric, seeds, budget, stopping rule, and promotion threshold. Retain null and negative runs.

## E5 — Systems impact

The original evaluation runner now has a narrow local scheduling measurement: five paired one-worker/four-worker campaigns, 80 historical-task grades with consistent outcomes, and a median paired speedup of 3.4857×. Sixteen synthetic reliability cases match their declared classifications. See the [report](scheduler-report.md), [implementation contract](execution.md) and [raw evidence](../results/cpu-scheduler-v1/summary.json).

The durable coordinator also has a frozen same-host CPU measurement of input freshness work. For serial workloads of 8/16/32 unchanged fixture jobs, targeted commit checks were 8/16/32 versus 64/256/1,024 for the previous full refresh. Full export refresh and offline bundle replay remain intact. The measured completion time at 32 jobs was 1.082s versus 28.479s on this machine; treat timing as descriptive. A v1 pilot with a commit-identity mismatch is retained and marked invalid; the cited result is the source-hash-audited v2 experiment. See the [freshness scaling report](freshness-scaling-report.md).

This completes a local systems measurement, not distributed execution or inference performance. The next systems questions are interrupted-run recovery, larger/heterogeneous CPU workloads and resource isolation. An upstream contribution would be a separate result and has not been made here.

## E6 — Repeated improvement

Study repeated system-proposed interventions only after independent downstream effects can be measured across tasks and verifier drift and compute cost are accounted for.

## Current stop point

The isolated-input E0 calibration, execution failure-injection checks, promotion-metric and replay-group regressions, and two E1 task/grader pairs are complete. Narrow E5 measurements cover local scheduling and durable freshness-check work. E2 has one corrected negative local pilot but no successful task trajectory. E3–E4, heterogeneous workload profiling, distributed systems evidence and E6 remain outstanding. No model learning, generalization, or capability gain is claimed.

## Restored experimental implementation

The source recovered at upstream commit `f5c92cf` is integrated in `src/vare`, including replay, freshness, curriculum, paired promotion and RVL integration contracts. Their existence does not change the evidence ladder. The imported L0 raw archive failed the retained hash/decompression audit; its old accuracy summary is excluded from current verified claims. Legacy source-pattern fixtures are retained under `benchmarks/seeds/`, separately from calibrated runtime historical tasks.

## Recovery evidence checkpoint

Persistent evaluation now has a frozen CPU recovery experiment: 24/24 fault cases and four calibrated historical decisions after claimant loss. See [recovery-report.md](recovery-report.md). Current terminal publication is fenced, changed inputs are invalidated at use/export and old attempts remain retained. This extends local E0/E5 systems evidence. Multi-host leases/consensus, schema migration, power-loss behavior, stronger resource containment and real comparable agent trajectories remain separate open work.
