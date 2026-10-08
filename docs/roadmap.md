# Roadmap

VARE should expand its claims only when the next evidence tier can be reproduced within the declared resource budget. The current scope includes independently graded historical source changes and measured local CPU evaluation execution.

## E0 — Harness integrity

Both current task graders reject isolated mutations of the task descriptor, task brief, and evaluator against the unchanged protocol lock. The latest calibration has two untampered controls and six tamper cases; all controls reach candidate-file validation and all six mutations are rejected. See [`protocol-integrity/cpu-calibration-v4`](../results/protocol-integrity/cpu-calibration-v4/summary.json).

The trust anchor is the version-controlled protocol lock. This calibration does not authenticate a coordinated edit to the lock and does not sandbox hostile candidate code; those limits are explicit in the [evidence notes](evidence.md).

A separate [promotion-gate regression](promotion-gate-report.md) found that seven non-finite report inputs were accepted before a fail-closed validation change. The frozen CPU grader rejects those inputs and preserves a finite positive control. This is one implementation contract, not an estimate of gate error rates.

A separate [replay group regression](replay-group-freshness-report.md) found that freshness filtering returned a partial group when one member was stale. The fix drops that group while preserving independent fresh groups and per-item mode. This is one replay invariant test, not a learning or performance result.

## E1 — Historical task calibration

Two pinned tasks now have calibrated pre-fix and fixed revisions:

1. [HF behavior-policy parity](../benchmarks/historical/rvl_behavior_policy_parity/TASK.md) checks rollout and learner probability parity with deterministic CPU fixtures. Protocol v3 added inherited `typical_p`; v4 adds `suppress_tokens` and `no_repeat_ngram_size`, applies suppression in the fixture, and rejects a candidate that v3 accepted.
2. [TRL accumulation-window normalization](../benchmarks/historical/trl_grpo_accumulation_scale/TASK.md) checks extracted production loss-normalization branches with deterministic scalar fixtures, masked numerator and reachability guards, normalizer/loss write guards through return, direct aliases, and denominator matching. Protocol v7 accepted two mutations that v8 now rejects: an early return and a zeroed numerator. The pinned fixed source still passes. These cases do not estimate a general error rate.

Both graders reject the pre-fix revision and accept the known fixed revision. Their raw results, protocol snapshots, and hashes are retained in [`results/`](../results/) and summarized in [`evidence.md`](evidence.md). These outcomes calibrate the tasks and graders; they do not show that an agent can discover the fix or that a model improves.

## E2 — Controlled post-training mechanism

The v1 diagnostic and its failed KL rule remain in [`synthetic-dpo-report.md`](synthetic-dpo-report.md). The separate v2 confirmation is reported in [`synthetic-dpo-v2-report.md`](synthetic-dpo-v2-report.md): a training-only development phase selected the update count; the frozen 10-seed confirmation passed its held-out NLL/KL rule, and the audit reconstructs all seed-by-arm metrics. This reaches a narrow synthetic mechanism result. It is not LLM training or capability evidence.

## Supporting evidence — Agent trajectories

The corrected [local CPU pilot](local-agent-pilot.md) ran one small model on one immutable task for three formal seeds. All attempts read source but produced no accepted edit; the grader rejected all three unchanged candidates. The original v1 cohort is invalidated due to a tool-history serialization bug. This is a retained negative tool-loop result, not successful task-solving evidence. Next, test a materially improved and frozen interaction protocol on a broader immutable task pack, while keeping task and model changes versioned. Do not compare it directly to this pilot unless the task, model and tool changes are versioned.

## E3 — Preference-signal robustness

The v2 result includes noisy-label and shuffled-label diagnostics, but they are not a confirmatory robustness estimate. Compare clean, noisy, and shifted preference conditions at matched update/data budgets under a new frozen protocol. Freeze condition definitions, calibration measures, primary metric, acceptance rule, and analysis before outcomes. If task selection/curriculum is studied, define one explicit intervention with a fixed-selector baseline and independent held-out outcome.

## E4 — Small-model learning

The frozen L2 RVL/Qwen campaign remains unrun: its original workload exceeds the current no-spend CPU budget. Do not edit its lock. A separate [`cpu_lm_dpo_head_v1`](../protocols/cpu_lm_dpo_head_v1.lock.json) study now defines a strict offline CPU run using an already-cached model, a custom narrow output-head adapter, disjoint arithmetic-choice data and held-out preference measures. It has not run formally yet. Retain null and negative outcomes. The synthetic E2 experiment cannot substitute for E4.

## E5 — Systems impact

The original evaluation runner now has a narrow local scheduling measurement: five paired one-worker/four-worker campaigns, 80 historical-task grades with consistent outcomes, and a median paired speedup of 3.4857×. Sixteen synthetic reliability cases match their declared classifications. See the [report](scheduler-report.md), [implementation contract](execution.md) and [raw evidence](../results/cpu-scheduler-v1/summary.json).

The durable coordinator also has a frozen same-host CPU measurement of input freshness work. For serial workloads of 8/16/32 unchanged fixture jobs, targeted commit checks were 8/16/32 versus 64/256/1,024 for the previous full refresh. Full export refresh and offline bundle replay remain intact. The measured completion time at 32 jobs was 1.082s versus 28.479s on this machine; treat timing as descriptive. A v1 pilot with a commit-identity mismatch is retained and marked invalid; the cited result is the source-hash-audited v2 experiment. See the [freshness scaling report](freshness-scaling-report.md).

This completes a local systems measurement, not distributed execution or inference performance. The next systems questions are interrupted-run recovery, larger/heterogeneous CPU workloads and resource isolation. An upstream contribution would be a separate result and has not been made here.

## E6 — Repeated improvement

Study repeated system-proposed interventions only after independent downstream effects can be measured across tasks and verifier drift and compute cost are accounted for.

## Current stop point

The isolated-input E0 calibration, execution failure-injection checks, promotion-metric and replay-group regressions, bounded environment-command output v4 regression, and two E1 task/grader pairs are complete. Narrow E5 measurements cover local scheduling and durable freshness-check work. E2 has one accepted synthetic preference-policy confirmation alongside the preserved v1 diagnostic non-pass. Agent evidence is separately negative. Confirmatory E3 robustness, completed E4 real-model learning, heterogeneous workload profiling, distributed systems evidence and E6 remain outstanding. No language-model learning, generalization, or capability gain is claimed before the frozen model run completes and passes its checks.

## Restored experimental implementation

The source recovered at upstream commit `f5c92cf` is integrated in `src/vare`, including replay, freshness, curriculum, paired promotion and RVL integration contracts. Their existence does not change the evidence ladder. The imported L0 raw archive failed the retained hash/decompression audit; its old accuracy summary is excluded from current verified claims. Legacy source-pattern fixtures are retained under `benchmarks/seeds/`, separately from calibrated runtime historical tasks.

## Recovery evidence checkpoint

Persistent evaluation now has a frozen CPU recovery experiment: 24/24 fault cases and four calibrated historical decisions after claimant loss. See [recovery-report.md](recovery-report.md). Current terminal publication is fenced, changed inputs are invalidated at use/export and old attempts remain retained. This extends local E0/E5 systems evidence. Multi-host leases/consensus, schema migration, power-loss behavior, stronger resource containment and a measured learner update remain separate open work.
