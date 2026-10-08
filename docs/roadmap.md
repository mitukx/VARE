# Roadmap

VARE should expand its claims only when the next evidence tier can be reproduced within the declared resource budget. The current scope includes independently graded historical source changes and measured local CPU evaluation execution.

## E0 — Harness integrity

All three current task graders reject isolated mutations of the task descriptor, task brief, and evaluator against the unchanged protocol lock. The latest calibration has three untampered controls and nine tamper cases; all controls reach candidate-file validation and all nine mutations are rejected. See [`protocol-integrity/cpu-calibration-v5`](../results/protocol-integrity/cpu-calibration-v5/summary.json).

The trust anchor is the version-controlled protocol lock. This calibration does not authenticate a coordinated edit to the lock and does not sandbox hostile candidate code; those limits are explicit in the [evidence notes](evidence.md).

A separate [promotion-gate regression](promotion-gate-report.md) found that seven non-finite report inputs were accepted before a fail-closed validation change. The frozen CPU grader rejects those inputs and preserves a finite positive control. This is one implementation contract, not an estimate of gate error rates.

A separate [replay group regression](replay-group-freshness-report.md) found that freshness filtering returned a partial group when one member was stale. The fix drops that group while preserving independent fresh groups and per-item mode. This is one replay invariant test, not a learning or performance result.

The RVL GRPO adapter now restores its saved incumbent after partial candidate-training failures that mutate model and optimizer state. The fake-trainer regression checks exception propagation and the next incumbent rollout. Two CPU smokes with one pinned real trainer revision tested distinct locations: after a completed update at the candidate boundary, and inside the real `train_step` after the real optimizer mutation but before return. Both reconstructed model/optimizer/RNG state and checked the next rollout. These remain narrow adapter-level correctness evidence; optimizer-kernel faults and uncatchable process loss are untested. See the [candidate-boundary report](rvl-grpo-partial-failure-report.md), [in-step fault report](rvl-grpo-midstep-fault-report.md), and respective run bundles.

## E1 — Historical task calibration

Three pinned tasks now have calibrated pre-fix and fixed revisions:

1. [HF behavior-policy parity](../benchmarks/historical/rvl_behavior_policy_parity/TASK.md) checks rollout and learner probability parity with deterministic CPU fixtures. Protocol v3 added inherited `typical_p`; v4 adds `suppress_tokens` and `no_repeat_ngram_size`, applies suppression in the fixture, and rejects a candidate that v3 accepted.
2. [Inherited bad-word neutrality](../benchmarks/historical/rvl_bad_words_neutrality/TASK.md) adds a separately locked single-token `bad_words_ids` CPU fixture without changing the v4 task. The mutation audit found one constructed candidate accepted by v4 and rejected by the new grader, while the fixed source passes both. This is one uncovered setting, not an error-rate estimate.
3. [TRL accumulation-window normalization](../benchmarks/historical/trl_grpo_accumulation_scale/TASK.md) checks extracted production loss-normalization branches with deterministic scalar fixtures, masked numerator and reachability guards, normalizer/loss write guards through return, direct aliases, and denominator matching. Protocol v7 accepted two mutations that v8 now rejects: an early return and a zeroed numerator. The pinned fixed source still passes. These cases do not estimate a general error rate.

All three graders reject the pre-fix revision and accept the known fixed revision. Their raw results, protocol snapshots, and hashes are retained in [`results/`](../results/) and summarized in [`evidence.md`](evidence.md). These outcomes calibrate the tasks and graders; they do not show that an agent can discover the fix or that a model improves.

## E2 — Controlled post-training mechanism

The v1 diagnostic and its failed KL rule remain in [`synthetic-dpo-report.md`](synthetic-dpo-report.md). The separate v2 confirmation is reported in [`synthetic-dpo-v2-report.md`](synthetic-dpo-v2-report.md): a training-only development phase selected the update count; the frozen 10-seed confirmation passed its held-out NLL/KL rule, and the audit reconstructs all seed-by-arm metrics. This reaches a narrow synthetic mechanism result. It is not LLM training or capability evidence.

## Supporting evidence — Agent trajectories

The corrected [local CPU pilot](local-agent-pilot.md) ran one small model on one immutable task for three formal seeds. All attempts read source but produced no accepted edit; the grader rejected all three unchanged candidates. The original v1 cohort is invalidated due to a tool-history serialization bug. This is a retained negative tool-loop result, not successful task-solving evidence. Next, test a materially improved and frozen interaction protocol on a broader immutable task pack, while keeping task and model changes versioned. Do not compare it directly to this pilot unless the task, model and tool changes are versioned.

## E3 — Preference-signal robustness

The v2 noise arm remains diagnostic and its shuffled-ID arm is invalid. A separate [`v1 noise/shift study`](synthetic-preference-robustness-v1-report.md) was committed before its run. A second implementation now re-trains from retained designs/labels and verifies the v1 arm metrics and gate without sharing runner/auditor code; however, it does not regenerate the Gaussian contexts/action-pair designs or establish cross-version equivalence. Across ten seeds, the clean arm passed its synthetic base-teacher NLL/KL rule; 20% and 40% orientation flips worsened mean NLL on every seed, and the base-trained policy had higher mean NLL under a declared shifted teacher. This closes one narrow synthetic E3 comparison. It does not cover realistic annotator disagreement, learned reward models, adaptive reward hacking, or language-model outputs. Further robustness claims require a different task/generator and independent review; do not tune this protocol.

## E4 — Small-model learning

**SNLI entailment base feasibility v1 is a non-pass.** The no-update, 512-row balanced screen reached 49.41% balanced accuracy against a frozen 58% minimum; its stratified 95% interval was [45.31%, 53.71%]. The same-host independent score replay passed, but the result only establishes that this task/prompt/model setup did not clear the feasibility gate. Retire the cohort and task formulation; do not tune or train on it. See the [report](cpu-snli-entailment-base-feasibility-v1-report.md) and [bundle](../results/snli-entailment-base-feasibility-v1/run-1/).

The separate [generated-arithmetic feasibility pilot](cpu-generated-arithmetic-feasibility-v1-report.md) scored 1/64 exact answers and failed its frozen minimum base-rollout gate before training. Retire that task setup rather than lowering its gate. It does not change the narrower GSM8K forced-choice result or close the free-form task-improvement gap.

The generated [code-repair feasibility screen](cpu-code-repair-feasibility-v1-report.md) also failed before training: 0/32 episode successes, no accepted edits or visible tests, and 60 unsafe/unauthorized attempts. Its CPU and RSS limits passed; an independent audit reconstructed the trajectory and grader results. Retire this model/task/tool pairing and exclude its public pilot from future training or confirmation. It does not establish a general coding limitation.

The frozen L2 RVL/Qwen campaign remains unrun: its original workload exceeds the current no-spend CPU budget. Do not edit its lock. The initial cached-model [`cpu_lm_dpo_head_v1`](../protocols/cpu_lm_dpo_head_v1.lock.json) result remains a preregistered non-pass. A separate train-only development phase then selected 20 updates for the locked [`cpu_lm_gsm8k_dpo_confirmation_v1`](../protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json): the Qwen model's conditional preference NLL improved by 0.00697 nats/question across all 1,319 public GSM8K test questions and three fresh seeds; the paired 95% interval was [−0.00800, −0.00591], with all KL values below 0.5. The bundle audit matched the pinned dataset files, rebuilt the pairs and reconstructed every metric. Accuracy remained near chance (0.4943 → 0.4956), so this is a small forced-choice result, not free-form generation or capability evidence. See the [report](cpu-lm-gsm8k-dpo-confirmation-v1-report.md) and [audited bundle](../results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/). Later sequence-level GSM8K studies and matched BoolQ runs are recorded in the [experiment index](experiments.md); none established an independently confirmed task improvement. The [current gaps](current-gaps.md) identify the unresolved evidence and safeguards against reusing consumed rows.

The separate [procedural binary-action DPO screen](cpu-procedural-entailment-dpo-development-v1-report.md) failed its development gate: the frozen base had near-zero Yes recall, scalar calibration beat contextual DPO, and DPO exceeded the KL cap. Its independent bundle audit passed, but this remains synthetic mechanism evidence only. Do not open its reserved confirmation cohorts or present it as task success.

As supporting reward-model evidence, the outcome-informed HH-RLHF v3 fixed-head scalar follow-up passed development and confirmation NLL gates, with independent same-host replay audits. Its positive scalar leaves ranking accuracy unchanged, ECE worsened, and the effect threshold was informed by the v2 post-hoc result. This narrow score-scaling result does not close the unresolved downstream task-success gap.

A separate [HH human-preference DPO development v2](hh-human-preference-dpo-development-v2-report.md) compared sequence-level DPO with matched chosen-only SFT on a frozen training cohort and hash-disjoint development prompts. DPO missed its +0.05 held-out preference-accuracy gain and two-seed consistency gates; the score-replay audit passed, but no confirmation followed. The length-only baseline exceeded both model policies. This is a policy-update non-pass on human-preference labels, not downstream task-success evidence.

## E5 — Systems impact

The original evaluation runner now has a narrow local scheduling measurement: five paired one-worker/four-worker campaigns, 80 historical-task grades with consistent outcomes, and a median paired speedup of 3.4857×. Sixteen synthetic reliability cases match their declared classifications. See the [report](scheduler-report.md), [implementation contract](execution.md) and [raw evidence](../results/cpu-scheduler-v1/summary.json).

The durable coordinator also has a frozen same-host CPU measurement of input freshness work. For serial workloads of 8/16/32 unchanged fixture jobs, targeted commit checks were 8/16/32 versus 64/256/1,024 for the previous full refresh. Full export refresh and offline bundle replay remain intact. The measured completion time at 32 jobs was 1.082s versus 28.479s on this machine; treat timing as descriptive. A v1 pilot with a commit-identity mismatch is retained and marked invalid; the cited result is the source-hash-audited v2 experiment. See the [freshness scaling report](freshness-scaling-report.md).

This completes a local systems measurement, not distributed execution or inference performance. **Update 2026-10-09:** source review found that nonmultiple rollout budgets could create an undersized group for group-relative training. The engine now rounds up to complete groups; see the [integrity report](rollout-group-integrity-v1-report.md). The one-command clean-checkout runner and independent audit for the synthetic preference robustness study are available in [`scripts/reproduce_synthetic_preference_robustness.py`](../scripts/reproduce_synthetic_preference_robustness.py). It supports fresh, version-specific reruns; an outside human review remains outstanding. An upstream contribution would be a separate result and has not been made here.

## E6 — Repeated improvement

Study repeated system-proposed interventions only after independent downstream effects can be measured across tasks and verifier drift and compute cost are accounted for.

## Current stop point

The isolated-input E0 calibration, execution failure-injection checks, promotion-metric and replay-group regressions, bounded environment-command output v4 regression, and three E1 task/grader pairs are complete. The RVL mutation audit covers one constructed `bad_words_ids` case only. Narrow E5 measurements cover local scheduling and durable freshness-check work. E2 has accepted synthetic preference-policy confirmations alongside the preserved v1 diagnostic non-pass. One narrow E3 study measures label flips and a predefined preference shift; realistic preference and reward-model robustness remains untested. E4 includes one small audited GSM8K forced-choice NLL improvement on a cached model, an HH-RLHF v3 score-scaling study whose outcome-informed NLL gate passed confirmation but whose ECE worsened, two HH human-preference DPO development non-passes, and a SNLI base-policy feasibility non-pass before training. None establishes downstream task gain or general reward-model quality. The GSM8K study remains verifier-labeled and updates only two output columns. Free-form task improvement, external reproduction, broader task diversity, heterogeneous workload profiling, distributed systems evidence and E6 remain outstanding. Agent evidence is separately negative. No generalization or capability gain is claimed.

**Current code-agent result:** the frozen screen failed at 0/32, with no accepted edit, visible test, or finish and 60 unsafe/unauthorized attempts; independent trajectory/data/grader audit passed, and CPU/RSS limits passed. Retire this model/task/tool pairing. See the [report](cpu-code-repair-feasibility-v1-report.md) and [bundle](../results/cpu-code-repair-feasibility-v1/run-1/). The current work order is in the [study decision](next-study-decision-2026-10-09.md): fix the incomplete-group path, then lower the cost of outside reproduction. Do not start another model screen until a distinct viable base/task pair can support independent task-success evidence. The next learning comparison still requires matched SFT, one RL method, multiple seeds and genuinely distinct held-out task compositions.

## Restored experimental implementation

The source recovered at upstream commit `f5c92cf` is integrated in `src/vare`, including replay, freshness, curriculum, paired promotion and RVL integration contracts. Their existence does not change the evidence ladder. The imported L0 raw archive failed the retained hash/decompression audit; its old accuracy summary is excluded from current verified claims. Legacy source-pattern fixtures are retained under `benchmarks/seeds/`, separately from calibrated runtime historical tasks.

## Recovery evidence checkpoint

Persistent evaluation now has a frozen CPU recovery experiment: 24/24 fault cases and four calibrated historical decisions after claimant loss. See [recovery-report.md](recovery-report.md). Current terminal publication is fenced, changed inputs are invalidated at use/export and old attempts remain retained. This extends local E0/E5 systems evidence. Multi-host leases/consensus, schema migration, power-loss behavior, stronger resource containment and a measured learner update remain separate open work.
