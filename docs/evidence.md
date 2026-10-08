# Calibration evidence

This page describes what the retained calibration artifacts establish and where the claims stop. Each run checks one pinned pre-fix revision against one already-published fixed revision. The result validates a task/grader pair; it does not show that an agent independently found the fix.

## HF behavior-policy parity

- Task: [brief](../benchmarks/historical/rvl_behavior_policy_parity/TASK.md), [descriptor](../benchmarks/historical/rvl_behavior_policy_parity/task.json), [locked protocol](../benchmarks/historical/rvl_behavior_policy_parity/protocol.lock.json).
- Source: [`Recursive-Verification-Lag` pre-fix revision `e788f11`](https://github.com/mitukx/Recursive-Verification-Lag/tree/e788f113ad6b246a361cd50a52eaf2867f2a66f8) and fixed revision [`c7e646b`](https://github.com/mitukx/Recursive-Verification-Lag/tree/c7e646b043cb56e5ea3c2623bb8a61e065451f72).
- Protocol: `rvl-hf-behavior-policy-parity-protocol-v3`; six distinct prompt/temperature conditions, invalid-temperature and model-state-restoration checks, and a check that pretrained `typical_p` cannot silently alter the scored distribution.
- Observed: pre-fix maximum absolute rollout log-probability error `0.738005434919931` and learner error `0.37112804442989056`; fixed errors `0.0` for both. The pre-fix revision failed and fixed revision passed.
- Artifacts: [v3 summary](../results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v3/summary.json), [pre-fix grader output](../results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v3/baseline.grade.json), [fixed grader output](../results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v3/calibration.grade.json), [hash manifest](../results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v3/manifest.json), [protocol snapshot](../results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v3/protocol_snapshot/). The [v2-to-v3 mutation study](../results/rvl-hf-behavior-policy-parity-v1/generation-config-mutation-v1/summary.json) records the old false acceptance and new rejection.
- Resources: local macOS arm64 / Python 3.9.6 calibration; no weights, third-party Python packages, GPU, paid API, or external compute.
- Limits: fixtures use deterministic CPU model/tokenizer doubles. The learner's `_sample_objective` method is exercised, but this is not a model run or training result. Candidate source runs in a subprocess with the current user's permissions; this is not a hostile-code sandbox.

The v1 and v2 calibrations are retained at [`cpu-calibration-v1`](../results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v1/) and [`cpu-calibration-v2`](../results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v2/). They used earlier protocols; v3 is current. The v2-to-v3 mutation study changes only the candidate's `GenerationConfig` to inherit `typical_p=0.72`: v2 passes with zero failures, while v3 rejects all six rollout cases. The v3 guard checks the configured value recorded by the fixture; it is not a full Hugging Face logits-warper or real-model simulation.

## TRL accumulation-window normalizer

- Task: [brief](../benchmarks/historical/trl_grpo_accumulation_scale/TASK.md), [descriptor](../benchmarks/historical/trl_grpo_accumulation_scale/task.json), [locked protocol](../benchmarks/historical/trl_grpo_accumulation_scale/protocol.lock.json).
- Source: [`trl` pre-fix revision `8697378`](https://github.com/huggingface/trl/tree/8697378709102608e3c9dc5bf582772e4ddee788) and fixed revision [`e1b2e21`](https://github.com/huggingface/trl/tree/e1b2e21975994e676d9a676e3ab005e9bee38e40). The public upstream issue is [#5619](https://github.com/huggingface/trl/issues/5619); the fix is [PR #6024](https://github.com/huggingface/trl/pull/6024).
- Current protocol: `trl-grpo-accumulation-window-normalizer-protocol-v7`; six arithmetic cases in each of two production branches (main DAPO/CISPO/VESPO and experimental DAPO), including a partial final window and unchanged evaluation behavior. The grader rejects unmodeled later writes to `normalizer`, requires direct division by that checked value, and checks downstream loss writes through return, permits only the pinned entropy and auxiliary-loss adjustments, and rejects in-place Tensor mutator calls or `out=` writes targeting checked locals and direct local-name aliases.
- Observed: 12 cases total. Pre-fix maximum absolute normalizer error `12.0`; fixed maximum error `0.0`. The pre-fix revision failed and fixed revision passed.
- Adversarial check: on the pinned fixed source, inserting `normalizer = normalizer * 2` after the otherwise-correct accumulation correction changed the production loss scale. The v1 grader accepted this candidate; v2 rejects it. Raw outputs and the patch are retained in [mutation evidence](../results/trl-grpo-accumulation-window-normalizer-v1/grader-mutation-v1/summary.json); the local regression is in [`tests/test_graders.py`](../tests/test_graders.py).
- Artifacts: [v7 summary](../results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v7/summary.json), [pre-fix grader output](../results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v7/baseline.grade.json), [fixed grader output](../results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v7/calibration.grade.json), [hash manifest](../results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v7/manifest.json), [protocol snapshot](../results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v7/protocol_snapshot/). Earlier protocol snapshots/results remain as history.
- The [loss-denominator mutation study](../results/trl-grpo-accumulation-window-normalizer-v1/loss-denominator-mutation-v1/summary.json) changes both pinned fixed-source DAPO branches to divide by `normalizer * 2`. Protocol v2 passes all 12 conditions with zero normalizer error; protocol v3 rejects the changed denominator in all 12 conditions while still measuring zero normalizer error.
- A separate [loss-dataflow mutation audit](../results/trl-loss-dataflow-audit-v1/summary-v3.json) confirmed that v3 accepted `loss = loss * 2` immediately after the checked division in both pinned branches. The candidate changes the policy-loss scale; v3 returned pass with zero failures. Protocol v4 rejects the same candidate in all 12 arithmetic conditions while accepting the pinned fixed source. The v4 guard is branch-local and source-structural; it does not execute the full loss expression or gradient update and may reject valid refactors.
- Resources: local macOS arm64 / Python 3.9.6 calibration; no weights, third-party Python packages, GPU, paid API, or external compute.
- Limits: the grader extracts and executes production normalization statements against scalar doubles and checks branch-local writes plus a direct AST match on the selected loss denominator. The v7 guard checks known downstream loss writes through return and direct local-name aliases, but it does not prove arbitrary semantic dataflow. The grader does not execute the full loss expression, import the trainer, perform a gradient update, measure model behavior, or prove semantic correctness for arbitrary refactors.

## Local CPU agent trajectory pilot

- Frozen setup: one pinned RVL historical task, pre-fix revision `e788f11`, locked v3 evaluator, and a local [Qwen2.5-0.5B-Instruct snapshot](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) on macOS arm64. The model page identifies Apache-2.0 licensing; exact snapshot files and hashes are in the [v2 protocol](../protocols/local_agent_trajectory_rvl_pilot_v2.json). Four CPU threads; CUDA unavailable and MPS unused.
- Formal seeds 54–56 ran in fresh workspaces with a 10-generation, 2,560-token and 300-second soft wall budget. All three diffs were empty and the locked evaluator rejected all three unchanged workspaces. Two runs read source and emitted invalid/placeholder patches without unified-diff headers; one read source twice and stopped without editing. No model files were downloaded during the runs; no GPU, paid API or external compute was used.
- Artifacts: [v2 summary](../results/local-agent-rvl-pilot-v2/summary.json), [v2 manifest](../results/local-agent-rvl-pilot-v2/manifest.json), [frozen v2 protocol](../protocols/local_agent_trajectory_rvl_pilot_v2.json), and bounded [runner](../scripts/run_local_agent.py). Individual raw results retain model outputs, tool calls/results, source hashes, diffs and grader outputs. The original [v1 summary](../results/local-agent-rvl-pilot-v1/summary.json) is invalidated and excluded because the runner serialized assistant tool calls incorrectly.
- Limits: this is a three-run negative tool-loop observation for one small local model, one prompt/tool contract and one public historical task. It does not estimate agent capability, successful task solving, novel fix discovery or generalization. Model loading used HF offline flags and the tool set had no network tool, but OS-level network egress was not disabled. Exploratory seeds 41–43 are private and excluded; public exploratory seeds 51–53 are disclosed in the v2 protocol and excluded from its formal cohort.

## Promotion-gate non-finite metric regression

- Frozen task: eight deterministic local CPU cases cover one finite positive control and seven invalid evidence inputs: non-finite candidate/incumbent primary score, slice score, cost, verifier disagreement, and paired task score. The task, grader, and expected baseline/fixed decisions are hash-locked before the baseline result.
- Baseline revision `d4627bf` accepted all seven invalid reports, including a candidate with `primary=NaN`; the finite control was accepted. Fixed source rejects all seven with `invalid_evaluation_metrics`, returns finite decision fields, and still accepts the control.
- Artifacts: [task](../benchmarks/regressions/promotion_gate_metrics/), [baseline record](../results/promotion-gate-metrics-v1/baseline.json), [fixed result](../results/promotion-gate-metrics-v1/summary.json), and [report](promotion-gate-report.md).
- Limits: this is a narrowly scoped input-validation regression on the current implementation. It shows the selected fail-open paths are closed; it does not estimate broad promotion-gate error rates, establish the correctness of underlying evaluation metrics, or measure model learning.

## Replay group freshness atomicity

- Frozen task: one four-member stale comparison group, one separate four-member fresh group, and a per-item control. If any member of a grouped cohort is inadmissible, the whole cohort must be excluded; independent fresh groups remain whole.
- Baseline revision `cd3129c` returned three members of the stale group plus all four fresh members (7 total). The fixed sampler excludes the stale group, returns the four fresh members whole, and keeps the ungrouped per-item mode at seven eligible experiences.
- Artifacts: [task and protocol](../benchmarks/regressions/replay_group_freshness/), [baseline/fixed results](../results/replay-group-freshness-v1/), and [report](replay-group-freshness-report.md).
- Limits: this is a deterministic replay-contract regression. It does not measure learning outcomes, replay quality, throughput, or general performance under production workloads.

## Reproduction and integrity limits

Reproduce the task calibrations and locked-input integrity trial using the commands in the [README](../README.md). The scripts verify task descriptor, brief, and evaluator hashes against the protocol lock and include snapshots and SHA-256 manifests with each output.

The fixed revisions are public historical changes used to validate the benchmark plumbing. They are not changes authored by this project. The measured outcome is limited to the declared regression contracts on those source revisions.

## Locked-input integrity calibration

- Protocol: `locked-task-input-tamper-calibration-v1`, run over both historical task graders.
- Controls: one untampered protocol per task. Both pass their locked-input checks and reach the expected missing-candidate-file check.
- Mutations: for each task, change the task descriptor, task brief, or grader in isolation while keeping the version-controlled protocol lock unchanged. All 6 of 6 cases are rejected with the expected hash-mismatch error.
- Artifacts: [summary](../results/protocol-integrity/cpu-calibration-v1/summary.json), [control outputs](../results/protocol-integrity/cpu-calibration-v1/controls.json), [tamper outputs](../results/protocol-integrity/cpu-calibration-v1/tamper_cases.json), [hash manifest](../results/protocol-integrity/cpu-calibration-v1/manifest.json), [protocol and script snapshot](../results/protocol-integrity/cpu-calibration-v1/protocol_snapshot/).
- The current task/evaluator set was rechecked separately: [v4 integrity summary](../results/protocol-integrity/cpu-calibration-v4/summary.json), [controls](../results/protocol-integrity/cpu-calibration-v4/controls.json) and [tamper cases](../results/protocol-integrity/cpu-calibration-v4/tamper_cases.json) retain two controls and six isolated mutations.
- Resources: local macOS arm64 / Python 3.9.6; no GitHub source fetch, model weights, third-party Python package, GPU, paid API, or external compute.
- Trust boundary: the unchanged Git-versioned protocol lock is the trust anchor. This experiment does not authenticate a coordinated edit to the lock and does not provide OS isolation for candidate code. It establishes only that isolated changes to the descriptor, brief, or grader are detected.

## Local CPU evaluation orchestration

The [scheduler report](scheduler-report.md) covers VARE's original evaluation execution code. The frozen protocol produced 16/16 expected synthetic classifications and 80/80 expected historical candidate decisions. Median paired one-worker/four-worker speedup was 3.4857× over five pairs. Raw campaign records, protocol/runner snapshots and manifests are retained at [`cpu-scheduler-v1`](../results/cpu-scheduler-v1/). `scripts/audit_scheduler.py` reconstructs the measured summary offline.

This provides local scheduling and reliability evidence; it does not change the claim limits of either task or imply agent/model improvement. The runner is not a hostile-code sandbox. The report discloses the first-pair timing outlier and the later CLI audit correction separately from the code used for measurement.

During grader v2 development, an initial AST walk accidentally included later `elif` branches and rejected the known fixed source. That failed calibration is retained at [`grader-v2-overrejection-failure`](../results/trl-grpo-accumulation-window-normalizer-v1/grader-v2-overrejection-failure/). Scoping the integrity check to the selected branch body fixed the over-rejection; the calibrated v2 output above is the final result.

## Restored control-plane source and imported artifacts

The source restored on `origin/main` at `f5c92cf` is integrated into `src/vare`: grouped replay, lag/freshness controls, transactional candidate hooks, paired promotion gates, environment campaigns and RVL adapters. These are experimental implementations with regression tests. Actual GPU training, model improvement and distributed-fleet performance remain unmeasured.

The imported artifact manifest has seven matching file hashes and one failure: `evidence/l0_contextual_bandit_seed7.json.gz` is truncated/corrupt, does not match its declared digest or size, and cannot be decompressed. The old L0 accuracy summary is therefore **not independently verified from retained raw data** and is not used as a headline result. The original artifact and manifest are retained; [the import audit](../results/restored-evidence-audit-v1/summary.json) records the failure. Re-running a toy experiment would create new evidence, not repair the old measurement.

The earlier source-pattern-only task seed is retained separately at [`benchmarks/seeds/rvl_static_parity`](../benchmarks/seeds/rvl_static_parity/) for its original environment-harness regression test. The calibrated locked historical tasks use runtime graders and are described above. The two descriptor formats have separate CLIs; the legacy `env-campaign` catalog does not consume the locked historical descriptors.

## Durable local evaluation recovery

The [recovery report](recovery-report.md) retains 24/24 matched fault cases across three replications, 117 synthetic jobs/126 attempts and four historical jobs recovered after one lost claimant with calibrated outcomes preserved. The [frozen protocol](../protocols/cpu_recovery_v1.json) precedes measurement. [Raw exports](../results/cpu-recovery-v1/) contain current states, transactional event chains, protected snapshots and attempt payloads. `scripts/audit_recovery.py` reconstructs acceptance offline. These are local coordination/freshness results; execution can repeat, and no multi-host, power-loss, hostile-process or model-learning claim is made.
