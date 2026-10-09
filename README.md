# VARE

VARE is a CPU-first research project for evaluating post-training signals and policy updates under limited compute. It combines frozen experiment protocols, independently checked outcomes, provenance-aware execution, and retained raw records.

The latest HH human-preference comparison tested length-normalized DPO against standard DPO and chosen-only SFT on a frozen CPU cohort. Length-normalized and standard DPO tied at 0.4974, below the frozen base at 0.5000 and length-only baseline at 0.5645; the decision was a non-pass. The runner hit a summary-aggregation `KeyError` after all nine adapters were saved; the retained adapters were replayed by the separate offline auditor, which passed selection, score, and decision checks. This is a preference-label comparison, not task success. Earlier HH reward-model v3 passed an outcome-informed score-scale NLL gate, but ranking accuracy stayed fixed and ECE worsened. Other real-model DPO/SFT/RLOO studies have not established downstream task gains; the GSM8K improvement is forced-choice NLL with accuracy near chance. Synthetic policy studies test mechanisms, not language-model capability. No general capability improvement or scale result is claimed.

**Latest feasibility screen:** the frozen offline CPU Qwen2.5-0.5B ticket-tool base screen completed 48 prompts within its resource caps, but achieved 0/48 exact successes, 0/48 schema-valid outputs, and 0/120 matched authorized calls. This retires that model/task/prompt/schema pairing; it does not establish general tool-use inability or an RL result. See the [report](docs/cpu-qwen-ticket-tool-feasibility-v1-report.md), [audited bundle](results/cpu-qwen-ticket-tool-feasibility-v1/run-1/), and [decision record](docs/next-study-decision-2026-10-09.md). The earlier [Qwen Math/TIR screen](docs/cpu-qwen-math-tir-feasibility-v1-report.md) and [code-repair feasibility screen](docs/cpu-code-repair-feasibility-v1-report.md) are also retained non-passes.

**Latest engine correctness regression:** `CapabilityLoop` now rejects a rollout `Attempt` whose task identity differs from the dispatched task. The baseline accepted and trained on the substituted task; the frozen CPU regression now passes. This is a narrow harness-correctness result, not policy-improvement evidence. See the [report](docs/engine-rollout-task-binding-v1-report.md) and [raw records](results/engine-rollout-task-binding-v1/run-1/).

**Latest RVL adapter regression:** `RVLGRPOHooks` now rejects generation metadata whose prompt ID or prompt differs from the dispatched task. A frozen CPU test using a mismatched fake backend passed after failing on baseline. It validates the adapter contract; no real-backend mismatch or optimizer/capability effect was measured. See the [report](docs/rvl-grpo-rollout-task-binding-v1-report.md) and [raw records](results/rvl-grpo-rollout-task-binding-v1/run-1/).

## Start here

- [VARE → RVL CPU update-path feasibility](docs/rvl-cpu-real-model-update-path-v1-report.md): retained v1 artifact-retention failure and fresh-cohort v2 frozen protocol; no task-efficacy claim.
- [VARE → RVL CPU update-path feasibility v2](docs/rvl-cpu-real-model-update-path-v2-report.md): v2's one-step CPU update and exact rollback passed, while the tokenizer/reload gate failed before completion; fresh-cohort v3 protocol is frozen.
- [VARE → RVL CPU update-path v3 protocol](protocols/rvl_cpu_real_model_update_path_v3.lock.json): fresh-cohort CPU feasibility rerun after retaining both prior harness failures; no task-efficacy claim.
- [VARE → RVL CPU update-path v3 report](docs/rvl-cpu-real-model-update-path-v3-report.md): one real-model GRPO step, incumbent rollback, tokenizer equality, and checkpoint round-trip recorded; runner exited during post-gate cleanup.
- [Provenance version validation v1](docs/provenance-version-validation-v1-report.md): future and malformed policy/verifier versions now fail closed in lag assessment and RVL replay ingestion; the original ready-item false-freshness reproduction is retained.
- [GRPO group-audit identifiability v1](docs/grpo-group-audit-identifiability-v1-report.md): frozen CPU-only exact counterexample comparing item-level and group-atomic clean-label audits. It studies observability of a synthetic group signal, not model capability.
- [GRPO expected-update audit identifiability v1](docs/grpo-expected-update-audit-identifiability-v1-report.md): exact synthetic witness that one-item audits do not identify a clipped-GRPO local update even when the full action group is visible; the construction is disclosed as exploratory and does not measure model capability.
- [GRPO audit-order characterization v1](docs/grpo-audit-order-characterization-v1-report.md): exact finite result showing that, for that four-member local update, all action-conditioned three-way label marginals identify the expected update while pairwise marginals need not; synthetic theory only, with novelty unresolved.
- [Focused RLVR literature scan](docs/research-literature-review-2026-10.md): recent verifier-noise, GRPO, reward-hacking, and audit-allocation overlap used to choose the question.
- [SNLI entailment base-feasibility screen](docs/cpu-snli-entailment-base-feasibility-v1-report.md): frozen CPU/offline no-update screen, audited non-pass, and claim limits.
- [Latest BoolQ study](docs/cpu-lm-boolq-verifier-rloo-development-v1-report.md): audited CPU-only binary verifier-reward RLOO non-pass, with raw bundle and limits.
- [RVL GRPO rollback validation](docs/rvl-grpo-partial-failure-report.md): actual pinned CPU trainer step followed by injected failure; incumbent model/optimizer/RNG restoration passed on a tiny random model.
- [RVL GRPO in-step fault validation](docs/rvl-grpo-midstep-fault-report.md): exception raised inside the actual pinned `train_step` immediately after a real optimizer mutation; all 12 rollback checks passed on CPU.
- [GRPO integrity reproduction packet](docs/grpo-integrity-reproduction-packet.md): one command runs group/replay regressions and both pinned real-trainer rollback smokes, retaining their outputs and hashes.
- [Rollout group integrity and evaluation recovery](docs/rollout-group-integrity-v1-report.md): complete GRPO/RLOO groups survive rollout budgets, replay capacity, and repeated round indices; the RVL adapter restores incumbent state after generation or partial-restore failures.
- [BoolQ v17 DPO/SFT/anchored-DPO study](docs/boolq-posttraining-development-v17-report.md): earlier matched comparison and its failed advancement gate.
- [HH-RLHF reward-model v2](docs/hh-reward-model-v2-report.md): development pass, repeated confirmation accuracy gain, but no confirmed NLL improvement.
- [HH human-preference DPO development v2](docs/hh-human-preference-dpo-development-v2-report.md): matched DPO/SFT update on human-labeled pairs; the frozen DPO-vs-base accuracy gate failed.
- [HH length-normalized DPO development v1](docs/cpu-hh-length-normalized-dpo-v1-report.md): three-arm, three-seed CPU comparison; candidate tied standard DPO and failed its gain gates. The runner summary error and adapter-based recovery are recorded.
- [HH DPO response-length diagnostic](docs/hh-human-preference-dpo-length-diagnostic-v1.md): post-hoc analysis of the same development cohort; it does not change the non-pass.
- [Procedural binary-action DPO study](docs/cpu-procedural-entailment-dpo-development-v1-report.md): frozen CPU development non-pass; DPO lost to scalar calibration and exceeded its KL limit.
- [HH-RLHF v2 post-hoc calibration diagnostic](docs/hh-reward-model-calibration-analysis.md): calibrated-versus-raw NLL on the already-opened confirmation bundle; exploratory only and does not change v2's non-pass.
- [HH-RLHF v3 fixed-head calibration](docs/hh-reward-model-v3-development-report.md): development and fresh confirmation NLL gates passed, with same-host replay audits; the result is narrow and outcome-informed. [Confirmation report](docs/hh-reward-model-v3-confirmation-report.md).
- [HH-RLHF v3 replay guide](docs/hh-reward-model-v3-reproduction.md): pinned assets, offline audit command, expected result, and reproduction limits.
- [BoolQ v17 technical walkthrough](docs/boolq-v17-walkthrough.md): model update, objectives, selection rule, audits, and limits in one path.
- [GSM8K model study](docs/cpu-lm-gsm8k-dpo-confirmation-v1-report.md): protocol, per-seed outcomes, audit, and limits.
- [Experiment index](docs/experiments.md): full study sequence and retained reports.
- [Current evidence gaps](docs/current-gaps.md): what the results support and what remains open.
- [Next-study decision](docs/next-study-decision-2026-10-09.md): current evidence-based stop decision, retired model/task pairings, and the gates required before another learner study.
- [Generated code-repair feasibility v1](docs/cpu-code-repair-feasibility-v1-report.md): audited 0/32 base-only result, tool-loop failure modes, CPU cost, and decision to retire the pairing.
- [Generated arithmetic feasibility v1](docs/cpu-generated-arithmetic-feasibility-v1-report.md): a preregistered CPU-only base-rollout gate that failed before training.
- [Evaluation runner walkthrough](docs/walkthrough.md): inspect the calibrated task runner and retained execution evidence.

Contributors should read [`AGENTS.md`](AGENTS.md) before changing experiments, code, or claims.

## Selected evidence

| Area | Result | What it supports |
| --- | --- | --- |
| Latest human-preference policy comparison | HH helpful-base: length-normalized DPO 0.4974, standard DPO 0.4974, base 0.5000, length-only 0.5645. | Frozen development non-pass. Same-host adapter replay passed; no downstream task success or external reproduction. |
| Latest real-model feasibility screen | Qwen2.5-0.5B ticket-tool base v1: 0/48 exact successes, 0/48 schema-valid outputs, and 0/120 matched authorized calls; resource caps passed. | Frozen task/schema gate failed; same-host audit passed. This does not establish general tool-use inability or an RL result. |
| Earlier real-model feasibility screen | SNLI binary entailment v1: 49.41% balanced accuracy on 512 balanced validation rows; same-host model-forward audit passed. | The frozen base-rate screen failed; no update or confirmation followed. This is not a post-training or capability result. |
| Latest model study | BoolQ v17: all three matched methods missed the frozen task-gain gate. Best balanced-accuracy gain was 0.11 percentage points against a 5-point threshold. | An audited development comparison. Confirmation rows remain unopened. |
| Latest model study | BoolQ binary verifier-RLOO v1: five-seed development and same-host independent audit completed; frozen gate non-pass. | RLOO gained 2.69 points over base on the selected development comparison, below the 5-point rule; its paired interval crossed zero. No confirmation or general learning claim. |
| Latest tool-call-attempt screen | Generated code repair: 0/32 successes across eight templates; no accepted edit, visible test, or finish. | Audited non-pass; 0/69 authorized schema-valid calls, 60 unsafe/unauthorized attempts. CPU/RSS passed, but the model/task/tool pairing is retired. The newer Qwen math screen made no calculator calls and does not measure tool use. |
| Human-preference reward model | HH-RLHF v2: confirmation accuracy +7.68 points vs baseline; NLL difference interval [−0.0539, +0.0177]. | Accuracy repeated on one fresh cohort; the joint confirmation gate failed because NLL improvement remained uncertain. |
| Human-preference policy update | HH helpful-base DPO v2: development pair accuracy 0.4128 vs frozen-base 0.4141; paired 95% interval [−0.0117, +0.0104]. | The frozen preference-gain and seed-consistency gates failed; no confirmation or downstream task-success claim. |
| Synthetic binary-action DPO | Procedural entailment v1: base BA 0.5078; scalar calibration 0.6211; contextual DPO 0.5573. | DPO lost to scalar calibration, missed the base/gain gates, and exceeded the KL cap; synthetic development evidence only. |
| Reward-score scaling | Outcome-informed HH-RLHF v3 passed its fixed-head ΔNLL gate on development (512 prompts) and confirmation (306 prompts); confirmation mean ΔNLL −0.1745, 95% interval [−0.2337, −0.1178]. | Narrow same-split score-scale evidence. Accuracy stayed fixed and ECE worsened; no task or RL gain is shown. |
| Model-level preference update | GSM8K: held-out conditional preference NLL changed by −0.00697 nats/question across three seeds; accuracy moved from 0.4943 to 0.4956. | A narrow forced-choice preference result, not free-form reasoning or capability evidence. |
| Synthetic preference robustness | Ten-seed clean/noise/shift study passed its declared synthetic NLL/KL rule; label flips worsened NLL on every seed. | Behavior under one known synthetic preference generator. |
| Evaluation reliability | Three historical source graders distinguish pinned pre-fix and fixed revisions; recovery study matched 24/24 frozen fault cases. | Specific grader and same-host recovery checks, not broad grader soundness or distributed reliability. |
| GRPO group-audit observability | 2,000-seed exact synthetic study: item-only audit balanced accuracy 49.2%/50.0%, group-atomic 100%, exact item-law TV 0. | A two-member partial-label identifiability counterexample. No optimizer, policy update, or capability claim; separately implemented same-host replay passed, outside reproduction pending. |
| Provenance version validation | Direct lag and RVL replay paths reject future or malformed versions; existing stale and pending data behavior remains covered. | Correctness regression fix with a pre-fix reproducer, not policy-quality evidence. |
| RVL GRPO adapter integrity | Two real pinned CPU GRPO smokes injected failure after optimizer mutation: one at the candidate boundary and one before `train_step` returned. Both restored incumbent model/optimizer/RNG state, and the next rollout used the incumbent. | One pinned trainer revision and a 3,696-parameter random GPT-2. These are rollback checks, not pretrained-model or learning-quality evidence. |

The [evidence notes](docs/evidence.md) and individual reports define each result's data, protocol, audit coverage, and claim boundary. The [roadmap](docs/roadmap.md) records unresolved evidence levels. The full historical series, including failed and incomplete attempts, remains in the [experiment index](docs/experiments.md).

## Post-training results

The latest [BoolQ v17 study](docs/boolq-posttraining-development-v17-report.md) tested DPO, answer SFT and anchored DPO on the same fresh validation questions across three seeds. All audits passed, but the best balanced-accuracy gain was 0.11 percentage points against the frozen 5-point gate. Checkpoint selection used this development set, and confirmation data was not opened. The result does not establish task or capability improvement.

The first locked real-model confirmation used 256 verifier-labeled training questions, all 1,319 GSM8K test questions, and three fresh adapter seeds. Mean held-out conditional preference NLL improved by 0.00697 nats/question (paired 95% interval [−0.00800, −0.00591]); all three seeds improved and stayed under the frozen KL ceiling. Accuracy remained close to chance (0.4943 → 0.4956). This is narrow forced-choice preference evidence using a custom two-token output-head adapter—not free-form response or reasoning evidence. The [protocol](protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json), [report](docs/cpu-lm-gsm8k-dpo-confirmation-v1-report.md), and [audited raw bundle](results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/) retain the full result. The prior cached-model arithmetic study remains a [separate non-pass](docs/cpu-lm-dpo-head-v1-report.md).

VARE retains two accepted synthetic preference-policy studies. The v2 confirmation improved held-out synthetic NLL on all 10 seeds, though its budget-selection chronology is not independently anchored. A separate frozen noise/shift study was committed before its run; its audit reconstructed all 80 arm-condition-seed records, and its clean arm passed the held-out NLL/KL rule on all 10 seeds. The flip arms worsened as label noise increased, and the base-trained policy also scored worse under the declared preference shift. A fresh shallow clone of commit `70e4660` reproduced its independent training/metric audit on Python 3.12.12, run by the project author; this is not external human reproduction. These are controlled results on synthetic policies and labels, not language-model updates or capability evidence. See the [v2 report](docs/synthetic-dpo-v2-report.md), [noise/shift report](docs/synthetic-preference-robustness-v1-report.md), and [remaining evidence gaps](docs/current-gaps.md).

A separate [procedural binary-action study](docs/cpu-procedural-entailment-dpo-development-v1-report.md) tested contextual DPO against scalar calibration and matched SFT on a generated implication task. It failed its development gate: the base had near-zero Yes recall, scalar calibration outscored contextual DPO, and DPO exceeded its KL cap. The auditor reconstructed the generated rows and stored-feature metrics. No confirmation data was opened; the result remains synthetic mechanism evidence only.

## Measured execution evidence

- **24/24 recovery fault cases** passed under a frozen protocol: process death, rollback, lost acknowledgement, late completion, input invalidation and retry exhaustion. The retained exports contain 117 synthetic jobs/126 attempts, plus four historical jobs recovered with all calibrated outcomes preserved. See the [recovery report](docs/recovery-report.md).

- **TRL grader mutation checks:** successive frozen audits found false accepts for branch-local rebinding, final-return rebinding, in-place loss mutation, direct-alias mutation, unreachable branches and zeroed policy-loss numerators. Protocol v8 rejects the early-return and zeroed-numerator mutations that v7 accepted, while accepting the unmodified fixed source across 12 arithmetic conditions. See the [v8 mutation comparison](results/trl-grpo-accumulation-window-normalizer-v1/grader-mutation-v8/summary.json), [frozen protocols and patches](benchmarks/audits/trl_loss_dataflow_v1/), and [v8 calibration](results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v8/summary.json). This remains a narrow source-structural check, not full trainer or gradient execution.

- **RVL generation-setting mutation checks:** v3 rejects inherited non-neutral `typical_p`; v4 adds neutral `suppress_tokens` and `no_repeat_ngram_size` checks. A candidate that carries pretrained suppression through is accepted by v3 and rejected by v4, while the pinned fixed source passes v4. See the [v4 mutation evidence](results/rvl-hf-behavior-policy-parity-v1/generation-config-mutation-v4/summary.json) and [v4 calibration](results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v4/summary.json).
- **Inherited `bad_words_ids` audit:** the existing v4 grader accepts one constructed candidate that adds a single-token bad-word constraint; the new locked CPU task rejects it while accepting the fixed source. The fixture covers one token-level case only. See the [audit report](docs/rvl-bad-words-report.md), [mutation comparison](results/rvl-hf-bad-words-neutrality-v1/mutation-audit-v2/summary.json), and [task calibration](results/rvl-hf-bad-words-neutrality-v1/cpu-calibration-v1/summary.json).

- **Local CPU agent pilot:** the corrected v2 tool loop ran one small local model on one pinned task for three formal seeds. It made no source edits and the locked grader rejected all three unchanged checkouts. The original v1 cohort is invalidated because of a tool-history serialization bug. See the [pilot report](docs/local-agent-pilot.md).

- **Promotion-gate input validation:** a frozen CPU regression task found that the pre-fix promotion gate accepted all seven tested reports containing `NaN` or infinity. The fixed gate rejects those inputs and preserves acceptance for a finite positive control. See the [report](docs/promotion-gate-report.md) and [raw result](results/promotion-gate-metrics-v1/summary.json).

- **Replay group freshness:** a frozen CPU regression task found that current-freshness filtering could return 3 of 4 members from a comparison group. The fixed sampler drops that entire group and retains a separate fresh group intact. See the [report](docs/replay-group-freshness-report.md) and [raw result](results/replay-group-freshness-v1/summary.json).

- **16/16 synthetic reliability cases** matched their declared outcomes, including incorrect provenance, source mutation, malformed output, timeouts, output flooding and subprocess cleanup.
- **Environment command output cap regression:** the frozen v4 task reproduces accepted stdout/stderr floods at the baseline; the fix rejects each overflow, retains at most the configured bytes per stream, and terminates the POSIX process group while preserving normal command behavior. See the [report](docs/environment-output-budget-report.md) and [raw results](results/environment-output-budget-v4/summary.json). This is not an OS resource sandbox.
- **Synthetic preference-optimization v1:** clean-label DPO-style updates lowered held-out synthetic NLL by 0.3523 nats/pair, but exceeded the frozen KL ceiling (0.5629 vs 0.5); its original reference accuracy also mishandled ties. Preserve it as a diagnostic non-pass in the [v1 report and bundle](docs/synthetic-dpo-report.md).
- **Synthetic preference-optimization v2:** a reported training-only sweep informed 100 updates; its raw output was not retained. A separate post-run development replay also selects 100 under the stated rule. On a separate 10-seed confirmation cohort, mean held-out NLL improved by 0.3074 nats/pair (paired 95% bootstrap interval 0.2928–0.3226), all seeds improved, and mean KL was 0.3874 under the recorded 0.5 ceiling. The audit reconstructs all 40 seed-by-arm results and identifies an invalid shuffled-ID arm, excluded from inference. See the [v2 report](docs/synthetic-dpo-v2-report.md) and [development/confirmation evidence](results/synthetic-dpo-cpu-v2/). Synthetic mechanism evidence only.
- **Preference noise/shift v1:** a protocol committed before the run compared clean training with 20%/40% pair-orientation flips across 10 seeds. The clean arm improved base-teacher held-out NLL by 0.3174 nats/pair (paired seed bootstrap 95% interval 0.3021–0.3328), and all 10 seeds passed the frozen rule under a 0.3952 mean KL. A separate implementation re-trained the policies and rechecked every retained metric without importing the runner or original auditor; it produced matching metrics on Python 3.9.6, 3.11.15, and 3.12.12. The original run's context/action-pair generation is not verified bit-for-bit across Python versions, and no outside person has reproduced it. This is synthetic sensitivity evidence only. See the [report](docs/synthetic-preference-robustness-v1-report.md), [independent audit records](results/synthetic-preference-robustness-v1/independent-audits/), and [bundle](results/synthetic-preference-robustness-v1/confirmation/).
- **80 historical-source evaluations** preserved the expected pre-fix rejection/fixed acceptance decisions across one-worker and four-worker campaigns.
- **3.4857× median paired speedup** with four workers versus this same runner with one worker, over five pairs on a local macOS arm64 CPU. This is a small local-grader measurement, not a distributed or model-serving result.

- **Freshness validation scales with the completing job:** under a frozen serial 8/16/32-job CPU protocol, commit-time fingerprint checks fell from 64/256/1,024 to 8/16/32. At 32 jobs, median completion time fell from 28.479s to 1.082s on this machine. Full input refresh remains at export. This repeated-fixture coordinator measurement is machine-local, not heterogeneous grader throughput. See the [frozen report](docs/freshness-scaling-report.md).

Start with the [ten-minute walkthrough](docs/walkthrough.md), [technical report](docs/scheduler-report.md), and [execution contract](docs/execution.md). Inspect retained evidence offline:

The [post-training plan](docs/post-training-plan.md) defines the no-cost learning experiments and their claim limits. The [current evidence gaps](docs/current-gaps.md) rank the remaining CPU-feasible work and state what the retained results support. [`cpu_lm_dpo_head_v1`](protocols/cpu_lm_dpo_head_v1.lock.json) freezes the bounded no-download model-level preference update; its result and first failed attempt are retained separately.

If the exact model snapshot and compatible `torch`, `transformers`, and `numpy` packages are already installed locally, run the frozen study with:

```bash
python scripts/run_cpu_lm_dpo_head.py \
  --model-dir /path/to/7ae557604adf67be50417f59c2c2f167def9a775 \
  --output results/cpu-lm-dpo-head-v1
python scripts/audit_cpu_lm_dpo_head.py results/cpu-lm-dpo-head-v1
```

The runner refuses any other snapshot revision, forces offline loading and CPU placement, and aborts at the protocol's wall-time or peak-memory limit. The audit reconstructs data, recorded-margin metrics and the decision; it does not rerun model inference or training.

```bash
python3 scripts/audit_scheduler.py results/cpu-scheduler-v1
python3 -m unittest discover -s tests -p 'test_runner.py' -v
python3 -m unittest discover -s tests -p 'test_durable.py' -v
python3 scripts/audit_recovery.py results/cpu-recovery-v1
```

For a new campaign, use `python3 -m vare run --plan PLAN.json --output NEW_DIRECTORY --workers 4`, then `python3 -m vare audit NEW_DIRECTORY`. The execution contract provides the plan format. The runner needs Python >=3.9, Git and POSIX process groups; it has no third-party Python dependency.

## Persistent evaluation and requirements

`durable-init`, `durable-work`, `durable-export` and `durable-audit` add same-host transactional claims, fenced leases, bounded crash retries, immutable attempt records and freshness invalidation. See the [requirements matrix](docs/requirements.md) and [recovery contract](docs/recovery.md). The installed package and experimental control-loop commands require Python >=3.11; standalone checkout evaluation uses Python >=3.9. All implementation is under `src/vare`, with a checkout bootstrap in `vare/`.

## Current scope

Two real upstream fixes calibrate the task and grading path:

| Task | What the grader exercises | Pre-fix result | Fixed result |
| --- | --- | --- | --- |
| [HF behavior-policy parity](benchmarks/historical/rvl_behavior_policy_parity/TASK.md) | Protocol v4 checks rollout probabilities and the learner's actual `_sample_objective` path with neutral sampling truncation, repetition, no-repeat-ngram, and token-suppression settings on CPU fixtures. | Maximum absolute rollout log-probability error `0.5570`; learner error `0.3711`; rejected. | Both errors `0.0`; accepted. |
| [Inherited bad-word neutrality](benchmarks/historical/rvl_bad_words_neutrality/TASK.md) | A separate locked grader checks that single-token `bad_words_ids` does not alter the six CPU-fixture rollout conditions; it also preserves the v4 checks. | Maximum rollout error `0.5570`; learner error `0.3711`; rejected. | Both errors `0.0`; accepted. |
| [TRL accumulation-window normalizer](benchmarks/historical/trl_grpo_accumulation_scale/TASK.md) | Protocol v8 executes six source-derived arithmetic cases in each of two trainer paths, checks masked numerator/returned-loss reachability, rejects later normalizer writes and unapproved loss mutations, and checks the selected denominator. | Maximum absolute normalizer error `12`; rejected. | Maximum error `0`; accepted. |

The second task tracks the upstream [TRL issue](https://github.com/huggingface/trl/issues/5619) and [fix](https://github.com/huggingface/trl/pull/6024). The first uses a pinned fix in [Recursive-Verification-Lag](https://github.com/mitukx/Recursive-Verification-Lag).

Raw grader outputs, summaries, protocol snapshots, and SHA-256 manifests are retained under [`results/`](results/). See the [evidence notes](docs/evidence.md) for exact revisions, metrics, and limits. Earlier HF protocol results remain as history; the separate bad-word task expands the covered settings without changing v4's lock.

The separate [integrity calibration](results/protocol-integrity/cpu-calibration-v5/summary.json) changes the task descriptor, task brief, and grader one at a time for all three locked historical task protocols. All nine changes are rejected against the unchanged protocol locks. The checked-in Git history is the trust anchor for those locks; this does not detect coordinated edits to a lock itself.

The [local agent pilot](docs/local-agent-pilot.md) is one small-model, one-task negative result: the corrected formal cohort read the task source, but produced no accepted source edits. It does not characterize coding agents generally or establish successful task solving or generalization. The original v1 runs are invalidated and excluded from inference.

The E1 task/evaluator calibrations do not measure a model training or capability improvement. The separate local agent pilot had no successful patch and is reported as a negative result. The TRL grader executes extracted production normalization statements with scalar doubles and checks a direct AST denominator contract; it does not execute the full loss expression or run a trainer/gradient update. The HF grader executes candidate source with local fixtures; it is not an operating-system sandbox for hostile code. Only run it on candidate code you trust.

The restored experimental control plane in `src/vare` includes grouped replay, freshness/curriculum controls, paired promotion gates, workspace-agent campaigns and RVL hooks. These are implementation/regression contracts; no model update through this control plane has been measured. Separate CPU model studies are summarized above. No successful local-agent task solution or GPU result is claimed. The imported L0 raw archive is corrupt; its old accuracy summary is excluded from verified claims (see the evidence notes).

## Reproduce the calibrations

Requirements: Python 3.9 or newer, Git, and network access to fetch the pinned public source revisions. The scripts use only the Python standard library. No model weights, GPU, paid API, or external compute are used for local calibration.

Run each calibration into a new directory outside the repository. The checked-in result directories already exist, so choose a fresh path:

```bash
python3 scripts/calibrate_task.py \
  --output /tmp/vare-rvl-calibration

python3 scripts/calibrate_task.py \
  --task-root benchmarks/historical/rvl_bad_words_neutrality \
  --output /tmp/vare-rvl-bad-words-calibration

python3 scripts/audit_rvl_bad_words_mutation.py \
  --output /tmp/vare-rvl-bad-words-mutation

python3 scripts/calibrate_task.py \
  --task-root benchmarks/historical/trl_grpo_accumulation_scale \
  --output /tmp/vare-trl-calibration

python3 scripts/calibrate_integrity.py \
  --output /tmp/vare-integrity-calibration
```

The calibrator verifies the task, brief, and grader hashes against the protocol lock; fetches the immutable pre-fix and fixed source revisions; grades both; and writes raw JSON, a summary, a source/protocol snapshot, and a manifest. The preregistered outcome requires the pre-fix revision to fail and the known fixed revision to pass.

To prepare and grade a candidate workspace manually:

```bash
python3 scripts/prepare_task.py \
  --workspace /tmp/vare-rvl-candidate
# Make a candidate change in that checkout.
python3 scripts/grade_task.py \
  --workspace /tmp/vare-rvl-candidate
```

For the TRL task, provide `--task-root benchmarks/historical/trl_grpo_accumulation_scale` to both commands. Candidate workspaces must be outside the VARE checkout. The grader and locked task files stay outside the candidate workspace.

## Roadmap

The [roadmap](docs/roadmap.md) tracks the evidence needed before expanding the claims. The main open learning gap is a reproducible improvement in an independently measured model task outcome. The synthetic shift study and narrow forced-choice result do not close that gap. The local agent pilot remains supporting evidence about task execution, not the project's primary research direction.

## Develop the full package

Use Python >=3.11 in a local virtual environment. The runtime has no third-party dependencies; pytest is only for development tests. Use an editable install from this checkout because task/protocol assets are repository files.

```bash
python -m pip install -e . pytest==8.4.2
python -m pytest
vare --help
```

## License

Apache-2.0. See [LICENSE](LICENSE).
