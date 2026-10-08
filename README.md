# VARE

VARE is a CPU-first research project for evaluating post-training signals and policy updates under limited compute. It combines frozen experiment protocols, independently checked outcomes, provenance-aware execution, and retained raw records.

HH-RLHF v2 passed development but **failed its joint confirmation gate**: accuracy repeated while the NLL interval crossed zero. An outcome-informed fixed-head follow-up v3 passed its NLL gate on development and a fresh confirmation cohort, with independent same-host score-replay audits. This is a narrow score-scale result: ranking accuracy was unchanged, ECE worsened, and the threshold was informed by v2. A separate human-preference DPO development comparison failed its frozen preference-accuracy gain and seed-consistency gates (0.4128 DPO vs 0.4141 base). The latest matched task-success comparison remains BoolQ v17, where all three methods missed the frozen task-gain gate. A generated-arithmetic rollout feasibility pilot also failed its predeclared base-accuracy gate (1/64); no training or confirmation followed. The strongest completed positive model result remains a narrow verifier-labeled answer-choice NLL improvement; accuracy remained near chance. Synthetic policy studies test mechanisms, not language-model capability. No general capability improvement or scale result is claimed.

## Start here

- [Latest BoolQ study](docs/boolq-posttraining-development-v17-report.md): matched DPO, SFT, and anchored-DPO comparison, including its failed advancement gate.
- [HH-RLHF reward-model v2](docs/hh-reward-model-v2-report.md): development pass, repeated confirmation accuracy gain, but no confirmed NLL improvement.
- [HH human-preference DPO development v2](docs/hh-human-preference-dpo-development-v2-report.md): matched DPO/SFT update on human-labeled pairs; the frozen DPO-vs-base accuracy gate failed.
- [HH DPO response-length diagnostic](docs/hh-human-preference-dpo-length-diagnostic-v1.md): post-hoc analysis of the same development cohort; it does not change the non-pass.
- [HH-RLHF v2 post-hoc calibration diagnostic](docs/hh-reward-model-calibration-analysis.md): calibrated-versus-raw NLL on the already-opened confirmation bundle; exploratory only and does not change v2's non-pass.
- [HH-RLHF v3 fixed-head calibration](docs/hh-reward-model-v3-development-report.md): development and fresh confirmation NLL gates passed, with same-host replay audits; the result is narrow and outcome-informed. [Confirmation report](docs/hh-reward-model-v3-confirmation-report.md).
- [HH-RLHF v3 replay guide](docs/hh-reward-model-v3-reproduction.md): pinned assets, offline audit command, expected result, and reproduction limits.
- [BoolQ v17 technical walkthrough](docs/boolq-v17-walkthrough.md): model update, objectives, selection rule, audits, and limits in one path.
- [GSM8K model study](docs/cpu-lm-gsm8k-dpo-confirmation-v1-report.md): protocol, per-seed outcomes, audit, and limits.
- [Experiment index](docs/experiments.md): full study sequence and retained reports.
- [Current evidence gaps](docs/current-gaps.md): what the results support and what remains open.
- [Generated arithmetic feasibility v1](docs/cpu-generated-arithmetic-feasibility-v1-report.md): a preregistered CPU-only base-rollout gate that failed before training.
- [Evaluation runner walkthrough](docs/walkthrough.md): inspect the calibrated task runner and retained execution evidence.

Contributors should read [`AGENTS.md`](AGENTS.md) before changing experiments, code, or claims.

## Selected evidence

| Area | Result | What it supports |
| --- | --- | --- |
| Latest model study | BoolQ v17: all three matched methods missed the frozen task-gain gate. Best balanced-accuracy gain was 0.11 percentage points against a 5-point threshold. | An audited development comparison. Confirmation rows remain unopened. |
| Human-preference reward model | HH-RLHF v2: confirmation accuracy +7.68 points vs baseline; NLL difference interval [−0.0539, +0.0177]. | Accuracy repeated on one fresh cohort; the joint confirmation gate failed because NLL improvement remained uncertain. |
| Human-preference policy update | HH helpful-base DPO v2: development pair accuracy 0.4128 vs frozen-base 0.4141; paired 95% interval [−0.0117, +0.0104]. | The frozen preference-gain and seed-consistency gates failed; no confirmation or downstream task-success claim. |
| Reward-score scaling | Outcome-informed HH-RLHF v3 passed its fixed-head ΔNLL gate on development (512 prompts) and confirmation (306 prompts); confirmation mean ΔNLL −0.1745, 95% interval [−0.2337, −0.1178]. | Narrow same-split score-scale evidence. Accuracy stayed fixed and ECE worsened; no task or RL gain is shown. |
| Model-level preference update | GSM8K: held-out conditional preference NLL changed by −0.00697 nats/question across three seeds; accuracy moved from 0.4943 to 0.4956. | A narrow forced-choice preference result, not free-form reasoning or capability evidence. |
| Synthetic preference robustness | Ten-seed clean/noise/shift study passed its declared synthetic NLL/KL rule; label flips worsened NLL on every seed. | Behavior under one known synthetic preference generator. |
| Evaluation reliability | Three historical source graders distinguish pinned pre-fix and fixed revisions; recovery study matched 24/24 frozen fault cases. | Specific grader and same-host recovery checks, not broad grader soundness or distributed reliability. |

The [evidence notes](docs/evidence.md) and individual reports define each result's data, protocol, audit coverage, and claim boundary. The [roadmap](docs/roadmap.md) records unresolved evidence levels. The full historical series, including failed and incomplete attempts, remains in the [experiment index](docs/experiments.md).

## Post-training results

The latest [BoolQ v17 study](docs/boolq-posttraining-development-v17-report.md) tested DPO, answer SFT and anchored DPO on the same fresh validation questions across three seeds. All audits passed, but the best balanced-accuracy gain was 0.11 percentage points against the frozen 5-point gate. Checkpoint selection used this development set, and confirmation data was not opened. The result does not establish task or capability improvement.

The first locked real-model confirmation used 256 verifier-labeled training questions, all 1,319 GSM8K test questions, and three fresh adapter seeds. Mean held-out conditional preference NLL improved by 0.00697 nats/question (paired 95% interval [−0.00800, −0.00591]); all three seeds improved and stayed under the frozen KL ceiling. Accuracy remained close to chance (0.4943 → 0.4956). This is narrow forced-choice preference evidence using a custom two-token output-head adapter—not free-form response or reasoning evidence. The [protocol](protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json), [report](docs/cpu-lm-gsm8k-dpo-confirmation-v1-report.md), and [audited raw bundle](results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/) retain the full result. The prior cached-model arithmetic study remains a [separate non-pass](docs/cpu-lm-dpo-head-v1-report.md).

VARE retains two accepted synthetic preference-policy studies. The v2 confirmation improved held-out synthetic NLL on all 10 seeds, though its budget-selection chronology is not independently anchored. A separate frozen noise/shift study was committed before its run; its audit reconstructed all 80 arm-condition-seed records, and its clean arm passed the held-out NLL/KL rule on all 10 seeds. The flip arms worsened as label noise increased, and the base-trained policy also scored worse under the declared preference shift. A fresh shallow clone of commit `70e4660` reproduced its independent training/metric audit on Python 3.12.12, run by the project author; this is not external human reproduction. These are controlled results on synthetic policies and labels, not language-model updates or capability evidence. See the [v2 report](docs/synthetic-dpo-v2-report.md), [noise/shift report](docs/synthetic-preference-robustness-v1-report.md), and [remaining evidence gaps](docs/current-gaps.md).

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
