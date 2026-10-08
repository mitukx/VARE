# VARE

VARE is a CPU-first research control plane for studying post-training signals and policy improvement under limited compute. It combines independently graded tasks, locked experiment protocols, provenance-aware replay, bounded execution, candidate promotion checks, and auditable retained records. Its learning evidence includes controlled synthetic policy updates, one preregistered cached-model non-pass, and one narrow verifier-labeled preference improvement on a cached language model. The latter changed held-out NLL by −0.00697 nats per question while accuracy stayed near chance; no capability gain is claimed. See the [GSM8K report](docs/cpu-lm-gsm8k-dpo-confirmation-v1-report.md) and [earlier model non-pass](docs/cpu-lm-dpo-head-v1-report.md).

The separate [sequence-level GSM8K development result](docs/cpu-lm-gsm8k-sequence-dpo-development-v2-report.md) lowered verifier-labeled validation preference NLL under its KL cap, but did not improve free-form exact-match accuracy. It is a small development candidate pending a separately locked confirmation, not a capability result.

The first 512-question CPU confirmation exceeded its two-hour limit with two of three seeds completed; it is explicitly [recorded as incomplete](docs/cpu-lm-gsm8k-sequence-dpo-confirmation-v1-incomplete.md). A smaller fresh-cohort confirmation is locked separately.

That fresh 128-question confirmation lowered verifier-preference NLL with a paired 95% interval below zero and increased exact-match in all three seeds, but mean KL was 0.737 against a 0.5 cap. Its frozen decision is **non-pass**; see the [confirmation report](docs/cpu-lm-gsm8k-sequence-dpo-confirmation-v2-report.md).

A lower-rate development update then met its KL cap but lowered mean exact-match below baseline (1.56% → 1.04%). It is a separate [development non-pass](docs/cpu-lm-gsm8k-sequence-dpo-development-v3-report.md); no further confirmation data was opened.

The rationale-versus-base-rollout [v4 development run](docs/cpu-lm-gsm8k-sequence-dpo-development-v4-report.md) reduced verifier-preference NLL, but exact-match was 0/16 for both base and all updated seeds. Its candidate gate was therefore non-informative and no confirmation was opened. The next attempt hit the frozen 6-GiB memory ceiling and is retained as incomplete. No free-form improvement has been demonstrated.

The reduced [v6 run](docs/cpu-lm-gsm8k-sequence-dpo-development-v6-report.md) stayed under the memory ceiling and passed its audit, but generation was truncated before final answers and exact-match remained 0/8 for the base and every update. The numeric-only preference setup is the next fresh development probe; the repository still does not demonstrate free-form improvement.

One numeric-only setup attempt stopped before inference because its batch size violated the decoder's unpadded-input requirement. It is retained as a [setup failure](docs/cpu-lm-gsm8k-sequence-dpo-development-v7-setup-failure.md); the follow-up uses single-prompt generation on fresh rows.

The audited numeric-only [v8 run](docs/cpu-lm-gsm8k-sequence-dpo-development-v8-report.md) lowered preference NLL but failed the nonzero-baseline gate: base exact-match was 0/32 and updated seeds were 1/32, 0/32, and 0/32. This single updated answer is not enough to support an improvement claim. A larger fresh development cohort is next.

The larger rank-8 [v9 run](docs/cpu-lm-gsm8k-sequence-dpo-development-v9-report.md) had 6/128 base exact matches and a mean 4.67/128 after update. Preference NLL improved at epoch 2 while exact-match fell; epoch 4 also breached the KL cap. It is an audited non-pass, and no capability gain is claimed.

In [v10](docs/cpu-lm-gsm8k-sequence-dpo-development-v10-report.md), exact-match was evaluated at every NLL/KL-eligible checkpoint. Epoch 2 was selected at 2/64, 0/64, and 2/64 across seeds, but base was 0/64 and the mean gain gate failed. The preference fit improved; task accuracy stayed very low and seed-sensitive.

Contributors should read [`AGENTS.md`](AGENTS.md) before changing experiments, code, or claims.

## Post-training results

The first locked real-model confirmation used 256 verifier-labeled training questions, all 1,319 GSM8K test questions, and three fresh adapter seeds. Mean held-out conditional preference NLL improved by 0.00697 nats/question (paired 95% interval [−0.00800, −0.00591]); all three seeds improved and stayed under the frozen KL ceiling. Accuracy remained close to chance (0.4943 → 0.4956). This is narrow forced-choice preference evidence using a custom two-token output-head adapter—not free-form response or reasoning evidence. The [protocol](protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json), [report](docs/cpu-lm-gsm8k-dpo-confirmation-v1-report.md), and [audited raw bundle](results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/) retain the full result. The prior cached-model arithmetic study remains a [separate non-pass](docs/cpu-lm-dpo-head-v1-report.md).

VARE retains two accepted synthetic preference-policy studies. The v2 confirmation improved held-out synthetic NLL on all 10 seeds, though its budget-selection chronology is not independently anchored. A separate frozen noise/shift study was committed before its run; its audit reconstructed all 80 arm-condition-seed records, and its clean arm passed the held-out NLL/KL rule on all 10 seeds. The flip arms worsened as label noise increased, and the base-trained policy also scored worse under the declared preference shift. These are controlled results on synthetic policies and labels, not language-model updates or capability evidence. See the [v2 report](docs/synthetic-dpo-v2-report.md), [noise/shift report](docs/synthetic-preference-robustness-v1-report.md), and [remaining evidence gaps](docs/current-gaps.md).

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
- **Preference noise/shift v1:** a protocol committed before the run compared clean training with 20%/40% pair-orientation flips across 10 seeds. The clean arm improved base-teacher held-out NLL by 0.3174 nats/pair (paired seed bootstrap 95% interval 0.3021–0.3328), and all 10 seeds passed the frozen rule under a 0.3952 mean KL. NLL worsened on every seed in both flip arms, and mean NLL also rose under the explicitly shifted teacher. The offline audit regenerated the data and reconstructed all 80 condition-level records; detached-clone verification and the [Ubuntu/Python 3.11 workflow](https://github.com/mitukx/VARE/actions/runs/37717689087) also passed. This is synthetic sensitivity evidence only. See the [report](docs/synthetic-preference-robustness-v1-report.md) and [bundle](results/synthetic-preference-robustness-v1/confirmation/).
- **80 historical-source evaluations** preserved the expected pre-fix rejection/fixed acceptance decisions across one-worker and four-worker campaigns.
- **3.4857× median paired speedup** with four workers versus this same runner with one worker, over five pairs on a local macOS arm64 CPU. This is a small local-grader measurement, not a distributed or model-serving result.

- **Freshness validation scales with the completing job:** under a frozen serial 8/16/32-job CPU protocol, commit-time fingerprint checks fell from 64/256/1,024 to 8/16/32. At 32 jobs, median completion time fell from 28.479s to 1.082s on this machine. Full input refresh remains at export. This repeated-fixture coordinator measurement is machine-local, not heterogeneous grader throughput. See the [frozen report](docs/freshness-scaling-report.md).

Start with the [ten-minute walkthrough](docs/walkthrough.md), [technical report](docs/scheduler-report.md), and [execution contract](docs/execution.md). Inspect retained evidence offline:

The [post-training plan](docs/post-training-plan.md) defines the no-cost learning experiments and their claim limits. The [current evidence gaps](docs/current-gaps.md) rank the remaining CPU-feasible work and state what the retained results support. [`cpu_lm_dpo_head_v1`](protocols/cpu_lm_dpo_head_v1.lock.json) freezes the bounded no-download model-level preference update; its result and first failed attempt are retained separately.

For a visual, read-only view of the retained experiments, serve [`ui/`](ui/) locally with the steps in [`docs/internal-workbench.md`](docs/internal-workbench.md). The interface reads committed JSON evidence and has no experiment submission/backend path.

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

The restored experimental control plane in `src/vare` includes grouped replay, freshness/curriculum controls, paired promotion gates, workspace-agent campaigns and RVL hooks. These are implementation/regression contracts. No real-model learning, successful local-agent task solution, or GPU result is claimed. The imported L0 raw archive is corrupt; its old accuracy summary is excluded from verified claims (see the evidence notes).

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

The [roadmap](docs/roadmap.md) tracks the evidence needed before expanding the claims. The next major gaps are signal-shift falsification and any feasible, independently measured model-level update. The current agent pilot remains supporting evidence about task execution, not the project's primary research direction.

## Develop the full package

Use Python >=3.11 in a local virtual environment. The runtime has no third-party dependencies; pytest is only for development tests. Use an editable install from this checkout because task/protocol assets are repository files.

```bash
python -m pip install -e . pytest==8.4.2
python -m pytest
vare --help
```

## License

Apache-2.0. See [LICENSE](LICENSE).
