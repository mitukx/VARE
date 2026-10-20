# MLSys coding pack v1

**Evidence status: fixture calibration of an authored task pack.** No model was loaded. No agent process was started. The numbers below are properties of the frozen baseline and oracle files. The first table is an earlier in-process execution. Later rows use parent clocks. They are not coding-agent capability.

## Question

Can a coding-agent evaluation use a frozen train/held-out split of small MLSys utilities, with hidden correctness checks and a noise-aware performance rule, through the existing `EngineeringTaskSpec` interface?

## Pack

`benchmarks/mlsys_coding_v1` contains nine single-file Python utilities. Each defect is local and executable. None is an upstream historical commit. The candidate workspace is only `subject/`. Hidden checks live in `evaluator/` and run with `cwd_base` set to `evaluator`, which `CatalogEnvironmentVerifier` already passes as `evaluator_root`.

`metadata.grader_visibility` is `hidden` on every task. The isolated repository grader on the open trusted-scoring branch refuses to start unless that field is `public` and the operator opts in. These tasks therefore do not enter the candidate-visible grader path.

| Split | Task | Family | Gate |
| --- | --- | --- | --- |
| train | `mlsys-v1-row-log-softmax` | numerics | hidden correctness |
| train | `mlsys-v1-paired-shuffle-loader` | data-loader | hidden correctness |
| train | `mlsys-v1-warmup-cosine` | training-schedule | hidden correctness |
| train | `mlsys-v1-grad-accum-tail` | gradient-accumulation | hidden correctness |
| train | `mlsys-v1-range-sums` | numeric-kernel | correctness and repeated timing |
| held-out | `mlsys-v1-aging-scheduler` | scheduler | hidden correctness |
| held-out | `mlsys-v1-global-grad-clip` | gradient-clip | hidden correctness |
| held-out | `mlsys-v1-dataset-standardize` | data-standardization | correctness and repeated timing |
| held-out | `mlsys-v1-causal-attention` | attention | hidden correctness |

The partition is in `split.json`. It was fixed before any policy rollout. Held-out tasks are different defects, not resized copies of train tasks. The manifest SHA-256 is `2c67d85a6806b3bbcbc63676f64fc955c5c7b3c9c4127a5a3caade1541efb4ab`, stored in [`protocols/mlsys_coding_agent_eval_v1.json`](../protocols/mlsys_coding_agent_eval_v1.json).

The previous manifest hash was `18c5999f823f9a550a561a913023b87f1fc7bf5a86f481d319e4778212d97499`. The decision rule, seeds, reward constants, and split membership did not change. The hash changed because each task's trusted `metadata.task_type` is now `correctness` or `performance`, the graders call the harness adapter, and the paired-ratio statistic lives once in `evaluator_timing.py`. PR #85 pinned the previous hash at commit `c8c0488980425e6d316abc21dfc136f16c7e265b`.

## Reward

Correctness tasks succeed when the single hidden command exits 0 and integrity holds. `reward.decide` reads the serialized return code, timeout flag, and output-limit flag. `CommandResult.passed` is a property and is not in the result schema.

Performance tasks ask the harness adapter for seven interleaved parent clocks. Even indexes time the reference first and odd indexes time the candidate first. Each call gets a new input, and the parent rejects a reused object or a reused value. The grade command prints one JSON object on stdout, which the existing command-result schema already retains. The child's exit code and stdout are not the score. `task_type` comes from trusted metadata. A missing or prefixed measurement record scores 0; it is not treated as a correctness success. The protocol recomputes the lower end of a Student-t 95% interval on the paired ratios `baseline_seconds / candidate_seconds`, using the sample standard deviation (divisor `n - 1`) and the critical value `2.446911851144167` at 6 degrees of freedom. That lower bound is `conservative_speedup`. It is not `baseline_ci95_low / candidate_ci95_high`. Success requires correctness and `conservative_speedup >= 2`. When correctness passed, the shaped reward is `min(1, speedup / 4)`. The cap at 4 is twice the success threshold. It was not fit to a model outcome.

The grade command exits 0 only when that speedup threshold is met, so `EnvironmentResult.passed` matches protocol success for these fixtures. The shaped reward is still the training target defined for a later GRPO arm: `grader.score` is 0 for a correct but slow patch, while the shaped reward stays positive.

`benchmarks/mlsys_coding_v1/harness_adapter.py` is the caller. It reads `task_type` from the trusted `task.json` outside the candidate workspace. The interface it proposes for PR #85 is `vare.separated_candidate_harness.grade_function`, with a per-task `module:function` entrypoint, a parent-owned input generator, a parent-owned reference, a parent-owned checker, and that trusted `task_type`. PR #85's description did not name this symbol when the adapter was written. Until the import succeeds, the adapter uses `candidate_call.py` and records `harness_id` `temporary_shim_candidate_call`, `replaced_by` the proposed symbol, and `nonce_enforced` false. That shim is marked temporary. It is not a second harness, and the seven attack regressions named for #85 are not duplicated here.

## Calibration on this host

`python -m pytest tests/test_mlsys_coding_v1.py` passed: every unpatched subject failed the pre-registered gate, every oracle passed, public smoke checks passed on the unpatched subjects, and the candidate workspace did not contain the hidden files.

One earlier same-host execution of the performance grade commands, before the reward helper was pointed at serialized return codes, recorded:

| Task | Unpatched conservative speedup | Unpatched sample stdevs (reference, candidate), seconds | Oracle conservative speedup | Oracle sample stdevs (reference, candidate), seconds |
| --- | --- | --- | --- | --- |
| `mlsys-v1-range-sums` | 0.9990 | 0.01788, 0.001326 | 233.11 | 0.002491, 5.180e-5 |
| `mlsys-v1-dataset-standardize` | 0.9569 | 0.009462, 0.002678 | 669.16 | 0.005418, 1.198e-5 |

These wall-clock values are descriptive of the earlier in-process grader. The locked rule is the threshold of 2, not these floats.

After the graders moved to parent clocks around a child call, one host execution recorded:

| Task | Unpatched conservative speedup | Correctness | Oracle conservative speedup |
| --- | --- | --- | --- |
| `mlsys-v1-range-sums` | 0.775 | passed | 3.91 |
| `mlsys-v1-dataset-standardize` | 0.967 | passed | 23.01 |

The unpatched subjects are the correct slow algorithms. Both stayed under the threshold of 2 on that earlier unpaired estimator. The range-sums oracle cleared 2 and did not reach the reward cap of 4, because the parent clock includes the child round trip. The dataset oracle cleared the cap. Those floats are descriptive of that estimator.

After the bound moved to interleaved paired ratios, one host execution recorded range-sums baseline 0.635 and oracle 3.29, and dataset-standardize baseline 0.860 and oracle 23.38. The same host then repeated an explicit copy of the slow reference 20 times per task while four busy Python processes ran. False passes were 0/20 for range sums (paired lower bound min/median/max 0.507/0.618/0.773) and 0/20 for dataset standardization (0.631/0.772/0.938). The one-sided 95% Clopper-Pearson upper bound on 0/40 is 0.072. A separate drift example, high load for seven calls then low load for seven, makes the old block ratio 2.20 while the interleaved paired lower bound is 0.75. A correct range-sum that patches both clocks at import, a range-sum that memoizes by identity and by value, the same cache on dataset standardization, and an import-time print all failed the gate. Raw trials are in [`results/mlsys-coding-reference-speed-false-pass-v1/summary.json`](../results/mlsys-coding-reference-speed-false-pass-v1/summary.json).

Correctness baselines failed on the intended defects (overflow, detached labels, warmup off-by-one, short-window scaling, missing aging, L1 norm, and a causal mask that drops the diagonal).

## Why no model baseline was run

The no-update arm in the evaluation protocol is `not_run`.

- The isolated repository grader mounts the task directory into the same container that executes candidate code. Hidden files outside the candidate workspace are still readable there. That limitation is stated on the open trusted-scoring branch and was not changed here.
- These tasks set `grader_visibility` to `hidden`, so that branch's public-grader opt-in refuses them.
- `TrustedJSONScorer` keeps expected JSON off the candidate mount, but it does not grade a repository patch against these hidden checks.
- The in-process evaluator is a trusted harness for this repository's own fixtures. It is not a sandbox for model-generated code.

No weights were loaded and no agent process was started. There is no model success rate to report.

## Prepared comparison

[`protocols/mlsys_coding_policy_comparison_v1.json`](../protocols/mlsys_coding_policy_comparison_v1.json) defines three arms and does not implement them.

- Held-out evaluation is identical: seeds 101, 211, and 307; one greedy sample per seed per held-out task; 8 turns; 256 new tokens per turn; 2048 new tokens per attempt; 600 seconds; 8 GiB RSS; CPU only; no network and no paid API.
- SFT and GRPO share a train-split budget of 20 updates and 20,000 tokens. SFT is behavior cloning on the train-split oracle patches. GRPO uses on-policy groups of 4 and the shaped protocol reward. No-update takes zero gradient steps. Dummy updates are not used to equalize training compute.
- Primary contrasts are SFT minus no-update and GRPO minus no-update, paired by seed. Acceptance requires every seed to improve and the 2.5 percentile of a 10,000-resample seed bootstrap of the mean paired difference to be positive. With three seeds that interval is wide. A point estimate is not acceptance.
- The default command `scripts/run_mlsys_coding_policy_comparison_v1.py` still exits 3. It does not load a model or grade a real patch. A synthetic train-split dry-run exists behind `--synthetic-train-dry-run`. It uses a Bernoulli stand-in, calls the existing GRPO loss and trajectory admission, and is not a capability result. Real grading stays refused: the live canary read of the current repository-grader mount plan succeeds from the candidate side.

Under temperature 0 the three v1 seeds are one greedy draw. Read as three independent seeds, the v1 rule has power 0.314 on an absolute +0.25 success lift (0.05 to 0.30) and power 0.068 on a +0.10 lift. The greedy reading has power 0.680 on the +0.25 lift. None of those reach 0.80. A generation sketch at the retained 12.29 tokens/s from a different CPU task puts one v1 arm's full 2048-token cap near 2000 seconds of generation, against a 7200-second wall cap. That rate is not a measurement of this pack. The retained 0.5B CPU training peak of 11924.6 MiB is above the 8 GiB cap.

## Draft comparison v2

[`protocols/mlsys_coding_policy_comparison_v2.json`](../protocols/mlsys_coding_policy_comparison_v2.json) is a draft for review. It is not authorized. Held-out evaluation was not run.

Held-out membership is not frozen. The current v1 held-out pool has four tasks, and the draft target is 15-20, so the set waits on a 20-30 task expansion. The selected ids will be frozen by hash before any training. `mlsys-v1-causal-attention` stays in the v1 split and is excluded from that future set: it is a masked softmax, and the train task `mlsys-v1-row-log-softmax` is the same stable-softmax repair. The three utilities under `benchmarks/mlsys_coding_v2` are unassigned pool candidates, not a held-out set. Train membership is unchanged.

The primary test is a task-level paired comparison of GRPO against no-update and of SFT against no-update. Each contrast is a one-sided paired t-test at alpha 0.05 on the per-task mean success differences, with a Holm correction across the two contrasts. A task-clustered bootstrap interval is reported and is not an acceptance gate. `benchmarks/mlsys_coding_v2/power.py` is the simulation: task-level difficulty heterogeneity, held-out mean success 35.4%, three training seeds, and one one-sided test. At 80% power, 10 tasks give 15.5 points with 8 samples or 8.5 points with 32; 20 tasks give 12 points or 6.8; 30 tasks give 8.5 or 6.8. The 15-task rows are 12.0 and 8.5. The target after the pool exists is 15-20 tasks at 32 samples, about 8 points. Those figures are single-test effects. Holm was not recomputed and is more conservative. The model is invalid near 0% success, so the pass@8 train gate comes first. v1's own rule still needs about a 0.491 absolute lift at a 0.5 baseline for 80% power.

The draft also accepts three action formats: a JSON tool call, a fenced whole file, and a unified diff. The training screen is pass@8 at temperature 1 on the five train tasks: at least two tasks have a correct sample, and at least one group of eight has nonzero reward variance. `format_failure_rate` is a separate metric. A reported Qwen2.5-0.5B-Instruct screen scored 0/5 greedy on the train tasks because every attempt failed the tool-call format. This draft does not re-run that screen.

## What remains open

- Hidden-test confidentiality from the candidate execution namespace. Until that is shown, this pack must not be used for untrusted model patches.
- A no-update feasibility run on the train split only, after that clearance, under the frozen decoding budget.
- An authorized no-update versus SFT versus GRPO run. The synthetic train-split dry-run does not substitute for it. Draft v2 is not an authorization to run held-out. Its 15-20 task held-out set is not chosen yet; it waits on a 20-30 task pool and a hash freeze before training.
- The SFT and GRPO arms on a real model. The synthetic dry-run calls the existing loss and does not train a language model.
- Historical upstream commits. This pack does not replace that evidence level.
- `scripts/run_local_agent.py` reads the older historical task schema (`source.files`). Pointing it at a task root would expose `evaluator/`. A future runner has to materialize `subject/` only.
