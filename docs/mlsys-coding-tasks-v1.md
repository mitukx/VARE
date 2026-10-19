# MLSys coding pack v1

**Evidence status: fixture calibration of an authored task pack.** No model was loaded. No agent process was started. The numbers below are properties of the frozen baseline and oracle files under the in-process evaluator. They are not coding-agent capability.

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

The partition is in `split.json`. It was fixed before any policy rollout. Held-out tasks are different defects, not resized copies of train tasks. The manifest SHA-256 is `18c5999f823f9a550a561a913023b87f1fc7bf5a86f481d319e4778212d97499`, stored in [`protocols/mlsys_coding_agent_eval_v1.json`](../protocols/mlsys_coding_agent_eval_v1.json).

## Reward

Correctness tasks succeed when the single hidden command exits 0 and integrity holds. `reward.decide` reads the serialized return code, timeout flag, and output-limit flag. `CommandResult.passed` is a property and is not in the result schema.

Performance tasks time a frozen slow reference and the candidate seven times in one process. The grade command prints one JSON object on stdout, which the existing command-result schema already retains. The protocol recomputes

```text
conservative_speedup = baseline_ci95_low / candidate_ci95_high
```

using the sample standard deviation (divisor `n - 1`) and the Student-t 95% critical value `2.446911851144167` at 6 degrees of freedom. Success requires correctness and `conservative_speedup >= 2`. When correctness passed, the shaped reward is `min(1, speedup / 4)`. The cap at 4 is twice the success threshold. It was not fit to a model outcome.

The grade command exits 0 only when that speedup threshold is met, so `EnvironmentResult.passed` matches protocol success for these fixtures. The shaped reward is still the training target defined for a later GRPO arm: `grader.score` is 0 for a correct but slow patch, while the shaped reward stays positive.

Candidate prints are redirected away from the grade stdout during timing.

## Calibration on this host

`python -m pytest tests/test_mlsys_coding_v1.py` passed: every unpatched subject failed the pre-registered gate, every oracle passed, public smoke checks passed on the unpatched subjects, and the candidate workspace did not contain the hidden files.

One earlier same-host execution of the performance grade commands, before the reward helper was pointed at serialized return codes, recorded:

| Task | Unpatched conservative speedup | Unpatched sample stdevs (reference, candidate), seconds | Oracle conservative speedup | Oracle sample stdevs (reference, candidate), seconds |
| --- | --- | --- | --- | --- |
| `mlsys-v1-range-sums` | 0.9990 | 0.01788, 0.001326 | 233.11 | 0.002491, 5.180e-5 |
| `mlsys-v1-dataset-standardize` | 0.9569 | 0.009462, 0.002678 | 669.16 | 0.005418, 1.198e-5 |

These wall-clock values are descriptive. The locked rule is the threshold of 2, not these floats. The subsequent test run confirmed the same pass/fail decisions and that both oracles also clear the reward cap of 4.

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
- `scripts/run_mlsys_coding_policy_comparison_v1.py` exits 3. It does not load a model, start a container, or apply an update.

## What remains open

- Hidden-test confidentiality from the candidate execution namespace. Until that is shown, this pack must not be used for untrusted model patches.
- A no-update feasibility run on the train split only, after that clearance, under the frozen decoding budget.
- The SFT and GRPO arms. They are specified and not implemented.
- Historical upstream commits. This pack does not replace that evidence level.
- `scripts/run_local_agent.py` reads the older historical task schema (`source.files`). Pointing it at a task root would expose `evaluator/`. A future runner has to materialize `subject/` only.
