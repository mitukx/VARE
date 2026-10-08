# Next study decision — 2026-10-09

## Decision

Prioritize post-training systems correctness and reviewability. The immediate change is to preserve complete comparison groups when `rollout_count` is not divisible by `samples_per_task`. That regression is fixed and recorded. A one-command, clean-checkout reproduction packet for the existing synthetic preference robustness study is also in place. Defer another language-model learner experiment until a distinct base/task pairing clears a frozen, no-update success and resource screen.

## Evidence for the change

`CapabilityLoop.run_round` already computed the number of task groups with a ceiling, but stopped adding rollouts once it reached the requested count. With four samples per task and a rollout target of six, this produced a group of four and a partial group of two. The partial group could reach group-relative training, changing the comparison population used by GRPO/RLOO. Existing grouped-replay coverage exercised only a divisible count and did not catch the gap.

The engine now treats `rollout_count` as a target budget when group size is greater than one: it runs enough full groups to meet or exceed the target. A target of six with group size four therefore means eight rollouts. Nonpositive targets fail explicitly. The focused regression covers generated rollouts and the batch passed to training.

See the [rollout-group integrity record](rollout-group-integrity-v1-report.md) for the exact contract and test result.

## Why this is the current priority

The recent generated code-repair feasibility screen completed with 0/32 successes and no valid authorized tool calls; its model/task/tool pairing is retired. Other recent base-policy screens and policy-update studies also failed their frozen task-success gates, while the positive model-level results remain narrow forced-choice or score-calibration results. Another nearby prompt, task, or learning-rate variant would not answer the central missing question.

The group-size defect is instead a correctness issue in VARE's own post-training path. It has a direct counterexample, a small CPU-only fix, and a regression that protects the group-relative objective's input contract. This is a bounded systems contribution; it does not demonstrate policy improvement or model capability.

The second step is to make existing evidence easier to inspect. Synthetic preference robustness v1 already has retained raw designs and labels, a separately implemented learner/auditor, a tamper check, several Python-version runs, and a clean-clone replay on the author's machine. Those checks are not outside reproduction. [`scripts/reproduce_synthetic_preference_robustness.py`](../scripts/reproduce_synthetic_preference_robustness.py) now runs the frozen protocol, original audit, and separate verifier with one command and writes a compact audit record. The generated bundle uses the current Python runtime's design stream; it is a fresh v1 run, not a bit-for-bit reconstruction of the historical run. This lowers review cost but does not count as external reproduction.

The next evidence milestone is for a technically independent person to run the command and inspect the retained bundle. Do not describe author-run or CI execution as outside human reproduction.

## Deferred work and advancement gate

Do not start another model-learning run until a different task/model pair is frozen and its base policy demonstrates a nontrivial, independently graded success rate under the available CPU and memory limits. If it passes, preregister a task-success comparison with matched SFT, one RL method, multiple seeds, fresh development tasks, and an untouched confirmation cohort. Training reward or preference NLL alone is not an advancement criterion.

Do not reuse opened HH, GSM8K, BoolQ, generated arithmetic, SNLI, or code-repair cohorts. Keep every non-pass visible. No paid API, cloud compute, or GPU is required for the selected systems and reproduction work.

## Alternatives reviewed

| Candidate | Decision | Reason |
| --- | --- | --- |
| Another small-model learner study | Defer | Current evidence has no viable new base/task pair; recent feasibility and advancement gates failed. |
| Generalized system feature work | Reject absent a counterexample | Additional machinery without a measured correctness gap would add surface area without evidence. |
| Fix incomplete rollout groups | **Immediate** | Concrete defect in group-relative training inputs; deterministic CPU regression; directly relevant to trainer correctness. |
| Clean-checkout reproduction packet | **Next** | Makes the strongest no-cost retained study easier to audit; remains author-run until an outside person reproduces it. |

This decision supersedes the earlier Math-first and generated-repair feasibility proposals in this file's history and does not change any frozen experiment protocol or result.
