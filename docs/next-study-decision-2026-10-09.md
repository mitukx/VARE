# Next study decision — 2026-10-09

## Decision

Prioritize post-training systems correctness and reviewability. The immediate changes are to preserve complete comparison groups through replay and to restore the active policy after evaluation failures. Both now have CPU regression coverage. A one-command, clean-checkout reproduction packet for the existing synthetic preference robustness study is also in place. Defer another language-model learner experiment until a distinct base/task pairing clears a frozen, no-update success and resource screen.

## Evidence for the change

`CapabilityLoop.run_round` already computed the number of task groups with a ceiling, but stopped adding rollouts once it reached the requested count. With four samples per task and a rollout target of six, this produced a group of four and a partial group of two. The partial group could reach group-relative training, changing the comparison population used by GRPO/RLOO. Existing grouped-replay coverage exercised only a divisible count and did not catch the gap. A second review found that reuse of the same caller-supplied `round_index` could merge separate rollout cohorts in replay; group IDs now also include a deterministic loop-local run serial.

The engine now treats `rollout_count` as a target budget when group size is greater than one: it runs enough full groups to meet or exceed the target. A target of six with group size four therefore means eight rollouts. Each rollout carries its declared group size; grouped replay excludes groups whose stored cardinality is incomplete after freshness filtering or per-item capacity eviction. Nonpositive targets fail explicitly. Reusing a `round_index` no longer reuses a group ID within one loop. Regression coverage includes nonmultiple generation, a 4+3 group split caused by capacity seven, and repeated round indices.

The RVL adapter also now restores the active policy after failed candidate evaluation, and reports unknown trainer state if recovery itself fails. A restore attempt invalidates the cached loaded-policy marker first, so a partial restore failure cannot make recovery skip the incumbent reload. The regression injects a candidate restore that mutates trainer state and then raises, and checks a forced incumbent restore. See the [rollout-group integrity record](rollout-group-integrity-v1-report.md) and [evaluation recovery record](rvl-grpo-evaluation-recovery-report.md).

## Why this is the current priority

The recent generated code-repair feasibility screen completed with 0/32 successes and no valid authorized tool calls; its model/task/tool pairing is retired. Other recent base-policy screens and policy-update studies also failed their frozen task-success gates, while the positive model-level results remain narrow forced-choice or score-calibration results. Another nearby prompt, task, or learning-rate variant would not answer the central missing question.

These defects were correctness issues in VARE's own post-training path. They have direct counterexamples and CPU-only regressions: group cardinality and identity are preserved through rollout/replay, and active-policy identity is restored after failed evaluation, including partial trainer restoration. These are bounded systems contributions; they do not demonstrate policy improvement or model capability.

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
| Fix rollout-group cardinality/identity and failed-evaluation recovery | **Completed** | Concrete defects in group-relative training inputs and shared trainer state; deterministic CPU regressions; directly relevant to trainer correctness. |
| Clean-checkout reproduction packet | **Ready for independent review** | One-command runner and two audits are available; the remaining step is an outside person's run/review, which author and CI runs do not satisfy. |

This decision supersedes the earlier Math-first and generated-repair feasibility proposals in this file's history and does not change any frozen experiment protocol or result.
