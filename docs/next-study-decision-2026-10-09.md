# Next study decision — 2026-10-09

## Decision

Keep post-training systems correctness and reviewability as VARE's foundation, and make independently graded task success after an update the next flagship evidence target. The immediate replay and evaluation-recovery changes are implemented with CPU regression coverage; the synthetic preference robustness study has a one-command clean-checkout reproduction packet. Do not continue neighboring preference/NLL screens or expand VARE with unmeasured features.

## Evidence for the change

`CapabilityLoop.run_round` already computed the number of task groups with a ceiling, but stopped adding rollouts once it reached the requested count. With four samples per task and a rollout target of six, this produced a group of four and a partial group of two. The partial group could reach group-relative training, changing the comparison population used by GRPO/RLOO. Existing grouped-replay coverage exercised only a divisible count and did not catch the gap. A second review found that reuse of the same caller-supplied `round_index` could merge separate rollout cohorts in replay; group IDs now also include a deterministic loop-local run serial.

The engine now treats `rollout_count` as a target budget when group size is greater than one: it runs enough full groups to meet or exceed the target. A target of six with group size four therefore means eight rollouts. Each rollout carries its declared group size. With group preservation enabled, the engine submits complete groups atomically after lag filtering; replay evicts declared groups as units and rejects a new group if it cannot fit intact or meet the priority rule. Legacy per-item insertion remains available, and grouped sampling still fails closed on incomplete legacy groups. For capacity seven and groups of four, replay retains one complete group rather than a four-plus-three fragment. Nonpositive targets fail explicitly. Reusing a `round_index` no longer reuses a group ID within one loop. Regression coverage includes nonmultiple generation, atomic admission/eviction, invalid group rejection, and repeated round indices.

The RVL adapter also now restores the active policy after failed candidate evaluation, and reports unknown trainer state if recovery itself fails. A restore attempt invalidates the cached loaded-policy marker first, so a partial restore failure cannot make recovery skip the incumbent reload. The regression injects a candidate restore that mutates trainer state and then raises, and checks a forced incumbent restore. See the [rollout-group integrity record](rollout-group-integrity-v1-report.md) and [evaluation recovery record](rvl-grpo-evaluation-recovery-report.md).

## Why this is the current priority

The recent generated code-repair feasibility screen completed with 0/32 successes and no valid authorized tool calls; its model/task/tool pairing is retired. Other recent base-policy screens and policy-update studies also failed their frozen task-success gates, while the positive model-level results remain narrow forced-choice or score-calibration results. Another nearby prompt, task, or learning-rate variant would not answer the central missing question.

These defects were correctness issues in VARE's own post-training path. They have direct counterexamples and CPU-only regressions: group cardinality and identity are preserved through rollout/replay, and active-policy identity is restored after failed evaluation, including partial trainer restoration. These are bounded systems contributions; they do not demonstrate policy improvement or model capability.

The second step is to make existing evidence easier to inspect. Synthetic preference robustness v1 already has retained raw designs and labels, a separately implemented learner/auditor, a tamper check, several Python-version runs, and a clean-clone replay on the author's machine. Those checks are not outside reproduction. [`scripts/reproduce_synthetic_preference_robustness.py`](../scripts/reproduce_synthetic_preference_robustness.py) now runs the frozen protocol, original audit, and separate verifier with one command and writes a compact audit record. The generated bundle uses the current Python runtime's design stream; it is a fresh v1 run, not a bit-for-bit reconstruction of the historical run. This lowers review cost but does not count as external reproduction.

The next evidence milestone is for a technically independent person to run the command and inspect the retained bundle. Do not describe author-run or CI execution as outside human reproduction.

## Deferred work and advancement gate

The distinct task candidate remains a short, offline feasibility chain for a multi-turn stateful tool-use task using a locally cached model. Its initial runtime gate failed: an explicit CPU-stream tool-call/result/tool-call/final-answer smoke did not complete within a 10-minute cap on this machine (observed RSS stayed below 6 GiB). No task pack was frozen or scored. A separate faster-looking run had used MLX's GPU default stream and is invalid as CPU evidence. Do not freeze or run the 40–48 task screen with this current runtime. This does not show that the model cannot use tools; it shows that the current CPU runtime did not clear the practical time gate.

Revisit this candidate only if a new, explicitly CPU-pinned runtime can complete the same multi-turn smoke within the predeclared cap. Then freeze a base-only pack with distinct operation compositions, a state-transition oracle, task-family and tool-validity gates, and strict wall-clock and memory caps. Do not retune the prompt or task after opening its scored cohort. Only if base success is nontrivial across each task family should a separate CPU update-cost smoke check gradients, checkpoint/adapter save and reload, and post-reload tool behavior. Only if that also fits the no-cost CPU limit should a formal comparison be frozen: base, matched successful-trace SFT, one RL method, multiple seeds, and a fresh untouched confirmation cohort. Measure exact environment success as primary; report tool validity, unsafe or unnecessary state changes, regressions, policy drift, reward-success mismatch, wall time, and peak memory. Training reward or preference NLL alone is not an advancement criterion. If either feasibility gate fails, retire that model/task/runtime pairing and return the centerpiece to systems correctness and independent reproducibility.

Do not reuse opened HH, GSM8K, BoolQ, generated arithmetic, SNLI, or code-repair cohorts. Keep every non-pass visible. No paid API, cloud compute, or GPU is required for the selected systems and reproduction work.

## Alternatives reviewed

| Candidate | Decision | Reason |
| --- | --- | --- |
| Another small-model learner study | Defer | Current evidence has no viable new base/task pair; recent feasibility and advancement gates failed. |
| Generalized system feature work | Reject absent a counterexample | Additional machinery without a measured correctness gap would add surface area without evidence. |
| Fix rollout-group cardinality/identity and failed-evaluation recovery | **Completed** | Concrete defects in group-relative training inputs and shared trainer state; deterministic CPU regressions; directly relevant to trainer correctness. |
| Clean-checkout reproduction packet | **Ready for independent review** | One-command runner and two audits are available; the remaining step is an outside person's run/review, which author and CI runs do not satisfy. |
| Cached-model stateful tool-use feasibility chain | **Deferred at runtime gate** | The explicit CPU multi-turn smoke exceeded 10 minutes without completing; no task pack was frozen or scored. Revisit only with a faster verified CPU path. |
| Upstream RL trainer issue | **Opportunistic only** | Current scan found nearby issues already closed or covered by active pull requests. Select a future issue only after checking ownership and reproducing a real production-path failure. |

This decision supersedes the earlier Math-first and generated-repair feasibility proposals in this file's history and does not change any frozen experiment protocol or result.
