# RVL evaluation-to-promotion integration contract v1

**Decision: integration contract established for the tested local path; stop expanding evaluation infrastructure.** This is implementation correctness evidence only. It is not a production-incident rate, RL research result, or model capability result.

## Path and change

The exercised path is `CapabilityLoop._run_round` → `RVLGRPOHooks.train_candidate` → `RVLGRPOHooks.evaluate` → `EvaluationReport` → `PromotionGate.decide` → `RVLGRPOHooks.promote` or `discard`.

The review found that a rejected report is discarded correctly, but an evaluator exception escaped before the decision branch. `RVLGRPOHooks.evaluate` restored the live incumbent, while the candidate snapshot remained in `_states`. `CapabilityLoop` now discards the candidate if either evaluation or decision raises. If that cleanup itself fails, it raises an explicit unknown-policy-state error rather than hiding the rollback failure.

The single integration regression uses actual VARE loop, adapter, report, gate, and lifecycle methods with a deterministic CPU backend/trainer. The score oracle is independently asserted from the known backend state: incumbent weight 0 emits two wrong answers (0/2); candidate weight 1 emits two correct answers (2/2). Report mutations happen only after the actual adapter evaluates and constructs reports.

## Trust and fail-closed contract

- The configured evaluator implementation, its frozen task manifest, and its scoring function are the trust root. Returned generations must bind to the requested task ID and prompt; task IDs must be unique.
- `EvaluationReport` policy IDs must equal the requested incumbent/candidate IDs. The primary metric, slices, count and cost must be structurally valid and finite.
- When `per_task_scores` is supplied, it must be a nonempty string-keyed map of finite scores in `[0,1]`; its size must equal `n` and its mean must equal `primary` within `1e-9`. Paired reports must have identical task-ID sets.
- With paired confidence gating enabled, missing per-task evidence fails closed. With the gate disabled, aggregate-only backends remain supported; in that mode VARE can validate fields but cannot independently reconstruct an aggregate or establish its row count. Such a backend is trusted by configuration, not cryptographically authenticated.
- Duplicate task IDs, task/prompt-mismatched backend rows, wrong report policy IDs, malformed or incomplete per-task maps, mismatched paired IDs, missing paired evidence under the paired gate, and evaluation exceptions cannot promote a candidate. The engine discards a candidate on report rejection or evaluator/decision exceptions and preserves the exact incumbent trainer/optimizer snapshot in the tested adapter.

This contract detects malformed, inconsistent and incorrectly bound backend results. It does not prove that a configured grader's labels are truthful, that task manifests are uncontaminated, or that a malicious evaluator is honest.

## Reproduction

Run:

```bash
PYTHONPATH=src .venv/bin/python reproducers/rvl_eval_promotion_e2e_v1.py
.venv/bin/pytest -q tests/test_rvl_evaluation_promotion_e2e.py tests/test_rvl_grpo_eval_identity.py tests/test_rvl_grpo_hooks.py tests/test_promotion.py
```

The historical control loads `PromotionGate` from pre-fix commit `2572ac7`; both arms use the same actual loop and RVL evaluation path. The injected fault leaves `n=2` and the aggregate intact but retains one of the two per-task scores. Raw output is [retained here](../results/rvl-eval-promotion-e2e-v1.json).

| Case | Decision | Active state afterward |
| --- | --- | --- |
| Historical gate, incomplete map | accepted (false promotion) | candidate `policy-1`, weight/optimizer step `1/1` |
| Current gate, same incomplete map | rejected: `invalid_evaluation_metrics` | incumbent `policy-0`, weight/optimizer step `0/0`; candidate snapshot removed |
| Current gate, complete valid map | accepted (positive control) | candidate `policy-1`, weight/optimizer step `1/1`; old snapshot removed |
| Current gate, complete evidence but no task-success gain | rejected: `insufficient_primary_gain` | incumbent `policy-0`, weight/optimizer step `0/0` |
| Historical engine, task-misbound evaluation raises | exception propagated, incumbent runtime restored | incumbent remains active, but stale candidate snapshot `policy-1` leaks |
| Current engine, same evaluation exception | exception propagated, candidate discarded | incumbent active/runtime exact; only `policy-0` snapshot remains |

The same integration test also checks missing evidence, NaN evidence, paired task-ID mismatch, aggregate-only reports with paired gating both enabled and disabled, report-policy mismatch, duplicate input task IDs, and a backend generation bound to the wrong task. Rejections preserve the exact incumbent state; the actual evaluator exception path now removes its candidate snapshot. A valid report with no task-success gain is a negative control and is rejected. Aggregate-only mode promotes a valid control when configured without paired confidence gating, preserving valid backend compatibility. The reproducer runs the historical `CapabilityLoop` source from the pre-fix `a4e0c31` commit to retain the candidate-leak baseline.

Focused status: **17 passed** across the integration, RVL identity, RVL hook and promotion suites. This local focused run is not a full-suite or external replication claim.

## Scientific gate and minimum missing resources

Stop evaluation-infrastructure work here unless a new concrete defect appears. The next scientific gate is an independently graded, real-model policy update with a frozen no-update, matched SFT, and one RL baseline, multiple seeds, and untouched task-composition confirmation.

The current host is an ARM Mac with 32 GiB RAM and has cached small-model weights and public datasets, but repository decisions have retired the attempted model/task cohorts or reserved their evaluation data. Thus the immediate missing resource is **a fresh, unused task/base pairing with an executable ground-truth grader and disjoint train, development, and confirmation examples**. Before training, that pairing must also pass a no-update feasibility gate and a measured CPU/MPS runtime budget for the frozen multi-seed arms. No paid API or rented GPU is required by this report; a free accelerator would only be needed if the chosen pairing cannot meet the frozen budget locally. No currently recorded result establishes independent task-success improvement after a real policy update.
