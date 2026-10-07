# Promotion gate: non-finite metric validation

`PromotionGate` decides whether a candidate replaces an incumbent using primary scores, slice scores, costs, verifier disagreement, and optional paired task scores. Comparisons with IEEE-754 `NaN` are false, so threshold checks alone can let malformed evidence pass.

## Frozen task and baseline

The [regression task](../benchmarks/regressions/promotion_gate_metrics/TASK.md) and [protocol lock](../benchmarks/regressions/promotion_gate_metrics/protocol.lock.json) were committed before measuring the baseline. The evaluator checks a finite positive control and seven malformed cases, expects malformed cases to be rejected with `invalid_evaluation_metrics`, and checks that returned decision metrics remain finite.

At baseline revision `d4627bf9e486413f4d95b2cd7c4ca294f5d95091`, the gate accepted every malformed case: candidate primary `NaN`, candidate primary `+inf`, incumbent primary `NaN`, candidate slice `NaN`, candidate cost `NaN`, candidate verifier disagreement `NaN`, and a paired task score of `NaN`. The finite positive control was also accepted. The [raw baseline result](../results/promotion-gate-metrics-v1/baseline.grade.json) and [metadata](../results/promotion-gate-metrics-v1/baseline.json) bind this observation to the pre-fix source and evaluator hashes.

## Fix and result

The gate now validates both reports before threshold arithmetic. It rejects non-finite scalar and slice metrics, malformed paired-score mappings, and paired scores outside their declared `[0,1]` range. Rejection uses the explicit `invalid_evaluation_metrics` reason and finite decision fields. A finite positive control remains accepted.

The frozen evaluator rejects all seven malformed cases and accepts the finite control on fixed revision `7fcc897fa49024a014a15c34aed4d0cf1fd2ec94`. The full test suite passes (83 tests), including the new cases in [`test_promotion_nonfinite.py`](../tests/test_promotion_nonfinite.py). The local CPU runtime was Python 3.12.14 on macOS arm64. See the [fixed aggregate](../results/promotion-gate-metrics-v1/summary.json) and [hash manifest](../results/promotion-gate-metrics-v1/manifest.json).

## Limits and reproduction

This demonstrates one concrete fail-open defect and its regression coverage. It does not estimate the gate's general false-acceptance rate or validate the correctness, calibration, independence, or provenance of finite metrics supplied by evaluators. It measures no agent, model, learning, or production deployment behavior.

Run the frozen grader with Python 3.11 or newer against a workspace containing the candidate source:

```bash
python3.12 benchmarks/regressions/promotion_gate_metrics/evaluator/grade.py \
  --workspace . \
  --json-out /tmp/promotion-gate-grade.json
```

The task uses only the Python standard library and local source; no model weights, GPU, paid API, or external compute are required.
