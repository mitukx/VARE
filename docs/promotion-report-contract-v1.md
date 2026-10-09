# Promotion evaluation report contract v1

## Finding

`PromotionGate` trusted `EvaluationReport.n` for the minimum evaluation-size check, while the paired bootstrap used the independently supplied `metadata["per_task_scores"]`. It did not check that those values described the same evaluation. A backend could declare 64 evaluated examples while retaining one paired score; with `min_paired_examples=1`, the gate accepted a candidate based on that single row.

This is a promotion correctness defect in VARE's candidate-selection path. It can allow an under-supported candidate to replace the incumbent when an evaluator adapter reports inconsistent counts. The reproduced input uses synthetic reports and establishes the gate behavior, not the frequency of such adapter mistakes in deployed runs.

## Contract and correction

When per-task scores are present, they are the retained evidence for the primary metric. The report is valid only when:

1. The score map is nonempty and contains exactly `n` rows.
2. Every score is finite and in `[0, 1]`.
3. The arithmetic mean of those scores agrees with `primary` to absolute and relative tolerance `1e-9`.

`PromotionGate` now fails closed with `invalid_evaluation_metrics` if any condition fails. Evaluators without per-task scores keep the existing aggregate-only path.

## Reproduction

Run:

```bash
.venv/bin/python reproducers/promotion_report_contract_v1.py --output results/promotion-report-contract-v1/reproduction.json
.venv/bin/python -m pytest -q tests/test_promotion_report_contract.py tests/test_promotion.py tests/test_promotion_nonfinite.py tests/test_promotion_paired.py
```

The reproducer loads the actual pre-fix `PromotionGate` source from commit `2572ac7`, executes it on the same input as the current implementation, and compares both with a small independent standard-library oracle. Retained output is [`reproduction.json`](../results/promotion-report-contract-v1/reproduction.json).

| Case | Before fix | After fix / oracle |
|---|---:|---:|
| Declared 64 rows, retained 1; candidate gain 0.20 | Promoted, paired `n=1` | Rejected as invalid; oracle rejects |
| Complete, consistent 64-row evidence | Promoted | Still promoted; oracle accepts |
| Full score map whose mean disagrees with `primary` | Could pass the separate count gate | Rejected as invalid; oracle rejects |

The adversarial report covered 1/64 (1.5625%) of its declared examples, yet the pre-fix gate accepted it. A regression test first failed against the original implementation on the under-supported case, then passed after the fix. Four focused promotion suites pass (22 tests).

## StrategyQA failure diagnosis

The frozen StrategyQA/Qwen CPU screen was inspected read-only. Among 200 stored outputs, 181 started with `Yes`, 17 with `No`, and 2 with another form; zero included the literal final-answer marker required by the frozen parser. The model emitted direct yes/no-looking answers but did not follow the requested response format. The parser matched its frozen specification, so this is a format-compliance mismatch, not evidence of zero semantic accuracy and not a parser coding error. The strict-format screen did not pass; because no row parsed, semantic accuracy on parseable outputs is undefined. The consumed cohort was not rescored or reused for a performance claim. Details are in the [frozen feasibility report](cpu-strategyqa-grpo-shift-feasibility-v2-report.md).

## Limits

- The reproducer validates VARE's report contract with constructed `EvaluationReport` objects; it does not show that an existing production evaluator emitted this malformed report.
- It does not establish a model-quality change, held-out capability gain, or novel RL method.
- Aggregate-only evaluation reports remain supported and cannot be checked against per-example records they do not provide.
- The StrategyQA cohort remains retired; the diagnosis does not authorize changing its prompt or parser and rescoring those same examples.
