# TRL AsyncGRPO whole-rollout batching contract v6 — 2026-10-09

## Question

Can the local complete-or-drop candidate preserve rollout groups in fixed-count and token-budget microbatches under a frozen set of small boundary cases, with every dropped row accounted for?

This is a follow-up to the rejected v3 candidate. It tests a source-level candidate extracted from pinned TRL revision `ed8cc2f4337fb9b7b1429dbd31009ad3577cd5b98`; it does not import the full trainer.

## Frozen contract and execution

Protocol v6 was frozen before execution after two superseded runner versions exposed mistakes in the boundary budget: v4 packed with budget 6, and v5 packed with 10 but inspected against 6. Those raw runs are retained and are not confirmatory evidence. V6 passes each case's budget to both the production packer and the invariant checker.

The locked protocol defines eight fixtures: four fixed-count and four token-budget cases. The runner AST-extracts `_balance_by_squared_length`, `FixedCountBatcher`, and `TokenBudgetBatcher`. It does not exercise trainer construction, logging reduction, the optimizer, distributed execution, or model outputs.

Reproduce from either pinned checkout with:

```bash
python /absolute/path/to/VARE/reproducers/trl_async_rollout_group_batching_contract_v6.py
```

## Results

The pristine pinned baseline failed all eight fixtures. It split same-rollout members across batches in the fixed boundary and token boundary cases, and the finite fixtures showed missing or partial group rows. This is a bounded source-path observation; finite-stream tail behavior is not evidence of loss in an unbounded production stream.

The local candidate passed all eight fixture predicates. It retained all 8/8 token-boundary rows and the exact-fit five-row token group. It dropped complete groups with raw metric lists whose sums matched the fixture omissions. In fixed boundary, it retained 4/9 rows and dropped 5/9. Fixed exact control retained all 4/4; the oversized and infeasible-prefix controls retained the intended 4-row groups.

## Independent audit and correction to the claim

A read-only independent audit found that the fixture-level pass overstates two properties:

1. **Production drop accounting is not exact.** The test sums raw values in metric lists. Pinned TRL's `_reduce_metric` treats a metric as a counter only if its name contains the word `total`; otherwise it reports a window mean. The candidate's `batch/dropped_whole_rollout_samples` and reason-specific `..._samples` keys therefore do not log exact dropped-row totals. The frozen v6 checker does not invoke `_reduce_metric`, so v6 does not pass the intended production accounting contract.
2. **Fixed-count retention is not established as efficient.** The candidate drops 5/9 rows on the boundary fixture. If group reordering is allowed, batches `A(3)+D(1)` and `B(2)+C(2)` retain 8/9 rows while satisfying exact capacity and whole-group integrity. If FIFO order is required, that constraint must be explicit and the candidate must be compared with the best FIFO schedule. V6 does not resolve that choice.

The token-budget candidate also has a 100,000-node DFS ceiling. The eight fixtures do not characterize false drops or runtime near that ceiling.

## Decision

**STOP this candidate as a complete trainer fix and do not submit it upstream.** V6 establishes that the candidate can satisfy the structural predicates on these eight fixtures, and that the pinned baseline violates group integrity in the boundary cases. It does not establish exact production drop accounting, optimal or acceptable sample retention, a policy-update effect, or a capability gain. This issue overlaps open TRL issue [#7206](https://github.com/huggingface/trl/issues/7206); novelty is not claimed.

Do not produce another heuristic patch variant until a new frozen comparison defines FIFO/reordering semantics, runs the real metric reducer, and compares accepted rows with a small exact oracle including search-limit adversaries. Only pursue full trainer integration if that check shows a meaningful benefit. Full trainer/optimizer work remains blocked by the missing `transformers` dependency in the local TRL checkout.

## Artifacts and limits

- Frozen [protocol](../protocols/trl_async_rollout_group_batching_contract_v6.lock.json), SHA-256-locked [reproducer](../reproducers/trl_async_rollout_group_batching_contract_v6.py), [source/hash manifest](../results/trl-async-rollout-group-batching-contract-v6/source-manifest.json), candidate source snapshot, and [base output](../results/trl-async-rollout-group-batching-contract-v6/base/) / [candidate output](../results/trl-async-rollout-group-batching-contract-v6/candidate/).
- Superseded v4 and v5 protocols and outputs are preserved as runner failures.
- Candidate changes exist only in the local pinned TRL checkout; no upstream PR was opened, no maintainer reviewed the change, and no full TRL test suite ran.
- Evidence class: finite source-level synthetic fixtures. No sample-efficiency, optimizer, downstream task-success, or model-capability result.
