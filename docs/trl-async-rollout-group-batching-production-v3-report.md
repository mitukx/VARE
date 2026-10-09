# TRL AsyncGRPO whole-rollout batching audit — 2026-10-09

## Research question

After AsyncGRPO scoring preserves a rollout ID, do the production fixed-count and token-budget batchers keep every fork from that rollout in one microbatch, without dropping rows or violating rank and capacity constraints?

The pinned source is TRL `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`. Its `_score_group()`/queue path and the batchers are separate boundaries: the local queue candidate groups admission, but emits sibling dictionaries one at a time to the batchers.

## Prior art and novelty

TRL issue [#7206](https://github.com/huggingface/trl/issues/7206) already identifies per-sample staleness drops that can partially consume a forked rollout and asks for complete-rollout accounting before token-budget packing. No novelty claim is made. This audit adds a pinned-source batch-boundary reproduction and tests whether a local patch can preserve group membership under the declared batching constraints.

## Frozen evidence

### Feasibility oracle v2

The standard-library exact enumerator evaluated four fixed-count and six token-budget fixtures. All ten matched their frozen dispositions. A separate same-host read-only audit verified protocol/script hashes, reran the command with byte-identical output, and checked row accounting, contiguous group placement, nonempty rank rows, and capacity assertions. This establishes feasibility only for those ten tiny reference cases.

The v1 protocol is retained as a failed fixture: FC2 encoded the intended rollout sizes as sample lengths. It produced no feasibility conclusion. Production-batcher v1 remains an environment/setup failure: pytest collection stopped because `transformers` was unavailable; no test body ran. Production protocol v2 did run on the base only, but its short finite fixture left tails buffered (`row_accounting_exact=false`) while `rollouts_unsplit=true`; it did not establish the intended split, and no v2 candidate run occurred. Protocol v3 expanded the stream to force the sibling into a second emitted microbatch.

### Pinned production definitions, protocol v3

To avoid installing unrelated ML dependencies, the frozen source-level runner AST-extracts the three production definitions under test (`_balance_by_squared_length`, `FixedCountBatcher`, `TokenBudgetBatcher`) from each checkout and executes those definitions with only the dataset base class and logger stubbed. It is not a full TRL import or Trainer test.

The two frozen boundary fixtures produced:

| Checkout | Fixed-count: rollout spans microbatches | Token-budget: rollout spans microbatches | All fixture rows accounted for |
| --- | --- | --- | --- |
| Pinned base | Yes | Yes | No; finite fixture tails remained buffered |
| Local candidate | No | No | Yes |

The base outputs specifically show rollout B in two emitted microbatches for both modes. Its incomplete tail is a finite-stream artifact and is not claimed to prove production stream loss. On these fixtures the local candidate preserved all rows and group boundaries. The candidate emitted three fixed-count batches and two token-budget batches; fixed-count batches were underfilled in some cases, which is a material compatibility question.

### Adversarial candidate falsification

An independent read-only audit and a separate frozen rerun found:

1. **Greedy false negative:** with 2 ranks, a 6-token row budget, and one five-sequence rollout of lengths `[3, 3, 2, 2, 2]`, exact packing exists (`[3,3]` and `[2,2,2]`), but the candidate raises `RuntimeError: whole-rollout prefix exceeds the token budget`.
2. **Oversized fixed-count rollout:** with 2 ranks and a fixed capacity of 4, a valid rollout of 5 sequences raises. The current candidate does not define a whole-rollout drop/metrics policy.
3. **Unpartitionable contiguous prefix:** group sizes `[1,4]`, 2 ranks, capacity 4 cannot be partitioned contiguously into batches with at least 2 and at most 4 rows while keeping groups whole. The candidate raises, but does not specify whether to drop a complete group or how to report it.

The second and third cases expose contract boundaries; they do not prove that a correct implementation must train every sampled group. A safe upstream contract could reject complete unbatchable groups with explicit accounting. The greedy false negative is a candidate algorithm defect because a valid packing exists.

## Decision

**STOP the current local patch as a complete trainer fix and do not submit it upstream.** The baseline batch-boundary split is reproducible and relevant to an already-open upstream issue, but the local candidate only passes two fixtures. Its token packer rejects an exactly feasible rollout, and fixed-count overflow/infeasible-prefix behavior is unresolved. Underfilled fixed-count batches may also change training-step sample weighting.

The next step is a narrow contract decision for *whole-group rejection and accounting under infeasible capacity*, followed by a new frozen candidate gate only if it can preserve fixed-count semantics and pass the exact-fit counterexample. Do not claim a policy-gradient or capability effect from this work.

## Limitations

- No full TRL dependency import, upstream test suite, `Trainer.train()`, optimizer update, distributed collective, epoch-stop/prefetch trace, throughput or memory measurement was run. The host lacked `transformers`; the source-level runner intentionally tested only extracted production definitions.
- The queue/scorer candidate and batcher candidate are local, uncommitted TRL changes. No upstream patch, maintainer response, or independent human reproduction exists.
- The result is not a model capability or task-success improvement. It is a narrow trainer dataflow reproduction plus a rejected local patch.

## Artifacts

- [Decision record](next-study-decision-2026-10-09.md)
- [Finite feasibility protocol v2](../protocols/trl_async_rollout_group_batching_feasibility_v2.lock.json) and [raw run](../results/trl-async-rollout-group-batching-feasibility-v2/attempt-1/)
- [Production boundary protocol v3](../protocols/trl_async_rollout_group_batching_production_v3.lock.json), [superseded v2 base observation](../results/trl-async-rollout-group-batching-production-v2/base/), and [v3 base/candidate outputs](../results/trl-async-rollout-group-batching-production-v3/)
- [Candidate falsification protocol](../protocols/trl_async_rollout_group_batching_candidate_falsification_v1.lock.json) and [raw counterexamples](../results/trl-async-rollout-group-batching-candidate-falsification-v1/run-1/)
