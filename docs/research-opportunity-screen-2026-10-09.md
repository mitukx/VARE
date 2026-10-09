# Post-training research opportunity screen — 2026-10-09

## Decision

**No candidate in this bounded screen survives the novelty gate.** Do not implement or present a duplicate as an original result. This screen adds no model, optimizer, or capability evidence. It is not an exhaustive literature or source review.

The repository was clean on `main` at `90f794c`; the latest provenance repair and its documentation CI run passed. The screen followed `AGENTS.md`, the current gaps, the active study decisions, and the retained batcher and LoRA reports.

## Ranked hypotheses

| Rank | Hypothesis and potential impact | Novelty/feasibility check | Decision |
|---|---|---|---|
| 1 | AsyncGRPO can split one forked rollout across microbatches or drop its members, changing the group supplied to GRPO. High training-correctness impact; cheap CPU source-path checks are possible. | TRL issue [#7206](https://github.com/huggingface/trl/issues/7206) already identifies partial rollout admission and calls for preserving rollout identity. VARE reproduced the boundary in pinned production batchers. Its latest candidate was stopped: v6 did not establish production drop accounting, retained only 4/9 rows in one fixed-count case, and left ordering and search-bound behavior unresolved. | **Reject as original; keep prior evidence and stopped decision.** No new variant or run. |
| 2 | Repeated PEFT LoRA merge/unmerge during vLLM sync can mutate frozen base weights. High correctness impact and inexpensive CPU reproduction. | TRL issue [#7423](https://github.com/huggingface/trl/issues/7423) and open fix PR [#7427](https://github.com/huggingface/trl/pull/7427) contain the mechanism, reproducer, and proposed restoration. VARE already measured BF16 drift on a cached model layer and stopped for overlap with issue [#6688](https://github.com/huggingface/trl/issues/6688). | **Reject as rediscovery.** No duplicate patch or regression. |
| 3 | Additive score centering may stabilize GRPO under train–sampler mismatch. Direct post-training relevance; small categorical CPU checks are feasible, but the claimed training stability requires actual training evidence. | The mechanism and top-k estimator are published in [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807), and TRL issue [#7520](https://github.com/huggingface/trl/issues/7520) proposes the integration. The paper derives the centered update as `Cov_q(R, score)` and specifies its composition with importance weights. TRL maintainers have requested reward curves under a mismatch setting they can reproduce; a toy CPU result would not meet that evidence bar. | **Reject as original.** Do not reproduce known algebra as a contribution. |

## Evidence and limits

For candidate 1, the retained v6 fixture pass is bounded to eight AST-extracted source-level cases. Independent review found that the fixture sums raw metric lists, while production reduces the candidate's `..._samples` keys by mean; exact logged drop counts were therefore not demonstrated. The fixed-boundary candidate emitted 4 of 9 rows where group reordering could retain 8 of 9 under the written capacity constraints. No full trainer, optimizer, distributed, throughput, or task-success result was obtained. See the [v6 report](trl-async-rollout-group-batching-contract-v6-report.md).

For candidate 2, VARE's earlier CPU study found that one unsafe BF16 merge/unmerge cycle changed 53,597 of 802,816 q_proj elements; the finding was stopped because TRL issue #6688 already contained deeper repeated-cycle and full-sync evidence. The new issue/PR inspection confirms the focused report and a proposed fix remain public. See the [VARE report](trl-peft-bf16-lora-merge-roundtrip-v1-report.md).

For candidate 3, the paper's exact full-distribution identity removes the constant-reward drift by subtracting the expected score under the sampler, and its appendix already handles weighted scores and top-k tail approximation. Its large-scale experiments are GPU-based; no claim about their results was independently reproduced here.

No implementation was started because each candidate failed the prior-art gate before a new experiment could support an original claim. The next useful step is an outside technical reproduction/review of an existing VARE evidence packet, or a new candidate only after a distinct source path is found with no equivalent public issue, PR, paper, or VARE record. This does not establish a novel contribution, upstream adoption, or independently measured model improvement.
