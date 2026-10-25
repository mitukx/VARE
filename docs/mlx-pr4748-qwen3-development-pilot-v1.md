# MLX #4748: certified Development task, no Qwen3 repair

**Decision: the task-certification pilot passed; the one-attempt model feasibility gate did not. Model learning, held-out evaluation, and model-coupled asynchronous comparison remain NO-GO.**

## What was tested

MLX PR [#4748](https://github.com/ml-explore/mlx/pull/4748), the `vmap(tri_inv)` upper/lower flag regression, was qualified as a new **Development-only** task. The source was pinned to baseline `c6abf0dbb4fc141fad739b32ae486c06728e49e5`; the public fix is `ae765377fbb9f0c238e2f4f6981def1fc01a75bb`. The candidate model received the task statement and the allowlisted `mlx/primitives.cpp` source through VARE tools. Its system prompt stated that a public test existed in the workspace, but the test contents were not in the model-visible allowlist and no test-running tool was available. It received neither the fix nor the independent oracle. The operator had inspected the public fix and test during qualification; external model pretraining exposure is unknown.

The frozen qualification order was baseline, predeclared wrong-upper mutant, then known fix. A separately written numerical oracle checked direct and vmapped triangular inverses at two mapping axes; the pinned upstream regression test was run outside the candidate workspace. The three source states were independently reconstructed and graded on CPU with network denied.

| Qualification state | Independent oracle | Upstream regression test | Interpretation |
| --- | ---: | ---: | --- |
| Baseline | 2/6 pass; all four vmap cases fail | Fail | Expected broken control |
| Wrong-upper mutant | 2/6 pass; all four vmap cases fail | Fail | Incorrect patch rejected |
| Exact upstream fix | 6/6 pass | Pass | Known fix accepted |

The independent reconstruction audit passed every check. The baseline build took 162.741 s; mutant and fixed builds took 10.741 s and 10.753 s. Maximum sampled process-tree RSS was 2,317,811,712 bytes. Minimum available memory was 12,914,786,304 bytes and minimum free swap was 456,851,456 bytes. Sampled swap did not grow beyond global preflight. macOS cgroup limits were unavailable and are not claimed. The full source tar is omitted from Git under the repository's 1 MiB tracked-blob policy; its exact 12,308,480-byte artifact is deterministically reconstructed from the pinned baseline Git commit during audit.

## Qwen3-4B attempt

One greedy attempt was frozen before inference with the cached `mlx-community/Qwen3-4B-Instruct-2507-4bit` snapshot at revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The checkpoint was not downloaded. The attempt used 8 turns, 384 generated tokens per turn, 3,072 total generated tokens, an 8,192-token context cap, and a 600-second wall limit. The lease-bound interface used a one-time hash-bound source-range lease; incomplete calls were neither repaired nor retried.

| Measure | Result |
| --- | ---: |
| Complete, schema-valid tool calls | 8/8 |
| Tool actions | 1 `list_files`, 7 `read_lines` |
| `search_text` calls (available to the model) | 0 |
| Generated tokens | 279 |
| Source edits applied / nonempty diff | 0 / 0 |
| Independent grader invocations | 0 |
| Training updates / held-out evaluations | 0 / 0 |
| Stop reason | 8-turn budget exhausted |

The model read consecutive 100-line ranges from the beginning of a 6,450-line C++ file and did not search for the relevant symbol. The trace contains no truncated, malformed, or policy-rejected call. This separates the observed failure from structured-output transport: the interface delivered every call, but this sample did not reach an edit or evaluator. It is a **single-sample inspection-strategy feasibility failure**, not evidence that Qwen3 cannot solve the task or a general coding-capability estimate. No prompt, budget, or interface retuning was performed on this consumed task.

The outer attempt took 111.153 s (108.75 s in the agent record). Peak sampled process-tree RSS was 2,580,807,680 bytes; MLX peak memory was 4,164,598,836 bytes. Minimum available memory was 8,970,829,824 bytes, minimum free swap 540,737,536 bytes, and sampled swap growth was zero. Network was denied for the model process. cgroup memory was unavailable.

Five earlier model-protocol launcher/preflight versions failed before generation and are retained with their dispositions. They are setup failures, not five additional model attempts. The v6 preflight passed, followed by the sole generation attempt.

## Interface and systems evidence

The existing tokenizer-only calibration compares four fixed edit fixtures. Under the pinned Qwen tokenizer, lease-bound edits used 406 serialized tool-output tokens in aggregate versus 508 for legacy structured replacement (20.08% fewer); the largest edit action was 73 versus 132 tokens (44.70% fewer). These are fixture token counts, not generated-model outcomes. The Qwen attempt above confirms complete tool-call parsing in one sample but does not show that the smaller edit payload improves task success.

Separately, the retained systems-only comparison ran a deterministic known-fix controller on two MLX-LM Development tasks through both interfaces and both synchronous/async modes: 8/8 cells applied the exact pinned fix and passed the trusted grader. One SIGKILL after durable edit/journal commit recovered and graded the exact state without a duplicate edit. The single four-cell async workload took 2.9363 s versus 5.3363 s synchronous (descriptive ratio 1.817); this is not a replicated speedup and is not model-generated work. Its independent audit retains a warning for duplicated terminal completion markers in each historical cell, despite valid journal chains and exactly one committed edit.

The focused interface suite passes **53 tests** covering strict parsing, truncation, policy rejection, stale/replayed hash-bound leases, path/symlink safety, atomic edits, a full mocked edit to a fixture oracle, and related grader gates. These tests establish controller behavior, not model quality.

## Research gates

- **Task certification:** one new Development family qualified; baseline and wrong mutant failed, exact fix passed. No held-out eligibility was added.
- **Agent interface:** complete transport calls in one Qwen sample; no patch reached grading. The model-task feasibility pilot is NO-GO under its predeclared stop rule.
- **Learning:** no positive trajectory from this task, no optimizer update, no SFT/GRPO, and no student comparison. NO-GO.
- **Held-out:** this Development task adds zero held-out families. The existing 15-task gate and prior NO-GO remain unchanged.
- **Asynchronous execution:** only the separate deterministic known-fix systems evidence above is available; no model-coupled async comparison was run.
- **RSI intervention:** not admitted; no intervention was selected or evaluated.

### Strongest new finding

On a newly source/grader-certified MLX Development task, the frozen Qwen3 run produced eight complete valid tool calls with a hash-bound edit interface, yet spent all eight turns on sequential source inspection and submitted no patch. This proves that the current one-attempt pipeline can retain and independently classify a real-model failure without confusing it with truncation or grader failure. It does **not** establish coding capability, inability, learning, held-out performance, or a benefit from the interface.

## Reproduction and retained evidence

The complete compact evidence package is in [`results/mlx-pr4748-qwen3-development-pilot-v1`](../results/mlx-pr4748-qwen3-development-pilot-v1/). It includes the frozen protocols and locks, all pre-generation setup failures, the raw qualification and model traces, resource samples, independent audit outputs, and a checksum index. The candidate source and model weights are identified by immutable Git/model revisions and hashes rather than copied into the repository. The original qualification archive hash and exact reconstruction command are recorded in the package README.

The task qualification is **not** a held-out result. Public PR history and operator inspection mean no claim of task novelty to the model is made. Local exposure scans cannot establish absence from pretraining.

## Repository state

The source interface resides in Draft PR [#131](https://github.com/mitukx/VARE/pull/131), parent head `1046507ed3b9472685f03e46c284176fee354448`. Its ordinary CI checks passed at that head, but `actual-optimizer-and-recovery` was **cancelled before the GRPO portion**. This report does not treat that check as passing. PR #131 remains unmerged and requires its existing review/dependency gates. A follow-up evidence PR is kept separate and Draft; neither PR is merged here.
