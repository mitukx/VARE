# Fresh Development Coding-Agent Diagnostic v1

## Executive result

**Coding capability: NO-GO under the frozen feasibility rule.** The offline Qwen3-4B attempt produced no source edit on either of the two fresh Development task families. The independent pinned graders then found the known target failures in each unchanged candidate: Transformers 4/6 cases passed (the four boundary targets failed); Ray 2/4 passed (both throughput controls passed, both wall-time/Unix-anchor targets failed). There were zero independently verified repairs.

**RL update: NO-GO.** There is no successful trajectory or nondegenerate verified reward signal to train on. No SFT, optimizer step, GRPO/RL update, or Held-Out access occurred.

## Frozen design and execution

- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`.
- Runtime: isolated Python 3.12.12 / MLX-LM 0.32.2; import preflight pinned MLX 0.32.4. Local Apple Metal, offline, greedy, one generation per task, no public tests, 2,048 generated-token cap, 4,096-token context cap, and one independent offline grade per task.
- The first Transformers launch failed at the real-Git-workspace guard before prompt encoding or generation. Addendum 1 preserved it as a zero-token infrastructure failure and bound replacement workspaces. That launch did not consume the one allowed model attempt.
- Both replacement workspaces passed the pinned base revision, allowed-file hash, clean status, and empty-diff checks. Resource gates passed immediately before both actual model loads.
- Each grade used the frozen image and selectors once, with no network and no feedback to the model. Exact executed argv, attempt records, raw logs, and hashes are retained alongside this report.

## Results by task

| Task family | Agent output | Independent result | Interpretation |
|---|---|---|---|
| Transformers #49418, quantized-cache residual boundary | 126 generated tokens; 4 tool turns; `context_budget_exhausted`; no edit; empty diff | 4 failed, 2 passed, 0 skipped | All four residual-boundary targets failed; below-capacity and single-token controls passed. |
| Ray #66694, multi-node operator wall time | 399 generated tokens; 5 tool turns; `context_budget_exhausted`; no edit; empty diff | 2 failed, 2 passed, 0 skipped | The wall-time target and Unix-anchor target failed; throughput controls passed. |

Both runs reached actual generation and consumed their one task slot. The first failed Transformers workspace launch remains separately classified as pre-inference infrastructure. The model did not receive grader feedback.

## Failure analysis

The largest common failure stage was **agent context/tool-use exhaustion before proposing an edit**, not a grader transport failure or an invalid candidate patch. Both attempts stopped as the rendered interaction history reached the frozen 4,096-token context limit. In Transformers, the model searched for `QuantizedCache`, then twice requested the same source range around the public wrapper instead of locating the internal quantized-layer update. In Ray, it inspected relevant portions of both allowlisted files, then emitted a no-tool response and stopped on context exhaustion. These traces suggest weak source localization and insufficient usable context under this prompt/controller configuration; two cases cannot establish a general cause.

The pinned graders behaved as expected: unchanged candidates reproduced the baseline target failures while preserving control passes. The grades are valid candidate outcomes, not successful repairs. No grader output was shown to the model.

## Scientific judgment

This is a bounded feasibility result for one cached 4B model, one greedy attempt per distinct task family, and a 4K context cap. It falsifies the frozen hypothesis that this configuration would produce at least one independently accepted repair on these two tasks. It does not prove that the model cannot code, that a larger-context or stronger model would fail, or that task families are absent from pretraining. Pretraining exposure remains **UNKNOWN**. No learning improvement, RL readiness, statistical generalization, or algorithmic novelty is supported.

The original feasibility decision rule requires at least one model-generated nonempty allowed-source patch passing every frozen correctness case. The result is 0/2, so capability feasibility remains NO-GO. The prescribed RL-readiness rule also remains NO-GO.

## Next executable experiment

First implement and unit-test one justified controller change that preserves the same task contract while avoiding repeated irrelevant source reads and the 4K history dead-end. Inspect which low-risk option is supported by current runner behavior (bounded context compaction or a larger context with a frozen model limit); change one factor only. Then select and source-qualify a new unexposed Development family, complete a family-overlap review, and freeze a new one-attempt protocol before inference. Do not rerun Transformers #49418 or Ray #66694, and do not allocate MLX #4778 until its possible broad Train-family overlap is explicitly resolved or accepted in a frozen protocol.
