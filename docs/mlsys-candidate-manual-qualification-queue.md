# MLSys candidate qualification queue — manual triage, not eligibility

**Prepared:** 2026-10-22 (JST). **Source:** frozen 57-PR metadata [survey](../results/mlsys-task-candidate-survey-v1/screen-1/screen.json), [review report](mlsys-task-candidate-survey-v1-report.md). This is an operational shortlist, not additional certified benchmark data, train/dev/held-out assignment, model exposure clearance or successful runtime evidence. **No model inference, repository test, task certification, or training was executed to create this document.**

The frozen metadata screen retained **24/57** PRs for *manual qualification*, and **33** were excluded at this coarse gate. The earlier source-bound [readiness audit](mlsys-actual-model-update-readiness-v2-report.md) recorded 19 existing entries, 14 unique definitions, **zero eligible held-out tasks**, against a minimum of 15. These are **different inventories** and must not be added together. No newly eligible held-out task is certified here.

## Review queue

Administrative order: fewer reported changed lines first, not descending suitability or verified compute cost. Changed files / test paths are counts from the frozen survey. Dates, upstream source hashes and details must be reconfirmed before any prospective protocol is frozen.

| Triage order | Upstream PR | Summary | Changed lines | Files / tests | Status |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | [ml-explore/mlx-lm#1938](https://github.com/ml-explore/mlx-lm/pull/1938) | Fix incorrect masking axis in phi3small | 9 | 2 / 1 | Unqualified; prior-use unknown |
| 2 | [huggingface/trl#7768](https://github.com/huggingface/trl/pull/7768) | Test that LoRA on an MoE model also trains its dense and shared expert layers | 16 | 1 / 1 | Unqualified; prior-use unknown |
| 3 | [ml-explore/mlx-lm#1935](https://github.com/ml-explore/mlx-lm/pull/1935) | Fix: max_tokens: 0 -> Server accepts max_tokens: 0, generator then rejects it | 17 | 2 / 1 | Unqualified; prior-use unknown |
| 4 | [huggingface/transformers#49147](https://github.com/huggingface/transformers/pull/49147) | This PR fixes Gemma3n fp16 loading when missing weights need to be in… | 24 | 3 / 1 | Unqualified; prior-use unknown |
| 5 | [huggingface/transformers#49116](https://github.com/huggingface/transformers/pull/49116) | qwen3 models map to wrong tokenizer class on the hub | 31 | 2 / 1 | Unqualified; prior-use unknown |
| 6 | [ml-explore/mlx-lm#1904](https://github.com/ml-explore/mlx-lm/pull/1904) | Fix incorrect assumption when parsing tool_calls[].function.arguments | 32 | 2 / 1 | **Existing VARE grader task; re-audit prior use** |
| 7 | [ml-explore/mlx-lm#1910](https://github.com/ml-explore/mlx-lm/pull/1910) | Fix Qwen3 Coder argument parsing when the top-level type is missing | 32 | 2 / 1 | Unqualified; prior-use unknown |
| 8 | [huggingface/transformers#49412](https://github.com/huggingface/transformers/pull/49412) | Fix ValueError when loading a fusion-enabled checkpoint twice | 35 | 2 / 1 | Unqualified; prior-use unknown |
| 9 | [ml-explore/mlx-lm#1902](https://github.com/ml-explore/mlx-lm/pull/1902) | Fix lm-eval continuation scoring after partial context truncation | 37 | 2 / 1 | Unqualified; prior-use unknown |
| 10 | [ml-explore/mlx-lm#1787](https://github.com/ml-explore/mlx-lm/pull/1787) | Stop gradients through Qwen3 MoE expert-routing indices | 39 | 2 / 1 | Unqualified; prior-use unknown |
| 11 | [ml-explore/mlx-lm#1926](https://github.com/ml-explore/mlx-lm/pull/1926) | Fix Falcon-H1 cache initialization and zero gradients | 39 | 2 / 1 | Unqualified; prior-use unknown |
| 12 | [huggingface/transformers#49208](https://github.com/huggingface/transformers/pull/49208) | correct final metrics due to resuming from checkpoint | 40 | 2 / 1 | Unqualified; prior-use unknown |
| 13 | [ml-explore/mlx-lm#1912](https://github.com/ml-explore/mlx-lm/pull/1912) | Keep the most likely token when top_p is small enough to mask everything | 41 | 2 / 1 | **Existing VARE grader task; re-audit prior use** |
| 14 | [ml-explore/mlx-lm#1930](https://github.com/ml-explore/mlx-lm/pull/1930) | Fix gradients through Mistral4 and MiniMax indices | 42 | 3 / 1 | Unqualified; prior-use unknown |
| 15 | [ml-explore/mlx#4662](https://github.com/ml-explore/mlx/pull/4662) | Fix conv input gradient shape with uneven or large padding | 43 | 2 / 1 | Unqualified; prior-use unknown |
| 16 | [ml-explore/mlx-lm#1921](https://github.com/ml-explore/mlx-lm/pull/1921) | Fix GraniteMoeHybrid padded batch logprobs (#1908) | 55 | 2 / 1 | Unqualified; prior-use unknown |
| 17 | [huggingface/transformers#45852](https://github.com/huggingface/transformers/pull/45852) | Fix slow Trainer path with 4D attention mask | 60 | 2 / 1 | **Existing VARE grader task; re-audit prior use** |
| 18 | [vllm-project/vllm#52626](https://github.com/vllm-project/vllm/pull/52626) | [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for weight sync | 67 | 2 / 1 | Unqualified; prior-use unknown |
| 19 | [huggingface/transformers#48319](https://github.com/huggingface/transformers/pull/48319) | Ignore a stale best checkpoint recorded in a resumed trainer state | 68 | 2 / 1 | Unqualified; prior-use unknown |
| 20 | [ml-explore/mlx-lm#1925](https://github.com/ml-explore/mlx-lm/pull/1925) | Honour the left-padding mask in MiniCPM3 and Phixtral batches | 85 | 3 / 1 | Unqualified; prior-use unknown |
| 21 | [ml-explore/mlx#4668](https://github.com/ml-explore/mlx/pull/4668) | Fix direct CPU conv writing past output | 90 | 2 / 1 | Unqualified; prior-use unknown |
| 22 | [ml-explore/mlx-lm#1795](https://github.com/ml-explore/mlx-lm/pull/1795) | Stop gradients through MoE routing indices in granitemoe, granitemoehybrid, lfm2_moe | 94 | 4 / 1 | Unqualified; prior-use unknown |
| 23 | [vllm-project/vllm#56841](https://github.com/vllm-project/vllm/pull/56841) | [Bugfix][KVConnector] Make ExampleHiddenStatesConnector abort-safe | 102 | 2 / 1 | Unqualified; prior-use unknown |
| 24 | [huggingface/transformers#49500](https://github.com/huggingface/transformers/pull/49500) | Save the PEFT adapter of a load-time sharded model from its own parameters | 163 | 3 / 2 | Unqualified; prior-use unknown |

The VARE grader tasks #1904, #1912 and Transformers #45852 are visibly marked as *already integrated into local grading studies*. This alone does **not** prove model exposure; however, they must not be silently promoted into a **newly blind** held-out pool. The separate development MLX-LM #1971 task is not on this coarse shortlist and has its own tracked study. All other entries have unknown historical exposure until their raw prior-use records are independently checked.

## Start with small-scope manual inspections

Based **only** on the frozen diff-size/test-path metadata, inspect these eight first (not a promise of usable grading): [ml-explore/mlx-lm#1938](https://github.com/ml-explore/mlx-lm/pull/1938), [huggingface/trl#7768](https://github.com/huggingface/trl/pull/7768), [ml-explore/mlx-lm#1935](https://github.com/ml-explore/mlx-lm/pull/1935), [huggingface/transformers#49147](https://github.com/huggingface/transformers/pull/49147), [huggingface/transformers#49116](https://github.com/huggingface/transformers/pull/49116), [ml-explore/mlx-lm#1910](https://github.com/ml-explore/mlx-lm/pull/1910), [huggingface/transformers#49412](https://github.com/huggingface/transformers/pull/49412), [ml-explore/mlx-lm#1902](https://github.com/ml-explore/mlx-lm/pull/1902). Some may still require GPU, upstream network fixtures or complex model loading and must be rejected from the CPU-only path rather than silently weakened.

For each candidate, complete all of the following **before** considering task certification:

1. **Immutable identity:** upstream repository and source PR; baseline/fixed full commit SHAs; target, test and prompt SHA-256; changed-file allowlist; license and per-file obligations.
2. **Prior exposure:** search frozen Qwen trials, coding-agent rollouts, SFT examples, development experiments and task paraphrases. Record provenance for each exclusion or uncertainty, not merely a statement of freshness. A date after the model's advertised training cutoff is only a proxy and cannot establish no memorization.
3. **Independent upstream reproduction:** source pinned by SHA with replacement refs disabled; baseline test **fails**, exact known fix **passes**, all on the declared Apple Silicon/CPU environment. Record missing dependencies and upstream limitations; no fabricated success.
4. **Narrow hidden grader:** expose only task/workspace/public checks to the candidate, never hidden expected values. Reject source/implementation hash drift, wrong source revision, symlink/traversal tricks, forged stdout, network use and modified evaluator resources; add adversarial wrong patches.
5. **Bias and overlap audit:** deduplicate related bug families, shared tests and duplicate prompts before computing unique task counts. Separate training, development and blind held-out membership *before* agent/model use; do not reuse previously exposed tasks as unexposed tests.
6. **Append-only qualification:** freeze prospective protocol and raw manifests; publish SHA-256 plus baseline/fix/mutant outcomes. Failed or unsupported tasks remain visibly failed; never change original frozen survey.

### Per-task outcome template

Use these exact fields when filing a future review; placeholders are **not qualified facts**:

```yaml
status: UNQUALIFIED
upstream_repository: owner/repo
upstream_pr: 0
baseline_commit: REQUIRED_FULL_SHA
fixed_commit: REQUIRED_FULL_SHA
target_path: REQUIRED
upstream_test_selector: REQUIRED
upstream_test_sha256: REQUIRED
candidate_visible_prompt_sha256: REQUIRED
per_file_license_verified: false
model_or_agent_prior_exposure: UNKNOWN
exposure_evidence_ref: REQUIRED
cpu_reproduction: NOT_RUN
baseline_fails: NOT_RUN
known_fix_passes: NOT_RUN
adversarial_invalid_patch_rejected: NOT_RUN
isolated_red_team: NOT_RUN
proposed_split: UNASSIGNED
protocol_sha256: NOT_FROZEN
qualification_evidence: NOT_AVAILABLE
```

The runner/admission gate must treat `UNKNOWN`, `NOT_RUN`, missing hashes, pending review and `UNQUALIFIED` as **ineligible**. Updating this checklist is not a qualification result. The previously frozen 15-unique-held-out gate remains **NO-GO**, pending a genuinely new audited inventory and disjoint split.

## Execution / decision boundary

The immediate engineering queue is [PR #98](https://github.com/mitukx/VARE/pull/98) live Colima calibration, [PR #99](https://github.com/mitukx/VARE/pull/99) diverse scales + cached values, and [PR #100](https://github.com/mitukx/VARE/pull/100) accurate failure taxonomy/raw diff. See the [E2E experimental preregistration proposal](coding-rl-readiness-and-e2e-plan.md). Do not begin model training, expose fresh candidate prompts to an agent, or declare any of these 24 held-out eligible while those prerequisite audits remain incomplete.
