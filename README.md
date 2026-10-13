# VARE

## Question

Under a fixed compute budget, which post-training signal improves an independently measured held-out outcome, and how do we know the update and the evaluation are correct?

Training reward is not that outcome. A base-model pass rate is not evidence that an update helps.

## Results

No retained run shows an independently confirmed held-out task-success gain after a real model update.

| Record | Observation | What it does not establish |
| --- | --- | --- |
| Exact-integer v1, SmolLM2-360M-Instruct, mlx | Strict pass@4 is 0/64 (`stop_zero`). Wilson 95% is [0, 0.05662405979280534]. | A reason to change the parser or to train. The post-hoc lenient rescore finds 19/256 expression matches and is not a gate result. See [v1](docs/heldout-exact-integer-base-gate-v1-result.md) and [the diagnostic](docs/two-tier-answer-scoring-v1.md). |
| Exact-integer v2, Qwen2.5-3B-Instruct 4-bit, mlx | Strict pass@4 is 29/64 (0.453125). Wilson 95% is [0.3372944778061167, 0.5742640277994588]. | A capability claim or a policy-update comparison. It authorizes only the separate [one-update smoke](docs/heldout-exact-integer-update-cost-smoke-v1.md). |
| OpenBookQA, Qwen2.5-0.5B | Base exact-answer success 45.31%, matched SFT 33.33%, GRPO 0/128 on every seed because outputs missed the frozen parser. | A general claim about GRPO. This pairing is retired. |
| ARC-Challenge, Qwen2.5-0.5B | The frozen GRPO-versus-SFT comparison missed its gate (38.84% vs 40.00%). | A gain from the update. This pairing is retired. |

The historical record stays in [the experiment index](docs/experiments.md), [the evidence ledger](docs/evidence.md), and [current gaps](docs/current-gaps.md). `docs/` is not moved or deleted.

## Status at this snapshot

VARE is a CPU-first control plane for post-training signals, policy updates, and evaluation reliability under limited compute. It integrates with existing training systems such as [Recursive-Verification-Lag (RVL)](https://github.com/mitukx/Recursive-Verification-Lag) instead of reimplementing a full RL stack.

The current reviewable work also includes:

- A narrow RVL rollback correction for a shared-module mode restoration edge. Its constructed CPU regression fails on the pinned baseline and passes with the patch. [RVL PR #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92) is open; production incidence and model-quality impact are unknown.
- A calibrated, protected grader for the public urllib3 #3352 historical bug and a VARE task-identity correction. The public task is retired from confirmation use. Two one-shot local model attempts produced no applicable patch; no training was run.
- A structured research screen that admitted no new training or GPU experiment because no candidate cleared both novelty and fresh-task feasibility gates.

For exact limits, raw records, reproduction commands, and the current pull-request state, start with the [reviewer guide](docs/reviewer-guide.md). Status and links in this repository were checked on 2026-10-11; use the linked pull requests for live status.

## Review path

1. [Reviewer guide](docs/reviewer-guide.md): what to read, reproduce, and claim.
2. [VARE PR #3](https://github.com/mitukx/VARE/pull/3): historical-task evaluator, adversarial checks, task-identity fix, and base-model feasibility screen.
3. [RVL PR #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92): shared-module rollback correction and upstream tests.
4. [VARE PR #2](https://github.com/mitukx/VARE/pull/2) (merged): rollback evidence and research-gate follow-up.
5. [Current evidence gaps](docs/current-gaps.md), [evidence ledger](docs/evidence.md), [experiment index](docs/experiments.md), and [roadmap](docs/roadmap.md): full historical record and remaining work.

VARE PR #2 was merged into `main` as [`09dfcd5`](https://github.com/mitukx/VARE/commit/09dfcd5e02d37ad78b86948e555bbfd169cffc88). PR #3 carries the historical-task evaluator and the reviewer-facing documentation update; shared documentation was reconciled after #2 landed. RVL #92 is a separate upstream change.

## Selected evidence

| Area | Observed result | What it does not establish |
| --- | --- | --- |
| RVL rollback | On pinned RVL `27ebf7f`, the shared-child regression fails exact mode restoration and the following Dropout output. The patch restores the checked model, optimizer, CPU RNG, and next-forward behavior; 14 trainer tests and 51 related CPU tests pass. PR #92 had 7/7 checks passing at the recorded snapshot. | Constructed CPU graph only; production incidence, CUDA/distributed behavior, model-quality impact, and independent human review remain unestablished. |
| Historical task evaluator | urllib3 #3352 pinned base fails 2/2 fail-to-pass cases and passes 3/3 controls; its public known fix passes all five. The evaluator exercises the actual request path and keeps protected assets outside the candidate workspace. | Agent-discovered fixes or general coding ability. The task and fix are public, so it is development-only. The two local model attempts yielded no applicable patch. |
| Real-model post-training | On the frozen OpenBookQA confirmation, base exact-answer success was 45.31%, matched SFT averaged 33.33%, and GRPO scored 0/128 for each of three seeds because outputs failed the frozen parser. The earlier ARC comparison also missed its frozen gate (38.84% GRPO vs. 40.00% SFT). | A general conclusion about GRPO or language models. These pairings and confirmation sets are retired. |

## Repository map

| Path | Contents |
| --- | --- |
| `src/vare/`, `tests/` | VARE control-plane code and regression tests |
| `benchmarks/` | Task definitions, protected graders, and task locks |
| `protocols/` | Frozen study definitions, gates, and hashes |
| `results/` | Raw outputs, manifests, and retained failed or passed runs |
| `docs/` | Technical reports, evidence summaries, decisions, and the [report template](docs/templates/paired-comparison-report.md) |
| `contributions/` | Upstream patch artifacts and status |
| `scripts/`, `reproducers/` | Reproduction, audit, and diagnostic commands |

## Held-out exact-integer records

The additive [seed-aware summary v3](docs/seeded-summary-v3-report.md) keeps the
frozen v2 primary decision intact and distinguishes pass@4 within a fitted run
from pooled outputs across runs. Synthetic baseline/fix evidence also checks
that metrics and progress hashes describe the same bytes. This is evaluation
correctness evidence, not a model-learning result.

The 64-task pack is `benchmarks/heldout/exact_integer_v1/`. v1 is `stop_zero`. v2 is the 29/64 base gate in the table above. A later train/dev/confirm seed list is [integer seed split v1](docs/integer-seed-split-v1.md). Report calculations are [eval stats](docs/eval-stats-v1.md). The [one-update smoke](docs/heldout-exact-integer-update-cost-smoke-v1.md) retained an accepted mlx record: one step, loss 3.004885196685791, reload matched, peak memory 1959434978 bytes, wall time 8.762955541998963 seconds. That record is not a held-out gain. [Format versus correctness v1](docs/format-vs-correctness-comparison-v1.md) still exits 2. The successor [v2 protocol](docs/format-vs-correctness-comparison-v2.md) freezes five seeds and 16 matched optimizer steps. It has no retained run.

Large checkpoints and benchmark content with uncertain redistribution terms remain outside version control. Read each report for its exact source revisions, environment, raw artifacts, and claim limits. See [LICENSE](LICENSE) for the repository license.
