# Reviewer guide

This page is a short route through VARE's implementation and evidence. It separates upstream code changes, reproduced third-party work, real-model outcomes, and unresolved claims. The detailed reports and raw artifacts remain the source of truth.

## Five-minute review

1. **Project scope and claim limits:** read the [research snapshot](../README.md#research-snapshot) and [current evidence gaps](current-gaps.md). VARE has not demonstrated an independently confirmed task-success improvement after a real model update.
2. **RVL rollback contribution:** PR [#90](https://github.com/mitukx/Recursive-Verification-Lag/pull/90) merged the original nested-mode rollback fix. The [v4 report](rvl-module-mode-snapshot-final-review-v4.md) records its baseline/fixed test and upstream status. A later shared-module alias case exposes a remaining edge in that merged implementation; the [follow-up report](rvl-module-mode-shared-alias-followup-v1.md) contains the minimal patch, failing baseline test, 14-test fixed run, and reproduction command. The follow-up has not been submitted.
3. **TRL trainer reproduction:** the [AsyncGRPO full-Trainer/DDP report](trl-async-accumulation-normalization-pr7249-full-trainer-ddp-v5-report.md) compares the pinned base with an existing TRL PR candidate through Trainer, Accelerate, and two-rank CPU/Gloo paths. This is reproduced upstream work, not a VARE implementation or a model-capability result. TRL PR [#7249](https://github.com/huggingface/trl/pull/7249) is open with no reviews as checked on 2026-10-10.
4. **Real-model outcomes:** the three-seed [ARC GRPO-versus-SFT report](qwen-arc-grpo-sft-comparison-v3-report.md) is a non-pass. The [GSM8K constant-shift audit](cpu-lm-gsm8k-dpo-constant-shift-audit-v1-report.md) and [ASDiv transfer study](cpu-lm-gsm8k-dpo-asdiv-transfer-v1-report.md) retire the narrow forced-choice DPO line. Neither establishes capability improvement.
5. **Historical record:** use the [experiment index](experiments.md), [evidence ledger](evidence.md), and [roadmap](roadmap.md) for prior studies, failed gates, and full limitations. Protocols and raw run bundles are linked from their reports.

## Reproduce the current RVL follow-up

The script fetches two clean copies of the pinned merged RVL commit, applies a test-only regression to one and the proposed patch to the other, then runs the relevant trainer tests. It requires Git, Python 3.12, PyTorch 2.9.1, Transformers 4.57.3, and network access for the source clone; all test execution is CPU-only.

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_shared_module_mode_followup_v1.sh \
  /tmp/vare-rvl-shared-mode-review
```

The raw baseline/fixed logs, runtime, status, and patch hashes are in [`results/rvl-module-mode-shared-alias-v1/`](../results/rvl-module-mode-shared-alias-v1/). Use a fresh output directory for each run.

## Code and evidence map

| Path | Contents |
| --- | --- |
| `src/vare/` | Evaluation, provenance, replay, promotion, and RVL integration code |
| `tests/` | Unit and integration regressions for VARE-owned behavior |
| `protocols/` | Frozen study definitions, acceptance gates, and hashes |
| `results/` | Raw outputs, manifests, and retained failed or passed runs |
| `docs/` | Technical reports, current evidence gaps, and historical decisions |
| `contributions/` | Pinned upstream patch artifacts; see its [index](../contributions/README.md) |
| `scripts/`, `reproducers/` | Reproduction, audit, and focused diagnostic commands |

The main code path is an outer control plane: task execution and provenance feed replay and evaluation; independent reports go through the promotion gate; a rejected candidate is rolled back. VARE integrates with RVL for local GRPO trainer execution rather than claiming to replace a complete RL training stack.

## Evidence standard

Treat each claim at its measured level. CPU fixtures establish correctness on those fixtures. Synthetic policy studies establish behavior under their specified generators. Real-model preference metrics do not imply task success. A task-success claim requires a frozen independent grader, unused confirmation examples, matched baselines, and uncertainty estimates. See the evidence ladder and claim rules in [`AGENTS.md`](../AGENTS.md).

Current limitations include no independently validated model-capability gain, no independent human reproduction of the retained results, and no evidence of production incidence for the rollback defect. The RVL follow-up patch is ready for independent review but remains local pending authorization to contact the upstream project.
