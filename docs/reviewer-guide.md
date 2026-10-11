# Reviewer guide

This guide separates code that is implemented, behavior reproduced on CPU, and model outcomes. The linked reports and raw records are the source of truth; the summaries below do not replace them. Repository and PR status was checked on 2026-10-11.

## A short review path

1. **Read the claim boundary.** [README](../README.md) states the current result; [AGENTS.md](../AGENTS.md) defines the evidence ladder and invariants. VARE has not shown independently confirmed task-success improvement after a real-model update.
2. **Inspect the upstream rollback correction.** RVL [PR #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92) proposes restoring shared-child module modes without recursive alias propagation. The [baseline/fix report](rvl-module-mode-shared-alias-followup-v1.md), [patch](../contributions/rvl-module-mode-shared-alias-followup.patch), [raw replay](../results/rvl-module-mode-shared-alias-v1/), and [reproducer](../scripts/reproduce_rvl_shared_module_mode_followup_v1.sh) are linked here. The regression is a constructed CPU case; PR #92 is open, its CI was green at the checked snapshot, and no independent human review is recorded.
3. **Inspect the task and grader path.** VARE [PR #3](https://github.com/mitukx/VARE/pull/3) contains the [urllib3 #3352 report](urllib3-3352-real-model-feasibility-v1-report.md), [task lock](../benchmarks/historical/urllib3_3352/protocol.lock.json), [grader](../benchmarks/historical/urllib3_3352/evaluator/grade.py), VARE catalog identity fix, adversarial records, and local model outputs. The public task is for development calibration only; neither one-shot model attempt produced a patch and no training was run.
4. **Check the research stop decision.** Merged VARE [PR #2](https://github.com/mitukx/VARE/pull/2) records rollback evidence and a follow-up prior-art screen. That screen did not admit a new training or GPU experiment; its candidate analysis is linked from the [experiment index](experiments.md).
5. **Read the model outcomes and history.** The [OpenBookQA confirmation report](openbookqa-qwen25-grpo-sft-confirmation-v1-report.md) records a frozen negative GRPO/SFT comparison. The [current gaps](current-gaps.md), [evidence ledger](evidence.md), [experiment index](experiments.md), and [roadmap](roadmap.md) retain limitations, prior decisions, and older results.

## Pull-request status snapshot

Snapshot checked 2026-10-11 via the GitHub PR API after VARE #2 merged and before VARE #3 was updated. “Checks” are CI results; they are not independent review. Use the linked PRs for live status.

| PR | State | Checks | Review record |
| --- | --- | --- | --- |
| [RVL #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92) | Open, mergeable | 7/7 passed | Automated Codex summary comment; no submitted human review |
| [VARE #2](https://github.com/mitukx/VARE/pull/2) | Merged as `09dfcd5` | 5/5 passed before merge | Automated Codex review and author comment; no independent human review recorded |
| [VARE #3](https://github.com/mitukx/VARE/pull/3) | Open at the pre-update snapshot | 5/5 passed on the earlier head; checks rerun after rebase | Automated Codex review; no independent human review recorded |

VARE #2 is merged at `09dfcd5`. VARE #3 is the review vehicle for the task/evaluator work and this documentation update; its checks should be read from the live PR after the rebase. RVL #92 remains a separate upstream PR. No independent human review is recorded for these contributions.

## Reproduce the RVL shared-module regression

From the repository root, use a fresh output directory. The script clones the pinned RVL baseline, applies a test-only regression to one clone and the proposed correction to another, then runs the focused and related trainer tests. Execution is CPU-only; Git and network access are needed to fetch the source.

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_shared_module_mode_followup_v1.sh \
  /tmp/vare-rvl-shared-mode-review
```

The expected baseline failure and fixed run are recorded in [`results/rvl-module-mode-shared-alias-v1/`](../results/rvl-module-mode-shared-alias-v1/). The test establishes behavior for the constructed shared-module graph, not production incidence or accelerator behavior.

## Reproduce urllib3 task calibration and checks

This path requires macOS with `/usr/bin/sandbox-exec`, Git, Python 3.12 for VARE, and Python 3.11 for the pinned task package. It fetches immutable source revisions. Use fresh output directories.

```bash
PYTHONPATH=src python3.12 scripts/calibrate_urllib3_3352_task.py \
  --output /tmp/urllib3-calibration

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3.12 -m pytest -q \
  tests/test_urllib3_3352_evaluator.py tests/test_environment_catalog.py

PYTHONPATH=src python3.12 scripts/adversarial_test_urllib3_3352.py \
  --base-run /tmp/urllib3-calibration/pinned-base/preparation \
  --output /tmp/urllib3-adversarial
```

The public known fix is a grader calibration control, not an agent result. Issue #3352 and its fix are public, so the task is excluded from confirmation use. The grader is macOS Seatbelt-specific and does not enforce a memory ceiling. Raw calibration, model, tamper, and AI-review records are retained under [`results/urllib3-3352-evaluation-v1/`](../results/urllib3-3352-evaluation-v1/). AI review is not human review.

## Code and evidence map

| Path | Review use |
| --- | --- |
| `src/vare/`, `tests/` | VARE implementation and local regression tests |
| `benchmarks/historical/` | Immutable task contracts and protected graders |
| `protocols/` | Frozen protocols, thresholds, seeds, and hashes |
| `results/` | Raw logs, manifests, model outputs, and review records |
| `docs/` | Technical reports, evidence summaries, and decisions |
| `contributions/` | Proposed upstream patches, separate from merged code |
| `scripts/`, `reproducers/` | Reproduction and audit entry points |

## Claim limits

| Evidence | Supports | Does not support |
| --- | --- | --- |
| CPU regression and clean-clone replay | Correctness on the pinned fixture and code path | Production prevalence, GPU/distributed correctness, or model gain |
| Known-fix task calibration | Grader distinguishes the pinned failing source from its known fix | Agent capability or general coding performance |
| One-shot model screen | Output validity and feasibility for the exact prompt, model, and task | Success probability, generalization, or a training effect |
| Frozen post-training comparison | The reported outcomes for the specific model, task, data, and protocol | Broad conclusions about an algorithm or task family |

An independent human reproduction or maintainer review is not recorded for the current work. Do not infer one from CI, author-run tests, or AI-agent review. No GPU/distributed verification or model-capability gain is claimed.
