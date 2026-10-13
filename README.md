# VARE

VARE is a CPU-first research project for studying post-training signals, policy updates, and evaluation reliability under limited compute. It acts as an outer improvement control plane and integrates with existing training systems such as [Recursive-Verification-Lag (RVL)](https://github.com/mitukx/Recursive-Verification-Lag), rather than reimplementing a full RL stack.

## Status at this snapshot

VARE has **not demonstrated an independently confirmed improvement in real-model task success after post-training**. Recent model studies are negative or failed their predeclared gates. The project retains those outcomes and does not treat runnable training code, grader calibration, or synthetic results as capability evidence.

The current reviewable work is:

- A VARE GRPO-adapter correction for the same shared-module restore edge. The adapter previously called `Module.train()` after the trainer restore; direct flag assignment passes a constructed CPU regression. This does not add production-incidence or model-quality evidence. See the [adapter report](docs/rvl-grpo-shared-module-mode-v1-report.md).
- A narrow RVL rollback correction for a shared-module mode restoration edge. Its constructed CPU regression fails on the pinned baseline and passes with the patch. [RVL PR #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92) is open; production incidence and model-quality impact are unknown.
- A calibrated, protected grader for the public urllib3 #3352 historical bug and a VARE task-identity correction. The public task is retired from confirmation use. Two one-shot local model attempts produced no applicable patch; no training was run.
- A structured research screen that admitted no new training or GPU experiment because no candidate cleared both novelty and fresh-task feasibility gates.

For exact limits, raw records, reproduction commands, and the current pull-request state, start with the [reviewer guide](docs/reviewer-guide.md). Status and links in this repository were checked on 2026-10-11; the adapter regression above was added on 2026-10-12. Use the linked pull requests for live status.

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
| VARE adapter rollback | On the previous adapter, a later shared-module alias restored the child to train mode and changed a mode-dependent stand-in forward. Direct flag assignment restores the checked flags without calling `train()`. The opposite registration order and an unshared nested pair keep their flags. | A production module graph, accelerator behavior, or a change in model quality. The stand-in forward is not Dropout. |
| RVL rollback | On pinned RVL `27ebf7f`, the shared-child regression fails exact mode restoration and the following Dropout output. The patch restores the checked model, optimizer, CPU RNG, and next-forward behavior; 14 trainer tests and 51 related CPU tests pass. PR #92 had 7/7 checks passing at the recorded snapshot. | Constructed CPU graph only; production incidence, CUDA/distributed behavior, model-quality impact, and independent human review remain unestablished. |
| Historical task evaluator | urllib3 #3352 pinned base fails 2/2 fail-to-pass cases and passes 3/3 controls; its public known fix passes all five. The evaluator exercises the actual request path and keeps protected assets outside the candidate workspace. | Agent-discovered fixes or general coding ability. The task and fix are public, so it is development-only. The two local model attempts yielded no applicable patch. |
| Real-model post-training | On the frozen OpenBookQA confirmation, base exact-answer success was 45.31%, matched SFT averaged 33.33%, and GRPO scored 0/128 for each of three seeds because outputs failed the frozen parser. The earlier ARC comparison also missed its frozen gate (38.84% GRPO vs. 40.00% SFT). | A general conclusion about GRPO or language models. These pairings and confirmation sets are retired. |

## Reproduce

From a Python 3.11 or newer checkout:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest --ignore=tests/test_math500_grader_v1.py --ignore=tests/test_math500_grader_v2.py
PYTHONPATH=src python3 scripts/check_tracked_blob_budget.py
```

The optional mathematical grader tests install `requirements/math500-grader-test.lock.txt` and are not part of that command.

The RVL shared-module regression clones a pinned upstream revision and runs CPU trainer tests:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_shared_module_mode_followup_v1.sh \
  /tmp/vare-rvl-shared-mode-review
```

The urllib3 #3352 calibration uses macOS `/usr/bin/sandbox-exec`, Git, Python 3.12 for VARE, and Python 3.11 for the pinned task package. It is not the Linux CI path.

```bash
PYTHONPATH=src python3.12 scripts/calibrate_urllib3_3352_task.py \
  --output /tmp/urllib3-calibration

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3.12 -m pytest -q \
  tests/test_urllib3_3352_evaluator.py tests/test_environment_catalog.py

PYTHONPATH=src python3.12 scripts/adversarial_test_urllib3_3352.py \
  --base-run /tmp/urllib3-calibration/pinned-base/preparation \
  --output /tmp/urllib3-adversarial
```

Further command context, expected records, and claim limits are in the [reviewer guide](docs/reviewer-guide.md).

## Experiment configs

Frozen study definitions are the JSON files in [`protocols/`](protocols/). A `*.lock.json` file is the recorded spec for a study that already has an outcome. Do not edit a lock after that outcome is recorded, and do not rerun a consumed prompt to change the recorded result. Raw outputs stay under [`results/`](results/). Summaries in `docs/` are derived from those files and their hashes.

## Repository map

| Path | Contents |
| --- | --- |
| `src/vare/`, `tests/` | VARE control-plane code and regression tests |
| `benchmarks/` | Task definitions, protected graders, and task locks |
| `protocols/` | Frozen study definitions, gates, and hashes |
| `results/` | Raw outputs, manifests, and retained failed or passed runs |
| `docs/` | Technical reports, evidence summaries, and decisions |
| `contributions/` | Upstream patch artifacts and status |
| `minigrpo/` | Small GRPO trainer for the integer seed-split train rows, scored on dev. CPU tests only; no retained model run. See [minigrpo](docs/minigrpo.md). |
| `selfimprove/` | Proposes recomputable integer problems, trains with minigrpo, and rolls a candidate back through `PromotionGate`. A format-only reward fixture is rejected in tests. No retained model round. See [selfimprove](docs/selfimprove.md). |
| `scripts/`, `reproducers/` | Reproduction, audit, and diagnostic commands |

Adapter checkpoints already committed under `results/` stay in git as retained raw records. [`protocols/tracked_large_blob_allowlist_v1.txt`](protocols/tracked_large_blob_allowlist_v1.txt) freezes those blobs at or above 1 MiB; `scripts/check_tracked_blob_budget.py` rejects a tree that adds, replaces, or drops one. New files at or above that size, and benchmark content with uncertain redistribution terms, stay outside the repository. Record a new external file's SHA-256 in the study manifest. Published history is not rewritten. See [artifact retention](docs/artifact-retention.md). Read each report for its exact source revisions, environment, raw artifacts, and claim limits. See [LICENSE](LICENSE) for the repository license.
