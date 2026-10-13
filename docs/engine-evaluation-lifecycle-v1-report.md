# Evaluation task lifetime before candidate rollback

## Problem and correction

`CapabilityLoop` used `asyncio.gather` for incumbent/candidate evaluation. A first exception propagates without stopping the sibling. The engine could discard a candidate snapshot and release its round lock while that sibling still used the snapshot. A minimal valid concurrent `LoopHooks` fixture reads the deleted state and raises `KeyError` after the round has returned.

The correction owns both evaluation tasks and shields their group. On error or caller cancellation, a separate shielded recovery task requests cancellation once for each live evaluator **unless already cancelling**, drains both to termination, and then discards the candidate. The round lock remains held through this process and through asynchronous discard. The first observed exception is re-raised; discard failure instead reports unknown policy state with its cause. Same-policy evaluation drains both tasks without discarding the incumbent. Successful evaluations remain concurrent.

```text
evaluation error / round cancellation
  -> request cancellation only for live, not-already-cancelling evaluators
  -> await both terminal evaluators
  -> discard candidate, if distinct
  -> release round lock and propagate original exception
```

This is the outer-loop lifetime boundary. [RVL inference cleanup](rvl-inference-transaction-v1-report.md) separately protects a generation worker and model state inside an adapter. The current change is stacked on that contribution, not a rewrite of its implementation.

## Fixed sources and reproduction

- Baseline: `79046a09768476909b89ff64faa77c512ecb8ced` (VARE inference correction head, before this engine change).
- Corrected implementation: `16fbbf6c7e38479e47c2d1ef2426422ff5fd3ce8`.
- Engine SHA256: `5ca49b9e1ec94e85c0de1f3f68a003806e30d4f7b926bcf083b9bfb5b391b9fa`.
- [Raw replay](../results/engine-evaluation-lifecycle-v1/), source/artifact hashes, Python/pytest/platform versions, XML and logs are retained.

From a clean checkout, use Python 3.12+, Git, and pytest 8.4.2:

```bash
python -m pip install -e . pytest==8.4.2
python scripts/reproduce_engine_evaluation_lifecycle_v1.py --output /tmp/vare-engine-evaluation
```

The entry point extracts pinned source, applies the same current test file to baseline and correction in separate processes, and runs a small observation probe against both. It asserts expected failure counts with no setup errors/skips. Shallow checkouts may fetch the pinned Git object. No model, private checkpoint, host-specific path, paid API, or accelerator is required. The default Python 3.12 CI job reruns it and uploads fresh results.

## Results

| Check | Baseline | Corrected |
| --- | --- | --- |
| New regression cases | 12 failures / 13 | 13 pass |
| Live sibling after failed round returns | Yes | No |
| Candidate access after snapshot discard | `KeyError` | Evaluator cancelled before discard |
| Event order | evaluation starts → discard → evaluation finishes | evaluation starts → evaluation finishes → discard |
| Unchanged success control | Concurrent evaluations, pass | Concurrent evaluations, pass |

The 13 cases cover either evaluator failing; secondary cleanup errors; repeated caller cancellation during evaluation cleanup and candidate discard; same-policy/no-candidate cleanup; discard error/cancellation; concurrency on success; and self/external cancellation preceding sibling failure. Full local suite: **362 pass**, including optional Torch and symbolic tests, in 70.57 seconds. Ruff 0.15.12 and mypy 1.19.1 pass. These are controlled task-lifetime observations, not a throughput benchmark or estimate of production incidence.

## Independent falsification and revision

A separate AI reviewer found a counterexample in the first proposed correction: cancelling an evaluator that was already unwinding its own cancellation interrupted asynchronous cleanup. The [failed proposal record](../results/engine-evaluation-lifecycle-v1/ai-review-counterexample.json) shows cleanup interrupted before discard. This failure was not discarded or relabeled as success.

The correction now tests `task.cancelling()` before sending cancellation. Two maintained regressions cover self/external prior cancellation. The reviewer then independently executed four combinations of prior self/external cancellation and secondary cleanup failure, with three additional caller cancellations in every case. All four retained the round lock and candidate, kept the evaluator's cancellation count at one, completed cleanup before discard, preserved the initial error object, and left no pending evaluator tasks. See [corrected raw observations](../results/engine-evaluation-lifecycle-v1/ai-review-corrected.json) and [review scope](../results/engine-evaluation-lifecycle-v1/ai-review.md). The reviewer read the maintained suite without rerunning it. This is AI review, not human or external-maintainer review.

## Contract and limits

- Hooks own their resources. A terminal evaluation coroutine must mean its model/thread/resource work is complete. Hook cleanup must itself be cancellation-safe. Draining a coroutine cannot terminate a hidden detached worker or prove arbitrary custom state restoration.
- Already-cancelling hooks are drained without another cancellation request. Hooks suppressing cancellation to resume ordinary work must clear their task cancellation state; otherwise the engine can only wait for termination. A noncooperative hook may wait indefinitely. No hard cancellation deadline is promised.
- The earliest exception observed at the evaluation boundary is preserved, including when later caller cancellations arrive. Secondary evaluator exceptions are retrieved. Failed discard explicitly reports unknown state rather than claiming rollback succeeded.
- This change covers evaluation-phase failure/cancellation recovery. Rollout-phase task ownership and interruption of successful-path promotion/discard are outside scope.
- Evidence is from valid generic `LoopHooks` fixtures on CPU. The default RVL model lock already serializes its evaluations and limits the examined baseline effect there mainly to continued/wasted evaluation. Neither actual model-quality impact nor accelerator/distributed behavior is measured here.
- No confirmation cohort, learning protocol, reward/parser threshold, or model checkpoint was consumed or changed. The result establishes a bounded control-plane correctness property, not a new learning algorithm or recursive improvement.
