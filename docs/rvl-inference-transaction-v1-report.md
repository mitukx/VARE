# RVL inference state and cancellation isolation

## Integration update

During this work, separate repository activity merged #5–#8 into main. This branch then incorporated main `1e7fda8265cdc1617525d0c740e90fcd41e92b81`; the pull request now targets main. No main merge was performed by this contribution. Implementation after integration is `39f3debe8f09197030f2b66e0e83605dca82b666`.

The [updated replay](../results/rvl-inference-transaction-v1/integration-main-1e7fda8/) adds that main as a fourth control: **12/14 failures on initial main, 10/14 on the alias-only PR, 10/14 on integration main, 0/14 after this correction**, with no setup errors or skips. Actual HF probes agree across both controls. The updated full local suite passes **349 tests** in 61.50 seconds; Ruff and package-wide mypy pass. The four-arm reproduction takes 27.59 seconds on the recorded host. Older three-arm evidence below is retained unchanged. Source/API type changes from main and the new control motivated this integration check, rather than another identical experiment.

The first integrated full-suite attempt had three environment failures (missing sparse fixtures and a subprocess resolving macOS's Python 3.9). Materializing the referenced fixtures and putting the Python 3.12 environment first on PATH resolves them without changing application code. The failed and successful logs are both retained in the updated bundle. Activate the documented virtual environment when reproducing the full suite.

## Result and ownership

On the pinned VARE main, active-policy inference changes mixed module flags and CPU RNG without changing policy identity. Cancelling evaluation can also release the adapter lock while HF generation is still running in a thread. A following rollout then enters the same model concurrently. The correction preserves ephemeral inference state and drains generation before releasing the lock.

The shared-module rollback correction is **existing work**, reused from [VARE #5](https://github.com/mitukx/VARE/pull/5), with a parallel proposal in [#4](https://github.com/mitukx/VARE/pull/4) and the corresponding trainer correction in [RVL #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92). This change adds real Torch rollback coverage, the distinct inference/cancellation correction, and a pinned three-arm reproduction. It does not introduce a new RL algorithm or demonstrate model improvement.

| Source | Immutable revision |
| --- | --- |
| VARE main baseline | `b8dc1baccaadfa7f699549c254a751c2b9bc0d95` |
| Alias-only control / PR #5 head | `5f2d2e61d2ea8f771d99315757550091053bad46` |
| Corrected implementation | `3139f4dc7cf4421e197af89c0010ecbad81e4d75` |
| Actual HF backend and trainer | RVL `27ebf7fe239d97504eeb960bf1407de220584dac` |

## Execution paths and causes

1. **Shared graph:** `named_modules()` deduplicates a shared child. `parent.train(mode)` recursively revisits that child through each parent edge. A later parent can overwrite the child's saved flag. PR #5 assigns the saved Boolean flags directly. This matches RVL #92's proposed contract and does not replay custom `train()` side effects.
2. **Same policy, changed runtime:** `HFLocalBackend._generate_sync()` calls `torch.manual_seed()`, switches the model to eval, and finally calls `model.train(was_training)`. Restoring only the root flag flattens mixed child modes. VARE `_restore()` skips a reload when `_loaded_id` already equals the requested policy. Successful active evaluation/rollout, and scorer failure after generation, therefore leave changed RNG/modes. The next candidate can snapshot this contaminated incumbent state.
3. **Cancellation:** HF generation uses `await asyncio.to_thread(...)`. Cancelling that await does not terminate the worker. The adapter's `async with _model_lock` exits while the worker still holds the model in eval mode; a second generation can start, and the first worker's `finally` can change the second generation's mode.

The corrected order is:

```text
acquire model lock -> load selected policy -> capture flags + RNG
  -> await shielded backend generation
  -> if cancelled, drain backend task (also across repeated cancels)
  -> restore captured flags + RNG -> release model lock -> propagate cancellation
```

The guard copies neither model weights nor optimizer state per prompt. Inference must treat those tensors as read-only. If ephemeral restoration fails, `_loaded_id` becomes unknown; evaluation attempts incumbent recovery, while rollout raises and forces a policy restore on subsequent access. Worker exceptions during cancellation are retrieved before the original cancellation is propagated.

## Decisive comparisons

Same test files, dependencies, and pinned RVL source for all arms; only VARE source changes. No model download, paid API, or accelerator execution. A random GPT-2 has vocab 5, embedding width 8, one layer/head, and at most two generated tokens. Its real RVL trainer performs one fixture update before each probe to populate AdamW moments; this is test setup, not a learning study.

| Regression family | Cases | Main failures | Alias-only failures | Corrected failures |
| --- | ---: | ---: | ---: | ---: |
| Candidate success/failure, shared/tree graph, exact state and next forward | 4 | 2 | 0 | 0 |
| Active evaluation/rollout, backend success/failure | 4 | 4 | 4 | 0 |
| Repeated cancellation, evaluation/rollout, worker success/failure | 4 | 4 | 4 | 0 |
| RNG cleanup failure, evaluation/rollout recovery | 2 | 2 | 2 | 0 |
| **Total** | **14** | **12** | **10** | **0** |

The alias-only control isolates the new defect from the already proposed rollback fix. The unshared rollback cases are passing negative controls. Exact tensor and nested optimizer-state comparison avoids treating mode restoration alone as a complete transaction check.

Actual HF probes, separate from the test-double regressions:

| Probe | Main / alias-only | Corrected |
| --- | --- | --- |
| Active evaluate, evaluate with scorer exception, active rollout | Model/optimizer tensors unchanged; CPU RNG and module modes changed in all three | Model/optimizer/CPU RNG/module modes all exact in all three |
| Cancel blocked evaluation, then start rollout | Cancelled caller returns, lock released, second generation overlaps first | Caller remains pending with lock held; second generation waits |
| Final state after both workers complete | CPU RNG/modes differ from initial snapshot | Entire trainer snapshot exact |

Full local suite: **338 passed**, including symbolic grader tests, Python 3.12.12 / Torch 2.9.1 / Transformers 4.57.3 / pytest 8.4.2 on macOS arm64, CPU. The reproduction took 19.56 seconds, including source fetch and three arms; the full suite took 60.76 seconds. These are single-run wall times, not performance benchmarks or confidence intervals. Deterministic state comparisons and controlled thread events establish the specified failures, not their frequency in production.

## Reproduce from a clean checkout

Use Python 3.12+, Git, network access for pinned Git objects, and a fresh output directory. No private checkpoint or host-specific path is required.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
# Linux: install the CPU torch wheel first from https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/rvl-inference-cpu-test.txt
python -m pip install -e .
python scripts/reproduce_rvl_inference_transaction_v1.py --output /tmp/vare-inference-replay
# Optional dependencies for the complete suite, including symbolic graders:
python -m pip install -r requirements/math500-grader-test.lock.txt
python -m pytest
```

The one-entry script extracts pinned baseline/control source, fetches pinned RVL into a temporary sparse checkout, runs the same 14 tests against each VARE arm, and exercises the actual HF backend. It requires the expected failure counts and no setup errors/skips, checks raw HF results, and writes XML, logs, JSON, exact installed-package versions, source hashes, and artifact hashes. Expected baseline pytest exit 1 is recorded, not hidden as a script success without assertions. CI's dedicated `rvl-inference-cpu` job runs this on Linux and uploads the fresh replay; CI is a separate execution, not human review.

### 2026-10-15 historical-API compatibility replay

The trainer-boundary freshness change added a verifier-version argument to the current adapter fixture. The first PR CI replay then failed before exercising inference behavior: the pinned historical source arms did not accept that newer constructor argument, so their 14 tests all failed during setup. The fixture now passes the freshness provider only when the loaded adapter exposes that API. The current adapter still receives the provider; the historical arms omit it, leaving the inference-state comparison unchanged. The rerun reproduced the expected counts (baseline 12/14, alias-only 10/14, integration main 10/14, fixed 0/14) and all four actual-HF probes completed. Raw logs, XML, JSON, runtime versions, and source/artifact hashes are preserved in [`pr58-freshness-compat`](../results/rvl-inference-transaction-v1/pr58-freshness-compat/). This corrects a replay-harness compatibility problem; it is not a new model or learning result.

Raw records: [`results/rvl-inference-transaction-v1`](../results/rvl-inference-transaction-v1/). [`manifest.json`](../results/rvl-inference-transaction-v1/manifest.json) covers the replay; `supplemental-manifest.json` covers the full-suite and AI-review records. The first full-suite attempt used an incomplete sparse checkout and lacked an inherited package import path (11 failures, 325 passes). After materializing the required data/result fixtures and setting the import path, 338 passed. Both logs are retained; this was an environment correction, not an application-code repair. The two newly added cleanup-failure tests account for the total increasing from 336 to 338.

## Independent adversarial review

A separate AI reviewer executed real HF generation with three repeated cancellations per case, worker success/error, populated AdamW state, scorer failure, and injected RNG-cleanup failure. Its [raw record](../results/rvl-inference-transaction-v1/ai-review-probes.json) found no new defect within the selected CPU contract. That development review used RVL PR #92's trainer checkout (`7568c6e`), whose generation backend is unchanged; the authoritative one-entry replay above uses RVL main. Static review and executed probes are distinguished in `ai-review.md`. No independent human reproduction, maintainer review, external merge, or GPU result is implied.

## Scope and remaining gaps

- The alias graph is constructed; incidence in default models is unknown. The actual GPT-2 probe uses mixed child modes intentionally. RVL defaults to disabled Dropout, so changed flags alone do not establish changed default-model output quality.
- Inference preserves CPU and already-initialized CUDA RNG in code. Only CPU was measured. MPS/XPU RNG, Python/NumPy RNG, new device initialization, GPU/distributed behavior, and arbitrary scorer side effects are outside this claim.
- The model lock coordinates this adapter. Other adapters or unrelated threads using process-global RNG need external coordination. Backends must await all work using the model; detached threads hidden behind a completed backend task cannot be protected this way.
- Cancellation can wait as long as backend completion. Python cannot safely kill an arbitrary generation thread; this change promises no hard cancellation deadline. Process isolation is appropriate when such a deadline is required.
- Model/buffer/optimizer writes by a custom inference backend are outside the read-only contract. Custom `Module.train()` side effects, changed topology, and arbitrary custom module attributes are not restored by a Boolean mode map.
- The reviewer separately reproduced an engine-level gap: generic concurrent `LoopHooks` evaluation can leave a sibling evaluator running after `asyncio.gather` raises and candidate discard begins. That is uncorrected here; it needs explicit task ownership/draining at the engine boundary. The default RVL model lock reduces this to waiting/wasted evaluation in the examined path.
- No model task-success gain, training-seed effect, reward-hacking immunity, or recursive improvement is established. No retired confirmation task was consumed.

## Five-minute technical explanation

1. Define checkpoint state versus ephemeral inference state (one minute).
2. Draw one shared child with two parents and explain deduplicated enumeration versus recursive `train()` (one minute).
3. Trace active-ID fast-path and seeded generation state changes (one minute).
4. Trace cancellation of `to_thread`, shield/drain, and lock ownership (one minute).
5. Show main / alias-only / fixed results, cleanup failure, and CPU/device limits (one minute).

Strongest objection: "The existing alias fix already solves this, or the fixture is only a fake backend." The alias-only arm still fails ten regression cases and all three actual HF state probes. Real random GPT-2/HF generation also overlaps under baseline cancellation. This rejects those explanations for the examined path, without estimating deployment impact.

Alternatives rejected: always restore a full checkpoint per prompt (large tensor copies, does not by itself stop a live worker); save only the root flag (cannot recover mixed modes); call `train()` in preorder (shared aliases overwrite children); shield without draining (caller can still release the lock); suppress cancellation permanently (breaks caller semantics). Process isolation remains an option for hard deadlines, outside this minimal adapter correction.
