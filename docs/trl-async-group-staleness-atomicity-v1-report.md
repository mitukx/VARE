# TRL AsyncGRPO group staleness atomicity v1

**Result: the pinned production queue iterator admits samples individually and can split a same-version, same-group batch across a policy-version transition. This is a source-level admission counterexample, not an optimizer or capability result.**

## Question

Does `RolloutQueueDataset` preserve all-or-none admission of rows from one logical GRPO prompt group when the policy version changes between queue reads?

TRL issue [#7206](https://github.com/huggingface/trl/issues/7206) describes one rollout yielding multiple training sequences and proposes retaining rollout identity and sequence membership so a complete rollout can be admitted before packing. Its separate token-normalization concern is already covered by open [PR #7249](https://github.com/huggingface/trl/pull/7249); this report examines only queue staleness.

## Frozen method

Protocol v2 was committed before the successful execution in commit `54cb543`. It pins TRL main `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98` (2026-10-08), hashes the production trainer and rollout-worker files, and uses a VARE-authored test of the production `RolloutQueueDataset.__iter__`.

The fixture queues three samples with the same `group_id=41` and `model_version=0`. The control holds the current version at 0. In the counterexample, it consumes one sample, advances the callback-visible policy version to 1, and then reads the rest with `max_staleness=0`. A current-version sentinel bounds the iterator. Accelerate `PartialState` is initialized because the production logger needs it when dropping stale rows.

The first v1 attempt is preserved as a setup failure: it omitted `PartialState()` and stopped at the first stale-row log call. No v1 acceptance criterion was evaluated. v2 adds only this runtime initialization; it does not change the data, threshold, or expected outcome.

## Results

| Case | Same-group rows yielded | Stale rows dropped |
|---|---:|---:|
| Fixed policy version | 3/3 | 0 |
| Version advances after first read | 1/3 | 2/3 |

The transition case then yields the fresh sentinel. The machine-measured behavior is therefore partial group admission: one old row (33.3%) was consumed, and the remaining two (66.7%) were dropped individually. This is the queue iterator's actual production code, invoked from a CPU fixture. The fixture advances a callback value between reads; it does not run a real optimizer step.

The raw stdout/stderr, exit code, v1 failure, command, and hashes are retained under [the run bundles](../results/trl-async-group-staleness-v2/run-1/). The v2 reproducer is [here](../scripts/reproduce_trl_async_group_staleness_v2.py), with the [frozen protocol](../protocols/trl_async_group_staleness_v2.lock.json).

## Interpretation and limitations

- The source checks `sample.model_version` independently for each dequeued sample and drops stale rows with `continue`; it carries `group_id` into the training example but does not make that staleness decision group-atomic.
- A regression can therefore remove members of a group after another member has already been admitted. This establishes a mismatch with an all-or-none admission contract; it does not yet quantify a change in gradient, optimizer update, reward, or task success.
- The test constructs queue samples directly. It does not exercise a live async worker, vLLM, the rollout-scoring path, token-budget packing, optimizer boundaries, or distributed hardware.
- `group_id` denotes a prompt's GRPO group in the pinned source. Individual forked training sequences do not retain their originating `rollout_id` in `RolloutSample`; the test does not claim to identify or isolate one completion's fork members.
- The behavior is already described in issue #7206. This reproduction is not a novelty claim. The only open linked PR found during the audit, #7249, addresses accumulation normalization, not group staleness.

## Decision

**Continue narrowly toward an end-to-end correctness assessment.** The queue-level failure is reproducible and low-cost, but an upstream patch should wait until a second frozen test carries samples through the real scoring/batching boundary and shows how partial admission changes the optimizer input. Then compare a minimal atomic-admission fix against the pinned baseline. Do not label this as a changed GRPO update or capability effect until the dataflow or gradient is measured.
