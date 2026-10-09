# AsyncGRPO freshness at the CPU Trainer step boundary

**Result:** on CPU, the pinned TRL queue/batcher/collator path can deliver a version-0 row to an actual Transformers `training_step` at current version 1 when `gradient_accumulation_steps=1` and `max_staleness=0`. At GA=2, both version-0 fork rows enter the same optimizer update at version 0. The base and local atomic-admission candidate produce identical traces.

This is a bounded dataflow finding about the difference between queue freshness checking and training-step use. It is not an unambiguous defect claim: TRL currently checks `max_staleness` when a sample leaves `RolloutQueueDataset`, before prefetch and before the next optimizer update. The [pinned config](https://github.com/huggingface/trl/blob/ed8cc2f4337fb9b7b1429db31009ad3577cd5b98/trl/experimental/async_grpo/async_grpo_config.py) describes a maximum version lag before discard but does not explicitly settle whether the cap applies at queue admission or at the later training use. [Issue #7206](https://github.com/huggingface/trl/issues/7206) discusses complete-rollout admission, a related but distinct boundary.

## Frozen protocol

Protocol `trl_async_token_budget_real_trainer_window_v4` and the script were frozen in commit `d44eee8` before confirmation. The protocol pins TRL `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98`, Transformers 4.57.3, Accelerate 1.12.0, and relevant source file hashes. The script sets `TrainingArguments(use_cpu=True, dataloader_pin_memory=False)` and records the selected device. Both baseline and candidate reported `cpu`; both exited 0.

It uses the real `Trainer.train()` loop with production `RolloutQueueDataset`, `TokenBudgetBatcher`, and `DataCollatorForRollout`, one process, a deterministic scalar model/loss, and SGD at 0.001. A local callback advances policy version at each real `on_step_end`, after `optimizer.step`, matching where pinned AsyncGRPO registers weight sync. GA=1 runs two optimizer steps; GA=2 runs one. The model is synthetic and does not represent a language model or GRPO objective.

## Results

| Source | GA | Rows actually trained | Version at each training step | Scalar parameter after updates |
|---|---:|---|---|---|
| Base | 1 | row 10, then row 11 | 0, then 1 | 0.1000 → 0.0990 → 0.0985 |
| Candidate | 1 | row 10, then row 11 | 0, then 1 | 0.1000 → 0.0990 → 0.0985 |
| Base | 2 | rows 10 and 11 in one update | 0, 0 | 0.1000 → 0.0990 |
| Candidate | 2 | rows 10 and 11 in one update | 0, 0 | 0.1000 → 0.0990 |

At GA=1, the production collator sees rows 10 and 11 at version 0. Accelerate 1.12.0's [`DataLoaderDispatcher`](https://github.com/huggingface/accelerate/blob/v1.12.0/src/accelerate/data_loader.py) prefetches the next iterable batch before yielding the current batch; after the first optimizer step and version increment, the Trainer uses the already collated row 11 in its second `training_step`. Thus row 11 is fresh when collated but one update old when trained. The scalar parameter makes the resulting optimizer step observable: it moves from 0.0990 to 0.0985. At GA=2, both old rows are used before the update and both are version-aligned. These observations were identical on pinned base and candidate.

The parameter delta is a synthetic optimizer measurement, not a causal comparison against a strict-freshness filter. The different accumulation widths also change optimizer scheduling, so the final parameter values must not be attributed solely to row staleness. The fixture uses `max_staleness=0`; the pinned config default is 4, under which lag 1 is allowed.

## Evidence and limitations

Reproduction commands, raw stdout/stderr, parsed results, exit codes, environment, and source identifiers are retained in [run-1](../results/trl-async-token-budget-real-trainer-window-v4/run-1/). An [independent audit](../results/trl-async-token-budget-real-trainer-window-v4/run-1/independent-audit.md) verified hashes, CPU device records, matching outputs, and the version trace. The script sets `use_cpu=True` and each result records `device: cpu`; minor protocol wording mismatch: the script does not contain a literal runtime assertion for the device. The preceding v3 attempt selected MPS and is retained as a protocol non-pass in the [v1 report](trl-async-token-budget-real-trainer-window-v1-report.md); v4 is the valid CPU run.

This is one deterministic, single-process, toy-model run. It does not measure real-model gradients, generalization, task success, training-run prevalence, or capability gain. It establishes that, under this supported configuration, a row can be queue-checked before a version transition yet used by the next actual Trainer step after that transition. The code currently performs the stale check at queue yield. A training-step freshness fix would need to preserve group accounting, distributed synchronization, and epoch-stop behavior.

**Decision: STOP extending the atomic-admission candidate as a freshness fix.** The candidate does not alter this boundary. Keep this result as a reproducible correctness question and do not open an upstream PR until the intended check/use contract is explicit and a correction is tested against rollout and training-window invariants. No model improvement is demonstrated.
