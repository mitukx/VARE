# RVL GRPO partial-update rollback validation v1

## Question and decision

After a real GRPO step mutates model and optimizer state, does VARE restore the active incumbent if the candidate path fails before snapshotting or promotion?

**Pass for this adapter boundary.** The frozen smoke verified all 11 checks: the pinned trainer changed model parameters and optimizer state; the injected failure propagated; model, optimizer, and CPU RNG returned to their incumbent snapshots; active policy, candidate registry, and counter remained unchanged; and the next online rollout returned the incumbent. CUDA remained uninitialized.

This complements the fake-trainer regression in [`test_rvl_grpo_hooks.py`](../tests/test_rvl_grpo_hooks.py), which raises from inside `train_step` after a partial mutation. The actual-trainer smoke lets the real update finish, then injects the failure at the next candidate boundary.

## Frozen setup

The protocol and validator were committed at `2b0a7f05da4cf7edbefd852096b2e3112890eb95` before the recorded run. The test loaded `HFCausalLMGRPOTrainer` from pinned [RVL commit `c7e646b`](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py) and verified five source-file SHA-256 values from the lock. That trainer snapshots CPU model tensors, optimizer state, and CPU RNG.

The model was a randomly initialized 3,696-parameter GPT-2 with a 16-token vocabulary, one layer, 16 hidden units, and two heads. Two one-token responses to a single prompt received different deterministic rewards. The test ran the pinned GRPO `train_step`, confirmed both model and optimizer mutation, then raised a controlled exception before VARE could snapshot or promote the candidate.

| Runtime measure | Result |
| --- | ---: |
| Python / PyTorch / Transformers | 3.12.12 / 2.9.1 / 4.57.3 |
| Device / PyTorch threads | CPU / 1 |
| Wall time | 2.981 s |
| CUDA initialized | No |
| Paid compute / pretrained weights / dataset | None |
| Outcome | 11/11 checks passed |

The [run bundle](../results/rvl-grpo-partial-failure-v1/run-1/) retains the frozen protocol, validator output, runtime/provenance record, and SHA-256 manifest. Reproduce from a compatible local environment with the pinned upstream source:

```bash
python scripts/validate_rvl_grpo_partial_failure.py \
  --rvl-source /path/to/Recursive-Verification-Lag/src/rvl_systems
```

## Limits

This validates one trainer revision and one CPU adapter boundary with a tiny random model. It does not exercise a failure in the middle of the upstream optimizer kernel, production `HFLocalBackend` generation, another RVL revision, process loss, a pretrained model, learning quality, or downstream capability. The fake-backend rollout checks whether the restored model state matches the incumbent snapshot; it is not an external human reproduction.
