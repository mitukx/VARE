# VARE GRPO adapter: shared-module mode rollback

## Finding

`RVLGRPOHooks` snapshots each named module's `training` flag and reapplies it after `trainer.restore_training_state()`. The previous restore walked `named_modules()` and called `Module.train(saved_mode)`. `named_modules()` lists a shared module once. `train()` still recurses through every parent. When a later aliased parent has a different saved mode, that recursive call overwrites the shared child's flag.

This is the same mechanism already reproduced on pinned RVL `27ebf7f` and corrected in open [RVL PR #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92) by assigning `module.training` directly. The VARE adapter applies its own mode map after the trainer restore, so a trainer that already assigns flags directly does not close the adapter path. The earlier adapter report restored modes with the recursive walk; that result remains valid for the non-shared fixture it measured. See [module-mode rollback v2](rvl-grpo-module-mode-rollback-v2-report.md).

## Regression

The test builds a small module graph with PyTorch's traversal rules: each object appears once in `named_modules()`, and `train()` still visits every parent edge. A stand-in forward returns its input in eval mode and zero in train mode. It is not a Dropout kernel and not a model forward. The trainer restore assigns flags directly, matching the patched RVL snapshot. `train_candidate` then runs one step that sets the whole graph to train mode and raises. Rollback must restore the incumbent flags, the stand-in forward, and the trainer weight, without calling `train()` again.

| Graph | Old adapter | Direct flag assignment |
| --- | --- | --- |
| Shared Dropout registered under `transformer` first, then under an alias whose saved mode differs | Shared child restored to train. Stand-in forward changed. `train()` was called again. | Saved flags, stand-in forward, and weight match. |
| Same shared child, alias registered first | Flags already matched. `train()` was still called again. | Flags match and `train()` is not called. |
| Nested parent/child, no sharing | Flags already matched. `train()` was still called again. | Distinct parent and child flags match and `train()` is not called. |

The old-adapter log is [`baseline-pytest.txt`](../results/rvl-grpo-shared-module-mode-v1/baseline-pytest.txt) (SHA-256 `25becbde62d8b87bd67762b7a333276a4b28a2057926f851bcad676569130a2c`). The same test after the assignment change is [`fixed-pytest.txt`](../results/rvl-grpo-shared-module-mode-v1/fixed-pytest.txt) (SHA-256 `c5b900fcfe85175112f0984dfbb2d1398b0b866b9271cc32a6f756fdcd24d831`): 18 passed, including the three graphs and the adjacent RVL hook, promotion, group-boundary, evaluation-identity, and rollout-binding tests. Python 3.12.3, pytest 8.4.2, no PyTorch, no model weights, no GPU.

## Change

Restore writes the captured boolean onto `module.training`. It does not call `Module.train()`. Custom `train()` overrides therefore do not run on rollback. The snapshot never stored their other side effects, which is the same limit recorded for the RVL patch.

## What this does not establish

The graph is constructed. It does not show that a Qwen, Gemma, or other production module tree shares a child across parents, that a real run hit the overwrite, or that the assignment changes task success or model quality. CUDA, distributed rollback, and optimizer-kernel faults were not run. No training experiment was started.
