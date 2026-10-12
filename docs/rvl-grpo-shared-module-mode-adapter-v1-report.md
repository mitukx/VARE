# VARE GRPO adapter: shared-module mode rollback

## Question

After a candidate update, does `RVLGRPOHooks` restore every captured module `training` flag when one child object is registered under two parents?

The same recursive-`train()` failure was already reproduced in RVL's trainer snapshot at `27ebf7f` and corrected in open [RVL PR #92](https://github.com/mitukx/Recursive-Verification-Lag/pull/92). VARE's adapter still applied modes by calling `module.train(saved_mode)` in `named_modules()` order. That preorder workaround restores a nested child that appears after its only parent. It does not cover a shared child: `named_modules()` yields the object once, and a later parent with a different saved mode calls `train()` and overwrites the child.

## Contract under test

The regression does not import PyTorch. `ContractModule` copies the two behaviors the RVL review measured on `nn.Module`:

- `named_modules()` yields each module object once;
- `train(mode)` sets that module's flag and recurses into every registered child.

The shared graph saves `left` and the shared child in eval mode and the later parent `right` in train mode. `train_step` calls `model.train(True)` before returning, which the probe records as the shared child being in train mode during the update. Incumbent rollback must then match the saved flags and the pre-update step counter. Promoting the candidate must install the post-update snapshot, where every saved module is in train mode and the step counter is 1.

A second graph has distinct parent and child modules with different saved modes. That case passed under the previous preorder `train()` restore and still passes after direct flag assignment. It is the non-alias control for the earlier [adapter mode report](rvl-grpo-module-mode-rollback-v2-report.md).

## Result

Before the assignment change, `test_rvl_grpo_rollback_preserves_shared_child_module_mode` failed after rollback: `left.drop` was `True` against the saved `False`. The optimizer-step counter had already been restored to 0, and the probe had observed `True` during `train_step`. The distinct nested-mode test passed. The pre-change transcript is [`baseline-failure.txt`](../results/rvl-grpo-shared-module-mode-adapter-v1/baseline-failure.txt).

The adapter now writes each captured boolean onto `module.training` after the trainer state restore. The same two tests then passed. The focused RVL adapter files passed 23 tests. The post-change transcript is [`after-fix.txt`](../results/rvl-grpo-shared-module-mode-adapter-v1/after-fix.txt).

The CI-equivalent suite, excluding the two math500 grader files, was then run on Python 3.12.12. This image has `python3` and no `python` executable. Without a `python` entry on `PATH`, eight environment tests failed with `FileNotFoundError: python`. With `python` linked to `python3`, the same command finished with 297 passed and 10 skipped. That link was local to the process and is not part of the change. The separate math500 grader job was not run. The suite note is [`full-suite.txt`](../results/rvl-grpo-shared-module-mode-adapter-v1/full-suite.txt).

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_rvl_grpo_hooks.py::test_rvl_grpo_rollback_preserves_shared_child_module_mode \
  tests/test_rvl_grpo_hooks.py::test_rvl_grpo_rollback_preserves_distinct_nested_module_modes
```

## Limits

This is an adapter rollback regression on a constructed module graph. It does not execute `torch.nn`, an active Dropout forward, a real optimizer, CUDA, or a distributed trainer. It does not show that a default Qwen module graph contains this alias, and it does not measure task success or model quality. Direct flag assignment restores the boolean stored in the snapshot. It does not replay side effects from a custom `Module.train()` override; the snapshot does not record those effects. RVL PR #92 remains a separate trainer-owned snapshot change and is not closed by this adapter fix.
