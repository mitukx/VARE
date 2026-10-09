# Trainer runner development attempts

The following harness-only failures occurred before the source-pinned replay and are excluded from its result:

1. The first collator call had one list nesting level too few; the production collator iterated sample dictionary keys as if they were rows.
2. The manual AsyncGRPOTrainer instance lacked `rollout_worker` and `weight_transfer`, which the class's `_inner_training_loop` wrapper inspects.
3. The manual instance lacked `_step_microbatches`, read by the trainer's production `training_step` override.
4. The first gradient capture read `parameter.grad` after the optimizer step had cleared it; the runner now records gradients immediately after `training_step`.

After these preflight corrections, the runner passed the frozen one-step Trainer protocol. These failures do not alter the frozen fixture or acceptance thresholds.
