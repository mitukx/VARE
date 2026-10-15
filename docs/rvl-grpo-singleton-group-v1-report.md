# RVL GRPO singleton-group guard

## Finding

At VARE base `093a45d`, `EngineConfig.samples_per_task` defaulted to `1`. The
engine records that size as a complete rollout group. In `RVLGRPOHooks`, group
relative centering of one reward against its own mean produces advantage
`0.0`; the hook nevertheless called `train_step` and could snapshot a candidate.
This is a zero reward-directed signal, though an optimizer may still make
non-reward changes such as weight decay.

## Correction

The RVL GRPO adapter now raises when a declared group has fewer than two
responses, before acquiring the model lock or calling the trainer. Generic
`EngineConfig` retains its one-sample default for non-GRPO integrations. RVL
GRPO callers must set `samples_per_task >= 2` and provide a complete group.

The regression verifies that singleton input raises without an optimizer step.
The end-to-end promotion fixture now uses a two-response group with one passing
and one failing reward, and checks that each advantage has the corresponding
sign. Before the guard, the new singleton regression reproduced the trainer
call with a zero advantage; with the guard, it fails closed.

## Validation and limits

The focused adapter, group-boundary, replay, reward-attestation, and promotion
suite passes 36 tests on CPU. Ruff lint and Python compilation pass. This is E0
harness/signal-path evidence only: no model was trained, no capability gain was
measured, and this does not show that any prior run used the default singleton
configuration. It also does not establish that a group of two provides a
statistically reliable gradient; it only prevents the degenerate one-member
case.
