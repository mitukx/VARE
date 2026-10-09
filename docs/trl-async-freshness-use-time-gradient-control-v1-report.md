# TRL AsyncGRPO use-time freshness gradient control v1

## Question

In the previously observed GA=1 schedule, does the version-0 row that reaches the second actual Trainer training step at version 1 contribute a nonzero update to a controlled scalar objective? The frozen control zeros only that stale row's loss gradient while preserving its batch and the optimizer-step schedule.

This is an intervention on a synthetic scalar Trainer objective. It is not a proposed AsyncGRPO fix and does not establish that TRL promises freshness at training-step use.

## Frozen protocol and provenance

The protocol was frozen in commit `a06fc661af90e1c41461ef57688e23d02d860318` before the confirmation run. See [`protocols/trl_async_freshness_use_time_gradient_control_v1.lock.json`](../protocols/trl_async_freshness_use_time_gradient_control_v1.lock.json). It pins TRL base and candidate source hashes, the reproducer hash, Transformers 4.57.3, and Accelerate 1.12.0. The run used `TrainingArguments(use_cpu=True)`, one scalar parameter, SGD with learning rate 0.001, token budget 2, `max_staleness=0`, and accumulation steps 1 or 2. No model weights, GPU/MPS, paid API, or external spend were used.

The arms are `queue_admission_only` and `mask_stale_signal_at_training_step`. In the second arm, only a row with positive policy-version lag receives a zero loss multiplier; rows, batches, and optimizer-step counts are retained. This is deliberately a simple counterfactual, not a group-aware drop policy.

## Result

The baseline and local candidate runs matched exactly for both arms; all four configurations reported CPU and exited 0.

| Accumulation | Arm | Row 11 training version / lag | Raw gradient | Used gradient | Parameter after relevant update |
|---|---|---:|---:|---:|---:|
| 1 | Queue admission only | 1 / 1 | 16 | 16 | 0.0985 |
| 1 | Stale signal masked | 1 / 1 | 16 | 0 | 0.0990 |
| 2 | Queue admission only | 0 / 0 | 16 | 16 | 0.0990 |
| 2 | Stale signal masked | 0 / 0 | 16 | 16 | 0.0990 |

The gradient columns are autograd derivatives of the returned loss **before** Trainer gradient clipping and accumulation normalization. Trainer's default `max_grad_norm=1` clips the scalar gradient, and the learning-rate scheduler changes the second update's effective learning rate. Therefore the measured parameter difference (`0.09849999845` versus `0.09899999946`, approximately `−0.000500001`) is the effect under the full recorded Trainer schedule, not the result of applying an unclipped gradient of 16 with constant-rate SGD. At GA=2 both rows are used before the policy version changes, so the arms match. The first GA=1 row moves the parameter from approximately 0.1000 to 0.0990 in both arms.

Raw outputs and commands are retained in [`results/trl-async-freshness-use-time-gradient-control-v1/run-1/`](../results/trl-async-freshness-use-time-gradient-control-v1/run-1/). Reproduction commands use the pinned local source and dependency paths in `commands.txt`; the frozen script hash is recorded in the protocol.

## Interpretation and limits

This establishes a narrow causal effect inside the controlled scalar objective: the stale row has a nonzero autograd gradient and, under the defined zero-gradient counterfactual, its contribution changes the scalar parameter by 0.0005 in this schedule. It also verifies that the same script behaves identically against the pinned TRL base and the local atomic-admission candidate.

It does **not** establish:

- a violation of a documented TRL contract: dequeue freshness versus training-use freshness remains semantically unresolved;
- a correct way to filter stale rows in GRPO, where dropping or masking a member can change group-relative advantages and normalization;
- a policy-gradient or language-model effect, finite-sample benefit, prevalence in ordinary runs, or downstream task/capability improvement;
- a fix, upstream PR, or independent external reproduction.

The exact parameter difference is specific to the chosen synthetic loss and optimizer schedule. It should not be generalized to real GRPO update magnitude or direction.

## Decision

**Stop the candidate-as-fix claim.** The control shows that this particular stale sample can affect this particular toy update, but does not settle what AsyncGRPO freshness should mean or how a valid group-aware correction should work. Do not extend the queue patch or propose it upstream on this evidence. The next useful step is to clarify the intended contract from upstream documentation/discussion and, only if training-use freshness is required, design a group-preserving executable reproducer that measures a real GRPO loss/gradient. The broader highest-value gap remains independently graded task success after an actual model update; this study does not close it.

## Independent audit and test status

A same-host independent agent audit verified all protocol-pinned hashes, exact equality of baseline and candidate JSON, four CPU/zero-exit configurations, the arm schedule, and the numerical claims. It confirmed that gradient 16 is pre-clipping and that the parameter delta reflects clipping plus the scheduler. The audit record is retained in the raw bundle. This was not an outside-human reproduction. No project-wide test suite was run because this experiment changes no VARE implementation code.
