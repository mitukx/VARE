# RVL–VARE trajectory objective parity (CPU)

Status: completed engineering validation, protocol v5.
Evidence tier: E0 integration correctness. This is not an RL algorithm result, model-quality result, or RSI result.

## Question

Does a complete grouped replay batch pass from the pinned RVL TokenReplay, through VARE's read-only replay reader and GRPO conversion, into the trajectory reference materializer without changing RVL's loss or parameter gradient?

The fixture deliberately contains two distinct VARE replay groups with the same prompt_id. It uses four completions of lengths 1, 3, 2, and 4 and per-group rewards [0, 1] and [0.25, 0.75]. The tiny CPU model has 1,024 parameters. Behavior log-probabilities are calculated from that same frozen model and temperature, so the primary run tests exact objective wiring at ratio one, away from clipping boundaries.

## Why the weight conversion matters

At a fixed advantage vector, RVL's trainer computes

    L = -(1/N) Σᵢ (1/Tᵢ) Σₜ min(rᵢₜ Aᵢ, clip(rᵢₜ) Aᵢ).

One trajectory group materializes a token weight 1/(n_g Tᵢ). When multiple groups are combined in one RVL batch, each local weight must be multiplied by n_g/N; this yields 1/(N Tᵢ) and makes the loss and gradient comparable. The protocol asserts both each group's materialized weights sum to one and the batch-scaled weights sum to one.

The bridge itself continues to emit group-local weights. The batch conversion is done in the independent parity oracle because RVL's trainer averages the complete sample batch.

## Frozen result

| Measure | Result |
|---|---:|
| RVL revision | c7e646b043cb56e5ea3c2623bb8a61e065451f72 |
| Runtime | macOS arm64, Python 3.12.12, PyTorch 2.9.1, CPU, 1 thread |
| Loss absolute difference | 1.4901161193847656e-08 |
| Maximum parameter-gradient absolute difference | 7.450580596923828e-09 |
| Prompt-ID-merged negative-control gradient difference | 0.0819641649723053 |
| Native gradient norm | 0.44544702768325806 |
| Intercepted optimizer calls | 1, counted no-op |
| Parameter update | none |
| Model/optimizer/RNG invariants | all passed |
| Wall time | 2.59 s |

The groupwise advantages were approximately [-0.999998, +0.999998] and [-0.999992, +0.999992]. Merging the two groups by shared prompt ID instead yields [-1.264907, +1.264907, -0.632454, +0.632454]. The negative-control scalar loss is near zero too, so the relevant control measure is its gradient difference, not its loss difference.

The runner executes the actual pinned RVL train_step forward/backward code. It replaces optimizer.step with a counted no-op, and then checks model parameters and buffers, optimizer state, CPU and Python RNG, and restored per-module training modes. Gradient buffers are diagnostic outputs and are cleared after the checks; they are not part of the unchanged-state claim.

## Reproduction

The run is offline and does not load pretrained weights, call a paid API, or use GPU/MPS:

    PYTHONPATH=src python3.12 scripts/experiments/rvl_trajectory_gradient_parity_v5.py \
      --rvl-git /path/to/Recursive-Verification-Lag

The local Git object database must contain the pinned RVL commit. The runner extracts that commit to a temporary directory and verifies six RVL source files, four VARE source files, and its own hash against the frozen lock. The output JSON records runtime, fixture, source hashes, loss and gradient differences, state invariants, and the no-op optimizer call.

Protocol: protocols/rvl_trajectory_gradient_parity_v5.lock.json
Runner: scripts/experiments/rvl_trajectory_gradient_parity_v5.py
Result: outputs/rvl_trajectory_gradient_parity_v5.json
Gate audit: `PYTHONPATH=src python3.12 scripts/experiments/audit_rvl_trajectory_gradient_parity_v5.py`; recorded output: outputs/rvl_trajectory_gradient_parity_v5.audit.json

The earlier v4 result and audit remain in `outputs/` as historical records, but v4's source lock predates the group-identity and read-time reward-snapshot fixes, so its result cannot be reproduced against the current source. Protocol v5 freezes the corrected source and is the current reproducible result. Runner setup failures for v1–v3 are summarized in outputs/rvl_trajectory_gradient_parity_preflight_failures.json. They did not modify model weights. V1 and v2 stopped before forward/backward; v3 reached backward and the intercepted no-op, then stopped because the lock lacked tolerance fields. No result from those runs is used as evidence.

## What this establishes

This demonstrates a narrow interoperability property: for complete one-turn replay groups, VARE's exact token-preserving bridge and per-group advantages reproduce the pinned RVL objective and gradients on a deterministic CPU fixture. The negative control demonstrates that merging distinct groups which share a prompt ID changes the gradient in this fixture.

It does not establish the authenticity of replay rewards, general correctness for multi-turn/tool/sub-agent trajectories, policy improvement, task success, scaling behavior, or external reproduction. Rewards are synthetic. RVL's SQLite hash semantics do not prove reward provenance. The experiment does not validate the sampler's complete effective configuration.

The result is therefore a concrete training-data/debugging artifact for review, not a claim of a new RL algorithm. Existing trajectory-tree, shared-prefix, and execution-weighting ideas are not claimed as new; QwenGyre already covers those directions. HarnessFix also studies trace provenance and harness-failure diagnosis, so the distinction here is the exact RVL-to-VARE loss/gradient contract rather than general trace representation or harness repair.

## Verification

The VARE suite excluding two Math500 grader files that need external fixtures passed. The suite includes trajectory-reference and RVL bridge regressions. Python 3.12 compileall and git diff --check passed.
