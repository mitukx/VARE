# RVL GRPO verifier-to-trainer prompt alignment v1

## Question

When a backend returns generation metadata for a different prompt, can the reward computed from the dispatched task reach the GRPO trainer alongside that mismatched generation?

## Frozen protocol

`protocols/rvl_grpo_verifier_trainer_prompt_alignment_v1.lock.json` freezes the CPU-only dataflow check before validation. The fixture dispatches `task-a` / `prompt a`, while a deterministic fake backend returns `task-b` / `prompt b` with prompt token ID `[202]` and response token ID `[7]`. A score function assigns the response reward using the outer `Attempt.task`. The test requires the fixed adapter to reject the mismatch before the trainer's `train_step`.

## Baseline observation

At `d8cc153dd242d3b989efcc42eea79408fbbb75c7`, a direct replay passed the `Attempt` to a fake verifier function, then called the real adapter method `train_candidate` with that `Experience`. The fake trainer's `train_step` was called once. Its captured `VerifiedGeneration` carried task ID `task-b`, prompt `prompt b`, and token IDs `[202]`, paired with reward `1.0` computed from the outer task `task-a`. The retained JSON observation and baseline frozen-test failure are under `results/rvl-grpo-verifier-trainer-prompt-alignment-v1/run-1/`.

## Fixed result

The frozen regression passed after the adapter fix. It asserts that the mismatch raises `ValueError` before `train_step`; the fake trainer remains untouched. The previous [adapter task-binding report](rvl-grpo-rollout-task-binding-v1-report.md) describes the underlying identity check.

## Evidence boundary

This is stronger dataflow evidence than checking metadata alone, but it remains E0 harness correctness. The backend and trainer are deterministic fakes. No production backend, real RVL trainer, optimizer update, task-success metric, or model behavior was exercised. Thus it demonstrates a reachable verifier-to-trainer misalignment under a backend contract violation and that the adapter guard blocks that fixture; it does not establish that the production backend emitted such a mismatch or that a real policy update was corrupted.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rvl_grpo_verifier_trainer_prompt_alignment.py
```

The full GitHub Actions run for `abdf2774448c778c1f79fff11f870cb55cda35df`
[passed](https://github.com/mitukx/VARE/actions/runs/37877362999) on 2026-10-09,
including the full pytest suite, the exact GRPO witness and audit-order checks,
the frozen partial-audit study audit, and the demo. This confirms the repository
checks at that revision; it is not an external reproduction or evidence of a
real-model effect.
