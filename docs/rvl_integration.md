# Recursive-Verification-Lag integration

The existing `mitukx/Recursive-Verification-Lag` repository owns the lower-level substrate: async/token-exact rollout, verifier execution, GRPO, durable replay, verification debt, weight publication and GPU experiment harnesses. VARE sits above that stack.

Recommended split:

- **RVL:** rollout/training/verifier execution, immutable behavior data, replay leases, weight sync, distributed/GPU evidence.
- **VARE:** failure discovery, curriculum allocation, current-freshness screening, verifier policy, experiment selection, independent held-out promotion and evidence governance.

## Integration surfaces

1. `RVLTokenReplayReader` opens RVL `TokenReplay` SQLite in read-only mode. It preserves group `policy_version`, `verifier_version`, behavior logprobs, reward provenance and replay status. VARE never rewrites RVL replay rows.
2. `RVLOuterPlanner` converts current replay state into freeze/throttle/verifier-refresh controls and task-family weights. Data outside current freshness bounds can trigger a systems intervention but cannot boost curriculum weight.
3. `RVLGRPOHooks` directly wraps RVL `HFLocalBackend` and `HFCausalLMGRPOTrainer` for transactional candidate updates. It requires token-exact generation metadata and restores the incumbent until VARE's held-out gate accepts the candidate.

Candidate training invalidates the shared runtime's loaded-policy marker before the first in-place update. If `train_step`, candidate snapshotting, or the first incumbent restore raises, the adapter removes the candidate snapshot and attempts to restore the saved incumbent before releasing the model lock. A deterministic fake-trainer regression partially mutates both policy and optimizer state, raises, and checks that a subsequent rollout still uses the incumbent. If rollback itself fails, the adapter raises an explicit unknown-state error; it does not claim the runtime is safe. This test covers the adapter contract only, not the behavior of a real RVL trainer or process/power-loss recovery.

VARE re-screens replay freshness **at training time**. Insertion-time freshness is not trusted after policy or verifier versions advance.

The critical question is whether these adapters preserve trustworthy learning and independent evaluation when used in a post-training experiment. The existing comparison is locked in `protocols/l2_rvl_qwen_v1.lock.json`; it requires a 0.5B model and a three-arm, three-seed run and remains unrun under the current no-spend CPU budget. Do not change that lock or present the hooks as training evidence. The next affordable experiment is the synthetic preference-policy control described in [`post-training-plan.md`](post-training-plan.md), with held-out preference metrics, explicit policy drift, per-seed results, and exact compute reporting. A toy result cannot establish model capability.
