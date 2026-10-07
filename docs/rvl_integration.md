# Recursive-Verification-Lag integration

The existing `mitukx/Recursive-Verification-Lag` repository owns the lower-level substrate: async/token-exact rollout, verifier execution, GRPO, durable replay, verification debt, weight publication and GPU experiment harnesses. VARE sits above that stack.

Recommended split:

- **RVL:** rollout/training/verifier execution, immutable behavior data, replay leases, weight sync, distributed/GPU evidence.
- **VARE:** failure discovery, curriculum allocation, current-freshness screening, verifier policy, experiment selection, independent held-out promotion and evidence governance.

## Integration surfaces

1. `RVLTokenReplayReader` opens RVL `TokenReplay` SQLite in read-only mode. It preserves group `policy_version`, `verifier_version`, behavior logprobs, reward provenance and replay status. VARE never rewrites RVL replay rows.
2. `RVLOuterPlanner` converts current replay state into freeze/throttle/verifier-refresh controls and task-family weights. Data outside current freshness bounds can trigger a systems intervention but cannot boost curriculum weight.
3. `RVLGRPOHooks` directly wraps RVL `HFLocalBackend` and `HFCausalLMGRPOTrainer` for transactional candidate updates. It requires token-exact generation metadata and restores the incumbent until VARE's held-out gate accepts the candidate.

VARE re-screens replay freshness **at training time**. Insertion-time freshness is not trusted after policy or verifier versions advance.

The critical experiment is not whether these adapters execute. It is whether VARE yields higher independent held-out capability gain per GPU-hour than fixed-curriculum/fixed-verifier baselines at matched compute. That comparison is locked in `protocols/l2_rvl_qwen_v1.lock.json`.
