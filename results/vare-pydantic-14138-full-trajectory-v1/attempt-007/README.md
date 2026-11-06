# Attempt 007: coverage-aware pagination, context-loss negative result

- Integration source head: `3fbb5c82161824fb0b376a47e975331bbda3b61d`; exact-head CI: 20/20 checks passed before prompt exposure.
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` on M1 Pro, 32 GiB unified memory with MLX Metal.
- Prospective Train task: `pydantic-14138-direct-builtin-scalar-union-schema`, fresh indices 26–33. Development remained sealed; no held-out task was accessed.
- Eight full 16-turn trajectories, 4,024 generated/action tokens, 128 tool turns. All token IDs, finite aligned sampling-time behavior log-probabilities, tool traces, independent grader results, and logs are retained.
- All eight pinned offline grades were trusted candidate failures (reward 0); no patch was submitted and no source file changed. GRPO correctly applied zero optimizer updates because group reward variance was zero. No checkpoint or Development evaluation was produced.
- Read pagination advanced across overlapping/expanded ranges, but compacted context lost source text; already-read responses had no source excerpt. That real model-coupled failure motivates the next small context replay correction. This is not a completed Long-Horizon RL result.
- The initial v16 freeze with a stale tool-contract digest was rejected before prompt exposure; its files and rejection record are preserved under `preflight-rejected-freeze/`.

The complete local run, including grader workspaces, remains in the project outputs directory. This GitHub bundle retains per-sample trajectories, grades, grader commands/logs, the frozen assignment/protocol/environment binding/run lock, compressed GRPO samples, and hashes; duplicated grader workspace trees are omitted.
