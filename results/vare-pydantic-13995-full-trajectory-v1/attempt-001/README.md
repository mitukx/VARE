# Real MLX Train attempt 001

This is a real pretrained-model, multi-turn coding-agent rollout on the prospectively qualified Train task #13995. It is a negative result: the eight complete trajectories received trusted reward zero, so the frozen GRPO protocol correctly skipped optimizer mutation. No checkpoint was written and no Development evaluation was run.

- Code head: `3e96be20502566bdb44658eb7195efccdc5882ce`
- Exact-head CI: 20/20 passed (push #2485, PR #2486, CPU evidence #941, optimizer/recovery #467).
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, MLX Metal on M1 Pro 32 GiB.
- Group: eight trajectories, 3,764 action tokens, 128 tool turns. Token IDs and finite sampling-time behavior log-probabilities are retained in `trajectories.jsonl.gz` and `training-samples.jsonl.gz`.
- Independent pinned offline grader: 8/8 trusted candidate failures, reward 0; no source changes or patch submissions.
- Optimizer: 0 updates because all trusted rewards were constant. No checkpoint/reload or Development comparison.
- Failure trace: each rollout searched once with a phrase that had no exact literal match, then spent the remaining 15 turns paging through 20-line source windows and exhausted the turn budget.
- The full-trajectory terminal-reward path was configured but no optimizer step ran. This is not a completed Long-Horizon RL result.
- The held-out NO-GO gate remains unchanged; no held-out task was accessed. Development task prompts remain sealed.

`inputs/` contains the frozen protocol, allocation, environment binding, run lock, pre-exposure preflight, and local model manifest. The task definition and raw grader commands/logs remain private. Grade records expose trusted outcomes and grader-log hashes only. Full raw run artifacts remain under the private experiment outputs.
