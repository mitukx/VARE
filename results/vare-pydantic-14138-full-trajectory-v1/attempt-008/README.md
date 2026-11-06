# Attempt 008: context replay worked; task-scope failure persisted

- Integration source head: f0883a70a564693a70eca84205bf01140562007e; exact-head CI: 20/20 checks passed before new-group prompt exposure.
- Model: mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b on M1 Pro 32 GiB with MLX Metal (mlx 0.32.4, mlx-lm 0.32.0).
- Fresh qualified Train indices 34–41; the same two qualified Development tasks remained sealed. No held-out task was assigned, accessed, or evaluated; the frozen NO-GO gate hash is unchanged.
- Eight 16-turn real model trajectories produced 5,697 action tokens over 128 tool turns. Token IDs, finite aligned behavior log-probabilities, and complete tool traces are retained in the compressed GRPO samples and per-sample trajectories.
- All eight pinned offline grades were trusted candidate failures (reward 0). No patch was submitted and no source file changed. GRPO correctly skipped mutation because reward variance was zero: zero optimizer updates, no checkpoint/reload, and no Development evaluation.
- The next blocker is task-scope control. Source replay restored covered pages, but samples35–38 read through line1619 after the target method ended; sample39 produced an invalid action; and the patch attempts in samples34, 40, and 41 all failed to apply. No source file changed. This is a real model-coupled negative result, not a completed Long-Horizon RL result.
- The initial freeze was rejected before model/prompt exposure because preflight used host Python 3.9 instead of the pinned MLX interpreter. Its files and correction addendum are retained. The corrected freeze passed with pinned Python 3.12 and MLX Metal before new-group prompt exposure.
- Per-sample trajectories and trusted grade JSON are included. Raw grader logs and commands remain in the private output directory; public copies omit hidden test-selector output while retaining exact log hashes and pinned grader identity.
