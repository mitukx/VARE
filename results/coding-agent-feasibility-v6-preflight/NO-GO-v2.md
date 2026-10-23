# Final Qwen3-4B Development resource decision

Date: 2026-10-21 (Asia/Tokyo)
Implementation commit: `3500b035395c969489624a9f5238c5d3626841ff`

**Decision: NO-GO before task inference.** The pinned Qwen3-4B checkpoint and MLX Metal device were verified, and available memory was 8.31 GiB. Free swap was 440,926,208 bytes (420.9 MiB), 95,944,704 bytes below the fixed 512 MiB minimum. The protocol-freeze command from the clean implementation commit returned the resource-floor error and wrote no frozen protocol files.

Task rollouts: 0. Agent successes: 0. Reward measurements: 0. Optimizer updates: 0. Held-out evaluations: 0. Phase A is unevaluated, and SFT/GRPO are NO-GO. Prior task calibration evidence remains separate and does not count as model behavior.

The recorded MLX non-task smoke test showed GPU execution and 52.49 decode tok/s, but free swap fell to 281,479,741 bytes afterward. This gives evidence that the current resource pressure is material; the smoke test is not a coding task rollout. Do not lower the threshold or infer capacity from available RAM alone. Re-audit and freeze only after the machine independently satisfies both resource floors.
