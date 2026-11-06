# Pydantic #14115 real-model rollout — attempt 001

This is an auditable negative real-model result from a prospective Train task. The pinned pretrained Qwen3 4B 4-bit model ran locally with MLX Metal on an Apple M1 Pro with 32 GiB unified memory. The model received the copy/deep-copy-only task prompt and used the allowlisted coding tools. The independent pinned offline grader returned reward 0 for all eight trajectories.

The full group was retained. The frozen learner correctly skipped optimization because the trusted rewards had no variance. No adapter update, checkpoint, reload, or Development evaluation occurred. Development prompts remained sealed, and the held-out NO-GO gate (0/15 eligible tasks) was not accessed or changed.

## Measurements

- 8/8 trusted candidate failures; all eight runs reached the 16-tool-turn budget.
- 2,028 generated token IDs and 2,028 finite sampling behavior log-probability entries.
- 128 tool turns: 127 no-match source searches and one source read; no source edits.
- Independent grader records show exact-selector test failures, with no timeout or OOM.
- GRPO optimizer updates: 0 (no_variable_trusted_rewards); no checkpoint, reload, or Development comparison.
- Full-trajectory terminal-reward credit assignment was configured, but the update was skipped. This does not establish Long-Horizon RL.

The raw local run retains the complete trajectories and grader logs. This GitHub bundle keeps the on-policy token IDs and behavior log-probabilities, tool actions/results, and sanitized trusted-grade records; absolute workstation paths and raw test output are omitted. SHA-256 hashes bind the committed artifacts to the raw local files.

## Reproduction and limits

The Train task prompt, model manifest, qualification summaries, and Train environment lock are included. The exact frozen protocol, assignment, run lock, environment binding, preflight, run metadata, and task definition are recorded by SHA-256 in `inputs/frozen-input-hashes.json`; their Development task identities and grader selector details remain sealed in the raw local experiment record. The qualified prompt covers copying and deep-copying because the frozen selector did not test unpickling. The Development set is distinct from the held-out set and remained sealed. No held-out claim, promotion, or Long-Horizon RL claim is made.
