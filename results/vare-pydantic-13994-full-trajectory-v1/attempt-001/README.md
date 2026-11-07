# Pydantic 13994 real-model Train result — attempt 001

This is an auditable negative result from a prospective Train task. The pinned pretrained Qwen3 4B 4-bit model ran locally with MLX Metal on an Apple M1 Pro with 32 GiB unified memory. The task was independently graded in the pinned offline container.

## Result

- Trusted rewards: 8 failures, 0 passes.
- Tool turns: 128; action counts: {"read_file": 100, "search_code": 28}.
- Generated token IDs: 3964; finite sampling behavior log-probabilities: 3964.
- Edit tool actions attempted: 0; source files changed: 0; all per-sample traces and trusted reward records are retained.
- GRPO optimizer updates: 0 (no_variable_trusted_rewards); no adapter update, checkpoint, reload, or Development evaluation occurred.
- Development prompts remained sealed. Held-out NO-GO remains unchanged at 0/15 eligible tasks; no held-out task was accessed, assigned, or evaluated.
- Full-trajectory terminal-reward credit assignment was configured, but no update ran. This does not establish Long-Horizon RL.

The complete sanitized trajectories retain policy prompt/completion token IDs, sampling-time behavior log-probabilities, tool actions and results, and hashes binding them to the local raw records. Absolute workstation paths, grader commands/output, and sealed Development identities/prompts are omitted. Independent trusted grade summaries are included; raw grader logs and selectors remain local.

## Frozen inputs and limits

Frozen experiment input hashes, the model revision and trainable parameter fingerprint, the exact integration head and its CI evidence, qualification arm outcomes, and the held-out gate hash are recorded under `inputs/`. Raw assignment, run-lock, and environment-binding bodies are not published because they contain sealed Development task identifiers. This result is a single eight-sample Train group; it makes no update, Dev improvement, promotion, or Long-Horizon RL claim.
