# Attempt 006 — real MLX Train rollout

This bundle records the v15 prospective Train/Development run on PR #199 head `4e03d2931ffe96e3f4e20624f0cea823cb7de439`, whose required exact-head CI passed 20/20 before Train prompts were exposed. The frozen NO-GO gate and held-out boundary remained in force. Only qualified Train task `pydantic-14138-direct-builtin-scalar-union-schema` was exposed; Development tasks remained sealed.

Eight real Qwen3-4B MLX trajectories (indices 18–25) produced 4,494 action tokens over 128 tool turns. Token IDs, finite aligned sampling-time behavior log-probabilities, complete tool traces, and offline independent grade attestations are included. All eight trajectories exhausted the 16-turn limit without submitting a patch. Trusted rewards were 0/8 positive, so GRPO correctly performed zero optimizer updates; no checkpoint or Development evaluation was produced. This is a preserved negative result, not a successful learning milestone.

The current reader advances only an identical repeated range. The traces show the agent expanding or overlapping requested ranges, which restarts the bounded reader at the first page. The next correction should track read coverage per source path for oversized requests and return the next unseen page.

The full local run, including grader workspaces, remains under the workspace `outputs/` tree. Duplicate grader baseline/test trees are omitted from this GitHub bundle. `training-samples.jsonl.gz` contains the complete JSONL sample rows; `summary.json` records both its compressed hash and the decompressed JSONL hash. It is compressed because the repository rejects any new tracked blob at or above 1 MiB. `SHA256SUMS` covers every file here except itself.
