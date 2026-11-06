# Attempt 009: real MLX continuation stopped on an untrusted grade

- Frozen assignment: prospective Train indices 42–49 on the already qualified pydantic-14138 Train task. Prompt variant v3 narrows the read scope to GenerateSchema._union_schema; the two qualified Development tasks remained sealed.
- Before prompt exposure, all 20 exact-head checks passed on ac0648973185166081377eba5f8ebe490a0b5690: push CI #2475, PR CI #2476, CPU evidence #936, optimizer/recovery E2E #462. The run used M1 Pro 32 GiB, MLX Metal, mlx 0.32.4, and mlx-lm 0.32.0.
- Six trajectories completed (indices 42–47), preserving token IDs, per-token behavior log-probabilities, and tool traces: 9,880 action tokens over 96 tool turns. Indices 42–46 received five independently trusted candidate-failure grades, all reward 0.
- Index 47 completed 16 tool turns and changed only its isolated candidate workspace. Its own compile_file tool reported a SyntaxError at line 1359. The independent grader exited 4 without exact-selector failure evidence; this is untrusted and receives no reward. The raw grader command and log remain private; their hashes are retained.
- The runner stopped at that trust failure. Indices 48–49 were never exposed. No complete eight-sample GRPO group was formed, no optimizer update or checkpoint reload occurred, and Development evaluation did not run.
- The frozen held-out NO-GO hash remains f91744b7e813ccaaff3396a63c639a63b8c364c61a97b361093a7a5b1c991ce4; eligible held-out tasks remain 0/15. No held-out task was assigned, accessed, or evaluated.
- The failure is not reported as learning or Long-Horizon RL. Five trusted zero rewards and one untrusted grader outcome are preserved separately.

The next source head adds a task-specific patch-scope check that rejects edits outside the for arg in args loop before mutation and returns the loop boundary in the tool error. It preserves the grader trust rules and gives the model a chance to correct an out-of-scope patch. That correction must pass exact-head CI before a fresh allocation can expose another prompt.
