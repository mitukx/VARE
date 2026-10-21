# MLX-LM #1904 calibration run 14

- Protocol: v9; implementation revision `7bd3ca06b4afbaf23c7c4b82d6c17186d2e877b7`; clean tracked worktree required and confirmed.
- All nine evaluator, candidate runner, Docker isolation, red-team, fixture, patch, and test artifacts matched protocol SHA-256 values. The candidate-visible task prompt hash matched; Git index flags were clear on the protocol and all calibration inputs; the exact runtime image was used.
- Baseline failed the regression case (`wrong_output`) and passed the JSON-string control. The pinned known fix matched its source hash and passed both cases. A fixed-source swap was rejected.
- All eight red-team probes completed with expected outcomes. Raw red-team SHA-256: `6e9d565e6a883a764815c3c1aa2abf5c15c9a7ef57659b6bb3dad9ae01cd9358`.
- Runtime: Apple M1 Pro/macOS arm64, Colima Docker 29.5.2/arm64, network disabled, source-only read-only candidate mount, UID 65534, 512 MiB cap, 1 CPU, 64 PIDs, 5 s per-call timeout. Total elapsed `10.34` s; peak container memory was not measured.
- Protocol v8 run 13 is retained as a failed environment attempt. Protocol v9 uses a canonical ephemeral candidate mount below the operator home cache, which is shared with Colima.
- Claim boundary: narrow E1 calibration for the frozen message-normalization function contract only. No Coding Agent rollout, Development score, SFT/GRPO update, or policy improvement is established.
- Calibration decision: PASS.
