# MLX-LM #1904 calibration run 12

- Protocol: v7; implementation revision `e6690c6035bb19637c9d999119afe5642cadf6ad`; clean tracked worktree required and confirmed at start.
- All eight locked grader, candidate runner, Docker isolation, red-team, and test artifacts matched protocol SHA-256 values. The exact protocol-pinned image was used; alternate image digests are rejected.
- Baseline failed the dict-arguments case and passed its JSON-string control. The pinned known fix matched its target source hash and passed both cases. A fixed-source swap was rejected.
- All eight red-team probes completed with expected outcomes and raw output hash `ce50dd950be8535eeda2cc4a1e15c22cffbb7116c107ca2173312f179d3b1eec`.
- Runtime: Apple M1 Pro/macOS arm64, Colima Docker 29.5.2/arm64, network disabled, source-only read-only candidate mount, UID 65534, 512 MiB, 1 CPU, 64 PIDs, 5 s case timeout. Elapsed `9.31` s; peak container memory not measured.
- Claim boundary: narrow E1 known-fix calibration of the frozen message-normalization function contract. No agent rollout, Development evaluation, or model improvement is established.
- Calibration decision: PASS.
