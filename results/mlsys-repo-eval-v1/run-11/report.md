# MLX-LM #1904 calibration run 11

- Protocol: v6. Frozen task semantics and previous protocol/evidence remain unchanged.
- Implementation revision: `bcf92bd240567f5a5bcc64aba19e7675ce670098`; tracked worktree clean at run start. All eight protocol-locked artifacts were verified before grading, including the isolated candidate and Docker isolation executors.
- Runtime image: exact v6-pinned Python digest; alternate image arguments fail closed.
- Pinned baseline failed the dict-arguments case while the JSON-string control passed. The pinned known fix passed both cases and matched its source hash. Fixed-source checkout substitution was rejected.
- All eight red-team probes passed; raw probe output is preserved and hash-bound by the calibration.
- Runtime: macOS arm64 / Apple M1 Pro, Colima Docker 29.5.2/arm64, network disabled, candidate source read-only, UID 65534, 512 MiB, 1 CPU, 64 PIDs, 5 s timeout. Elapsed 9.45 s. Peak memory was not measured.
- Claim boundary: narrow E1 known-fix grader calibration only. This is not an agent rollout, development-task certification, or model improvement result.
