# MLX-LM #1904 calibration run 10

- Protocol: v5, locked implementation hashes; frozen task semantics, baseline, fix, and v4 protocol remain unchanged.
- Implementation revision: `7b8be673c6927b93efee7aac407e182d116f216c`; tracked worktree clean at run start. All six protocol-locked implementation hashes were verified before grading.
- Pinned source: MLX-LM `2af5f7ce8620115bf37fbbddf9817f083118df59`; target SHA-256 matched v5. The baseline failed dict arguments and retained the JSON-string control. Pinned fix `caed1943d38ecb58f6e466a72bed98310c9d45a5` passed both cases and matched its exact source hash. Fixed-source substitution was rejected.
- All eight protocol-required red-team probes completed with the expected outcomes. The calibration record contains the raw red-team result SHA-256 and the bundle manifest binds raw and implementation snapshots.
- Runtime: macOS arm64 / Apple M1 Pro, Colima Docker 29.5.2/arm64, digest-pinned Python 3.12 image; network disabled, candidate source read-only, UID 65534, 512 MiB, 1 CPU, 64 PIDs, 5 s timeout. Total measured wall time was 10.49 s. Peak container memory was not measured.
- Claim boundary: narrow E1 known-fix grader calibration. No model or agent rollout, Development task certification, or model improvement is established.
