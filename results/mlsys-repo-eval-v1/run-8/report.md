# MLX-LM #1904 calibration run 8

- Source checkout: `2af5f7ce8620115bf37fbbddf9817f083118df59`; target SHA-256 matched the frozen protocol.
- Implementation revision: `527ee0e3d5ff12ce53a2854e100b9ba2f15aea3c`.
- Outcome: baseline failed the dict-arguments case while preserving the JSON-string control; pinned known fix passed both. Fixed-source swap was rejected. All eight required red-team probes completed and their raw result hash is bound into `calibration.json`.
- Runtime: macOS arm64, Colima Docker 29.5.2/arm64, pinned Python 3.12 image, network disabled, read-only candidate mount, 512 MiB / 1 CPU / 64 PIDs / 5 s. Candidate workspaces used a Colima-shared temporary root. Peak memory was not measured.
- `implementation_worktree_clean_at_start` is false because this runner creates the output directory before sampling Git status; the run output directory itself is untracked at that instant. The exact implementation is separately snapshotted below and the HEAD is recorded. This field-order issue is corrected in a follow-up source change and a fresh calibration.
- This is narrow E1 known-fix harness calibration only: it does not establish an Agent rollout or model improvement.
