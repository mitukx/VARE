# MLX-LM #1904 calibration run 9

- Source checkout: `2af5f7ce8620115bf37fbbddf9817f083118df59`; `mlx_lm/server.py` SHA-256 equals the frozen baseline hash.
- Implementation revision: `96e8de1d30de934dbc65d2d533eff5dfec73f67c`; the tracked worktree was clean when the run started.
- Outcome: baseline failed the dict-arguments case and passed the JSON-string control. The pinned known fix passed both cases and matched the fixed-source hash; switching the source checkout to that fixed revision was rejected. All eight required red-team probes passed their audit outcomes, and `calibration.json` records the exact red-team bytes SHA-256.
- Runtime: macOS arm64 / Apple M1 Pro, Colima Docker 29.5.2/arm64; digest-pinned Python 3.12 image; candidate network disabled, read-only root and subject mount, UID 65534, 512 MiB, 1 CPU, 64 PIDs, 5 s timeout. Peak memory was not measured.
- Run 7 is retained as a failed attempt caused by Colima not sharing its temporary-directory root; run 8 is also retained and passed but recorded the worktree dirty because output creation preceded the status check. Run 9 uses a Colima-shared temporary root and records clean source state.
- Evidence level: narrow E1 known-fix calibration only. No Coding Agent rollout, development task certification, or model improvement is established.
