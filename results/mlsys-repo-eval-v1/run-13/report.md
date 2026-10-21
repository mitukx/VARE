# MLX-LM #1904 calibration run 13

- Protocol: v8; implementation revision `a39020d92870aa7b58d8208f458183bcdd0d10a5`; clean tracked worktree and all locked artifacts were verified before grading.
- The task prompt hash matched the frozen prompt. The pinned source and known-fix source hashes matched their recorded revisions, and a fixed-source swap was rejected.
- Calibration **failed closed**. Baseline and known-fix calls returned `missing_result`; consequently known-fix pass was false. Red-team calls also could not reach expected grader outcomes, so the suite did not pass and evidence was not accepted.
- Root cause: canonicalizing the temporary directory did not make it available inside Colima. The host's `/private/var/folders/...` temporary path is outside Colima's shared mounts; Docker reported `bind source path does not exist`. This is an execution-environment failure, not a baseline/fix result.
- Runtime: Apple Silicon macOS with Colima Docker 29.5.2/arm64, pinned image `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`. The failed attempt is retained unchanged.
- Claim boundary: no calibration pass is claimed from this run. Protocol v9 stages ephemeral candidate mounts below the operator home cache, which is shared with Colima, and requires a fresh run.
