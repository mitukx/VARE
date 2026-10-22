# MLX-LM #1904 v10 calibration — run 16

- Protocol: `mlsys-repository-task-mlxlm-1904-v10`; frozen protocol v9 and runs 1–14 remain unchanged.
- Implementation commit: `c21ad2ff72d086d922020cdeb4f8277135a9895e`.
- Host: macOS arm64 / Apple Silicon; Docker context `colima`; Docker server `29.5.2/arm64`.
- Image: `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`.
- Source: MLX-LM baseline `2af5f7ce8620115bf37fbbddf9817f083118df59`; fixed revision `caed1943d38ecb58f6e466a72bed98310c9d45a5`.

Result: baseline FAIL (`dict_arguments` wrong output); exact known fix PASS; fixed source bytes match the v10-pinned SHA-256; source-revision swap rejected; all 8 red-team probes passed. The new `refs/replace` regression test also confirms fixed-source verification ignores replacement refs and rejects a hash mismatch.

This is an E1 historical grader calibration using the known fix. It is not an agent rollout, model result, or policy improvement. Docker memory was capped at 512 MiB; peak container memory was not measured. The first v10 attempt (run 15) failed closed because the source mirror was not checked out at the required baseline; it ran no grading arms and is retained separately.

Reproduce from a clean checkout at the implementation commit. The source mirror must have both pinned commits and the source-root worktree must be clean at the frozen baseline. Use a fresh output path because the runner refuses to overwrite evidence:

```bash
git -C "$HOME/.cache/vare/source-mirrors/mlx-lm" worktree add --detach \
  "$HOME/.cache/vare/source-mirrors/mlx-lm-baseline-2af5f7" \
  2af5f7ce8620115bf37fbbddf9817f083118df59
DOCKER_CONTEXT=colima PYTHONPATH=src python scripts/run_mlxlm_1904_calibration.py \
  --source-root "$HOME/.cache/vare/source-mirrors/mlx-lm-baseline-2af5f7" \
  --output /tmp/vare-mlxlm-1904-calibration-replay.json \
  --docker-context colima
```
