# MLX-LM #1904 v11 calibration — run 17

**Status: PASSED (E1, narrow historical function-contract calibration).** This was a Train-task grader and isolation check using the pinned baseline and known fix. It was not a model rollout, coding-agent success, policy update, or capability result.

## Frozen setup

- VARE implementation commit: `36af6f17908b50cb53bb159013eecfa8b0263da0`; worktree clean at run start.
- Protocol: `mlsys-repository-task-mlxlm-1904-v11`, SHA-256 `d7e5680faf5c7b15f32d7e432ee62ce8ddcdf6b835b27cc964988ce6294c2b77`.
- Source: MLX-LM baseline `2af5f7ce8620115bf37fbbddf9817f083118df59`; known fix `caed1943d38ecb58f6e466a72bed98310c9d45a5`.
- Runtime: macOS arm64, Colima Docker server `29.5.2/arm64`, image `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`; candidate container limits were 512 MiB, 1 CPU, 64 PIDs, and 5 seconds.

## Results

- Baseline: **FAIL**, as expected. `dict_arguments` returned `wrong_output`; the JSON-string control passed.
- Known fix: **PASS** on both the dict-argument case and JSON-string control. The staged source SHA-256 matched the blob read from the pinned fixed revision: `361a790e0743e50e66cea6509e64be8b8b67c6612e2274331624cbd244b29ead`.
- Fixed-source checkout swap: **rejected**.
- Red-team v2 suite: **9/9 probes passed**. This includes expected rejections of forged grader output, memory exhaustion, and timeout; isolation checks passed for grader paths, Git history, environment, network, subprocess writes, and a symlinked `mlx_lm` parent.
- The parent-symlink probe confirmed rejection before `grade_calls` was invoked (`workspace_parent_symlink_rejected=true`, `grade_call_attempted=false`). The original final-component-only check had accepted a symlinked parent in a pre-fix local reproducer.
- Total wall time: 9.66 seconds. Container peak memory was not measured.

## Decision and limits

Protocol v11 is **PASSED** for this bounded calibration. The regression fix reads each path component with no-follow directory/file descriptors, rejects a non-regular target, then grades the exact byte snapshot whose hash is recorded. Historical v1 red-team receipts remain verifiable under their original eight-probe schema; new v2 receipts require the parent-symlink probe.

The results do not certify a general-purpose sandbox, reproduce the full upstream HTTP/model-serving test, establish a coding-model repair, or demonstrate SFT/RL/policy improvement. No model inference, training, Development evaluation, or Held-out access occurred.

Raw machine records are `calibration.json` and `calibration.red-team.json`. The frozen protocol copy is `snapshots/protocol-v11.json`; per-file hashes are in `SHA256SUMS` and the run manifest.

## Reproduction

From a clean VARE checkout containing the v11 implementation and protocol, with the pinned image available in Colima and a clean `mlx-lm` mirror containing both source commits:

```bash
git -C ../mlx-lm-source worktree add --detach \
  "$PWD/../mlx-lm-baseline-1904" \
  2af5f7ce8620115bf37fbbddf9817f083118df59
mkdir -p ../colima-tmp
TMPDIR="$PWD/../colima-tmp" PYTHONPATH=src DOCKER_CONTEXT=colima \
  work/ci-py311/bin/python scripts/run_mlxlm_1904_calibration.py \
  --source-root ../mlx-lm-baseline-1904 \
  --output results/mlsys-repo-eval-v1/run-17/calibration.json
```

The runner refuses to overwrite either result JSON. Use a new run directory for a replay.
