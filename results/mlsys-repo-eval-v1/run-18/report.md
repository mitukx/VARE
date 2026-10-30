# MLX-LM #1904 v12 calibration — run 18

**Status: PASSED (E1, narrow historical function-contract calibration).** This is a Train-task grader and isolation check using the pinned baseline and known fix. It is not a model rollout, coding-agent success, policy update, or capability result.

## Frozen setup

- VARE implementation commit: `a7d39bda1da662c18bc7b6738b95f605b679ee30`; implementation worktree was clean at run start.
- Protocol: `mlsys-repository-task-mlxlm-1904-v12`, SHA-256 `36c5932ab46e00573535f1d6826396cc24cc5021703566f49b9169cad45460c4`.
- Source: MLX-LM baseline `2af5f7ce8620115bf37fbbddf9817f083118df59`; known fix `caed1943d38ecb58f6e466a72bed98310c9d45a5`.
- Runtime: macOS arm64, Colima Docker server `29.5.2/arm64`, image `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`; candidate container limits were 512 MiB, 1 CPU, 64 PIDs, and 5 seconds.

## Results

- Baseline: **FAIL**, as expected. `dict_arguments` returned `wrong_output`; the JSON-string control passed.
- Known fix: **PASS** on both controls. The staged source SHA matched the blob read from the pinned fixed revision: `361a790e0743e50e66cea6509e64be8b8b67c6612e2274331624cbd244b29ead`.
- Fixed-source checkout swap: **rejected**.
- Red-team v3 suite: **10/10 probes passed**. This includes expected rejection of forged grader output, memory exhaustion, timeout, parent symlink, and FIFO; isolation checks also passed for grader paths, Git history, environment, network, and subprocess writes.
- The unwritten-FIFO probe was rejected as non-regular in `0.00012` seconds; `grade_calls` was not invoked. This is below the frozen one-second bound.
- Total wall time: 9.67 seconds. Container peak memory was not measured.

## Decision and limits

Protocol v12 is **PASSED** for this bounded calibration. The leaf source is opened with `O_NONBLOCK` and `O_NOFOLLOW` relative to a no-follow directory descriptor; descriptor metadata must show a regular file before the exact bytes are hashed and staged. Historical v1/v2 red-team receipts remain verifiable under their original schemas. Run 17 and its v11 protocol remain unchanged.

This does not certify a general-purpose sandbox, reproduce the full upstream HTTP/model-serving test, establish a coding-model repair, or demonstrate SFT/RL/policy improvement. No model inference, training, Development evaluation, or Held-out access occurred.

Raw machine records are `calibration.json` and `calibration.red-team.json`. The frozen protocol copy is `snapshots/protocol-v12.json`; hashes are recorded in `SHA256SUMS` and the run manifest.

## Reproduction

Use a clean VARE commit containing protocol v12, a clean `mlx-lm` checkout with both pinned commits, and the cached Colima image. Keep the baseline worktree and Docker temporary directory outside the VARE repository so its clean-worktree guard passes:

```bash
git -C ../mlx-lm-source worktree add --detach \
  "$PWD/../mlx-lm-baseline-1904-v12" \
  2af5f7ce8620115bf37fbbddf9817f083118df59
mkdir -p ../colima-tmp
TMPDIR="$PWD/../colima-tmp" PYTHONPATH=src DOCKER_CONTEXT=colima \
  work/ci-py311/bin/python scripts/run_mlxlm_1904_calibration.py \
  --source-root ../mlx-lm-baseline-1904-v12 \
  --output results/mlsys-repo-eval-v1/run-18-replay/calibration.json
shasum -a 256 \
  results/mlsys-repo-eval-v1/run-18-replay/calibration.json \
  results/mlsys-repo-eval-v1/run-18-replay/calibration.red-team.json
```
