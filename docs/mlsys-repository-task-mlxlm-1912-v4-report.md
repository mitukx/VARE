# MLX-LM #1912 isolated grader v4 calibration

**Result: PASS for the frozen one-task grader calibration.** This is not a model rollout or capability result.

The prospective v4 protocol was executed against exact MLX-LM baseline `47e43a4526d42a6dcee758f07eb04b78a8439ecb` and fix `87b7b583a697537aa68f47130b40884700b5f55f`. File-backed SHA-256 values matched the frozen protocol. The run used the protocol-pinned arm64 Docker image `localhost:5000/vare/mlx-cpu-task@sha256:847edcf1b2b15d00914674e89931f9d38c4aa8a0f1d763cbb96c1359a0647388` through Colima (`DOCKER_CONTEXT=colima`).

At implementation commit `83cb9deaa5e1137bb38e3abf18111a06b523aef6`, the 14 parent-owned exact output cases produced baseline **FAIL** and known-fix **PASS**. The pinned source swap was rejected. The red-team visibility/network/output-forgery/subprocess-write/timeout/OOM suite passed. A supplemental constant-index-zero/zero-value mutant failed exactly the two new nonzero-value/shifted-argmax cases. Its evidence is explicitly supplemental and does not change v4's primary calibration definition.

The first attempt (`run-1`) is retained as NOT_RUN: the earlier local source mirror lacked historical blobs needed to create the pinned checkout. It graded no candidate. Calibration succeeded with a fresh full upstream clone in `run-2` without changing the pinned revisions or protocol.

## Evidence and reproduction

Raw calibration, all case results, source/image identity, red-team output, and the supplemental mutant record are under [`results/mlxlm-1912-isolated-v4/`](../results/mlxlm-1912-isolated-v4/). Each run has SHA-256 manifests. Reproduction from a clean checkout of this implementation revision, with Colima active and the image already present:

```bash
DOCKER_CONTEXT=colima python scripts/run_mlxlm_1912_calibration.py \
  --source-repo /path/to/full/mlx-lm-clone \
  --image 'localhost:5000/vare/mlx-cpu-task@sha256:847edcf1b2b15d00914674e89931f9d38c4aa8a0f1d763cbb96c1359a0647388' \
  --output results/mlxlm-1912-isolated-v4/reproduction
```

The source clone must contain both locked commits and exact target blobs. The existing partial local mirror did not satisfy this requirement; a fresh full clone did. The container exercises the supported CPU contract only; this does not validate Metal behavior, arbitrary Python repositories, or model-generated edits.
