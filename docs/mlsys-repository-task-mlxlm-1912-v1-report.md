# MLX-LM #1912 isolated task calibration v1

**Result: narrow development-task calibration passed; Coding-Agent rollout remains NO-GO.**

This adds one task-specific path for frozen Development task
`mlxlm-1912-top-p-min-token`, using the existing candidate Docker runner. The
candidate can edit only `mlx_lm/sample_utils.py`. The task statement and the
MLX-LM Python source are exported without Git objects, tests, license files,
grader code, or expected values. A fixed invocation bridge calls the candidate
function and returns observed kept indices and values; exact expectations stay
in the trusted parent process.

## Frozen source and protocol

- Repository: [ml-explore/mlx-lm PR #1912](https://github.com/ml-explore/mlx-lm/pull/1912)
- Task split: Development, from the catalog frozen by VARE PR #95.
- Baseline: `47e43a4526d42a6dcee758f07eb04b78a8439ecb`; target SHA-256
  `c93c1eef794725f9f7ce77b6212f61eb6d0fe17b9cd87c06cec6470ee12b07f2`.
- Known fix: `87b7b583a697537aa68f47130b40884700b5f55f`; target SHA-256
  `3aac92d804b6b91c82a4e2844697a8723de26f6af57e10013a482b967e739cfa`.
- Upstream regression: `tests/test_sample_utils.py::TestSampleUtils.test_apply_top_p_keeps_top_token_at_tiny_thresholds`, SHA-256
  `dbe871f37a2930ae6686ce67ffc0a641971ad11943a1e94833fa232c6938852`.
- License: MIT; pinned license SHA-256
  `ccfab7ccb2ea306f71531c8a77bb55507606cd90768b1e32b8b52ab5b48cf01`.
- Protocol: [v3](../benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/protocol-v3.json).

The locked checks preserve the upstream test's core assertions: small positive
`top_p` values retain the maximum token, including two-row 4096-value inputs in
float32, float16, and bfloat16. This is a narrow parent-owned subset rather
than execution of the upstream pytest file inside the candidate container.

## Calibration and isolation

Run 1 used clean implementation commit `e9d59ade60b1f7e764847f03bf8524ac9ab9f178`
and the exact local image
`localhost:5000/vare/mlx-cpu-task@sha256:847edcf1b2b15d00914674e89931f9d38c4aa8a0f1d763cbb96c1359a0647388`
(`linux/arm64`, MLX CPU backend). The clean implementation, task prompt,
implementation locks, image identity, and Git index flags were checked before
execution.

- Baseline failed 6 of 12 exact-output cases; the aggregate grade failed.
- The known fix passed all 12 cases, and its target bytes matched the pinned
  fixed revision.
- Replacing the baseline with the fixed checkout was rejected.
- All nine red-team checks passed: grader path and Git history were absent,
  grader secret and network were unavailable, forged stdout was discarded,
  subprocess writes were blocked, timeout and memory exhaustion failed closed,
  and the source swap was refused.
- Calibration wall time was 3.94 seconds. Per-case elapsed times are retained
  in the raw bundle. Peak memory was not measured. This was CPU execution; no
  Metal/GPU result is claimed.

The immutable run is [run 1](../results/mlxlm-1912-isolated-v3/run-1/), with
the protocol hash, implementation hashes, red-team digest, and SHA-256
manifest. Preliminary adapter iterations are preserved under the v1 and v2
result directories; they are not used as final evidence because the adapter
contract and clean-worktree gate were strengthened in v3.

## Scope and next gate

This is E1-style known-fix calibration for one Development task. No model or
agent generated a patch. Qwen3-4B pass@k, reward variance, optimizer cost, SFT,
GRPO, and held-out success are unmeasured. Do not start those runs yet: the
other two requested Development tasks still need task-specific baseline/fix and
red-team calibration under equivalent isolation, and the 15-task untouched
held-out gate is not met by this result. The host/runtime evidence also does not
establish Metal semantic equivalence for this CPU-only container.

Reproduction on an Apple Silicon host with the pinned image loaded:

```bash
git clone https://github.com/ml-explore/mlx-lm.git /tmp/mlx-lm-task
DOCKER_CONTEXT=colima python scripts/run_mlxlm_1912_calibration.py \
  --source-repo /tmp/mlx-lm-task \
  --image localhost:5000/vare/mlx-cpu-task@sha256:847edcf1b2b15d00914674e89931f9d38c4aa8a0f1d763cbb96c1359a0647388 \
  --output results/mlxlm-1912-isolated-v3/reproduced-run
```

The clone must contain both pinned source commits; otherwise fetch them before
running. The script refuses a dirty implementation worktree or a different
image.
