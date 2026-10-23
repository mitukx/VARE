# MLX-LM #1971 v8 isolated grader calibration

Date: 2026-10-23

## Hypothesis and scope

The v8 cache mutant moves value scaling after cache insertion while preserving the no-cache rotary-position branch. The grader should accept the exact pinned fix, reject the broken baseline, distinguish a constant-scale shortcut, and reject the post-cache mutant only on incremental cached cases. This is a synthetic, CPU-only grader calibration for one Development task; it is not a Coding-Agent rollout, Metal validation, or policy-improvement result.

## Frozen inputs

- Protocol: `vare.mlsys-repository-task.mlxlm-1971.v8`
- Protocol SHA-256: `7f463ee8f4fb9c8d84dfdd6b1f59b6968110b09a666898ff531856be01cc666c`
- Implementation commit: `37ae70f4e93d6ebf9c5a35efff3149c0f5f86e4a`
- Upstream baseline: `5f3c79d6bb1864a85d5343818192bd061b161e3a`
- Exact fix: `8fda5e976a2da412d9f8119361edefd405381cf9`
- Source manifest SHA-256: `686d220bc113df0a16542a02546c98b07d19f9ab6228dde13b77683efcc5be67`
- Runtime: Colima, Linux/ARM64, pinned CPU-only image `localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe`

## Results

Calibration passed in 16.46 seconds. Baseline failed all 6 full-forward and cached cases as `wrong_output`; the exact known fix passed all 6. The constant-0.5 mutant passed only the two scale-0.5 cases and failed the scale-0.25 and scale-0.8 cases. The post-cache mutant passed all three no-cache full-forward cases and failed all three cached cases as `wrong_output`. The independent source-swap check and all red-team controls passed.


Historical calibration caveat: the retained v5 cache-mutant run also failed the full-forward cases, so it cannot be cited as isolated evidence for cached execution. V4's acceptance predicate treated any non-pass as a rejected mutant, including execution failures. Neither historical protocol/result is changed; v8's exact full-pass/cached-wrong-output pattern is the first calibration here that isolates the cache-specific defect.

The v7 run remains preserved at [v7 run 1](../results/mlxlm-1971-isolated-v7/run-1/). It failed its calibration gate because its mutant generator also altered the no-cache rotary-position branch, making full-forward cases fail; it is not counted as cache-specific evidence.

## Reproduction

From the VARE checkout, with the pinned upstream baseline available as a clean checkout:

```sh
DOCKER_CONTEXT=colima PYTHONPATH=src python scripts/run_mlxlm_1971_calibration_v8.py \
  --source-repo /path/to/mlx-lm-baseline \
  --image 'localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe' \
  --output /tmp/mlxlm-1971-v8-run
```

Raw result SHA-256: `38cd9820e3a2f44f251970eed3e5bcda9abfca19dc98e3ddd62e28b98ae1687c`.

## Limitations

This validates a narrow grader contract on a synthetic random model and CPU-only container. It does not show model-generated repair, task success by an agent, Metal correctness, transfer to other bug families, or policy improvement. It must not be used as evidence that v7 passed.
