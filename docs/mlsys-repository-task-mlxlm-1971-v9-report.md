# MLX-LM #1971 isolated grader calibration v9

Date: 2026-10-23

## Hypothesis and scope

The current public `export_candidate` and `grade_workspace` entrypoints must enforce the v9 protocol, including its complete transitive evaluator hash lock and Git index-flag checks. Under that boundary, the pinned baseline should fail semantically, the exact upstream fix should pass, and the scale/cache mutations should produce their frozen per-case outcome patterns. This is synthetic CPU-only grader calibration, not an LLM Coding-Agent rollout, Metal validation, or policy-improvement result.

## Frozen inputs and provenance

- Protocol: `vare.mlsys-repository-task.mlxlm-1971.v9`
- Protocol SHA-256: `aa8fec4c3a3fbba119b83009c7f3c164b762758a5c1593faf3a78f3464678fb2`
- Implementation commit: `3f12eefd71d1d2213b37e3a174b446cb5155e979`
- Upstream baseline: `ml-explore/mlx-lm` commit `5f3c79d6bb1864a85d5343818192bd061b161e3a`
- Exact fix: `8fda5e976a2da412d9f8119361edefd405381cf9`
- Source manifest SHA-256: `686d220bc113df0a16542a02546c98b07d19f9ab6228dde13b77683efcc5be67` (201 Python files)
- Candidate image: `localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe`
- Runtime: Colima, Linux/ARM64, CPU-only, network disabled, 1 CPU, 1536 MiB, 30-second timeout
- Raw evidence SHA-256: `5095574212f7d6ef4fe6e89e9a87cbce1e6a99cdae1e7aaf6bd803af05c7ed13`

## Results

The v9 calibration passed in 16.10 seconds. The baseline failed all six full-forward and cached cases as `wrong_output`; the exact known fix passed all six. The constant-0.5 mutant passed the two 0.5 cases and failed the four 0.25/0.8 cases as `wrong_output`. The post-cache-scaling mutant passed all three full-forward cases and failed all three cached cases as `wrong_output`. The source-swap control was rejected, and the v9 red-team suite passed, including visibility/forgery controls and bounded resource-exhaustion controls.

The run exercised the public entrypoints after they were switched to v9. V9 verifies the frozen task inputs and all 20 locked implementation files, including transitive evaluator imports, rejects hidden Git `assume-unchanged`/`skip-worktree` flags, and checks candidate source inventory before grading.

## Reproduction

With the pinned upstream baseline checkout and Colima image available:

```sh
DOCKER_CONTEXT=colima PYTHONPATH=src python scripts/run_mlxlm_1971_calibration_v9.py \
  --source-repo /path/to/mlx-lm-baseline \
  --image 'localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe' \
  --output /tmp/mlxlm-1971-v9-run
```

The checked-in raw record is [`calibration.json`](../results/mlxlm-1971-isolated-v9/run-1/calibration.json); its checksum is retained alongside it.

## Historical records and limitations

V1–v8 protocols and evidence remain unchanged. V7 is a retained non-pass because its cache mutant also altered the no-cache branch. V8 passed its versioned runner calibration but did not bind the public dispatcher or all transitive evaluator imports; it is historical evidence for that runner revision, not the current public grading route. V4's broad non-pass predicate and v5's non-isolating cache mutant remain documented historical limitations. V9 supersedes those paths for current use without rewriting their records.

This validates a narrow grader contract for one synthetic random small-model Development task in a CPU-only container. It does not demonstrate model-generated repair, Metal correctness, task-family transfer, or policy improvement. No LLM inference, SFT, or GRPO was run.
