# MLX-LM #1971 isolated task calibration v1

## Result

The frozen #1971 Development task passed a one-task isolated CPU calibration on implementation commit `dfa911b0d317327b1eebad8698f5893ab3ec3cfc`.

- Pinned baseline source `5f3c79d6bb1864a85d5343818192bd061b161e3a` failed the independent output comparison (`wrong_output`). Its source SHA-256 matched the frozen baseline hash.
- The known upstream fix `8fda5e976a2da412d9f8119361edefd405381cf9` passed. A patch applied to the pinned baseline reproduced the fixed source SHA-256 exactly.
- The source-swap probe rejected the fixed checkout as a baseline.
- The shared red-team suite passed its grader/Git/environment visibility, network, forged-output, subprocess-write, timeout, and memory-exhaustion checks.
- Candidate execution used a pinned Linux ARM64 Docker image with network disabled, 1.5 GiB memory, one CPU, 64 PIDs, read-only root, and an unprivileged UID. End-to-end calibration took 3.98 seconds. Peak host memory was not measured.

The grader runs a candidate-visible adapter in the isolated process and compares returned model outputs to host-held golden outputs generated from the pinned fixed source. The expected outputs, upstream tests, Git history, and fixed source are not copied into the candidate workspace. The task prompt describes only the requested behavior and the single allowed source file.

## Scope and limitations

This is a synthetic, randomly initialized small MiMo-V2-Flash model case, not a pretrained checkpoint, model-generated patch, agent rollout, Metal performance result, or policy update. The CPU container establishes this test's computation and grading semantics, not Metal equivalence. It certifies only this task adapter and patch allowlist; it does not establish a general-purpose safe Python repository sandbox. No Qwen inference or SFT/GRPO run is authorized by this calibration.

## Reproduction

The protocol is frozen in [`protocol-v1.json`](../benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v1.json). With the pinned MLX-LM source repository and the named image available:

```bash
DOCKER_CONTEXT=colima python scripts/run_mlxlm_1971_calibration.py \
  --source-repo /path/to/mlx-lm \
  --output /tmp/mlxlm-1971-run
```

The script requires a clean implementation worktree, verifies all protocol hashes and Git index flags, and writes a machine-readable result and SHA-256 file. Retained raw output is in [`run-1`](../results/mlxlm-1971-isolated-v1/run-1/).

## Decision

**GO for adding this task-specific isolated grading calibration to the Development evaluation path. NO-GO for Qwen coding-agent feasibility, SFT, and GRPO.** The other Development tasks still need independent calibration, and no model-generated patch or independent held-out capability result exists. The frozen held-out split remains unopened.
