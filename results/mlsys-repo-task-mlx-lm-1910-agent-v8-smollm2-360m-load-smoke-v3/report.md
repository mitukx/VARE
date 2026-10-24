# SmolLM2-360M MLX model-load smoke

**Status: COMPLETED — resource feasibility only.**

The frozen hypothesis was that the locally cached 360M-parameter MLX instruction model could initialize offline within a strict local resource budget. The pinned checkpoint loaded successfully in 2.386 seconds. The loader returned Model and TokenizerWrapper objects. The process generated zero tokens, loaded no task prompt or grader, and made no optimizer update.

| Measure | Result | Frozen limit |
| --- | ---: | ---: |
| Wall time | 2.386 s | 90 s |
| Peak sampled child RSS | 814.34 MiB | 3072 MiB |
| Swap at start / end | 3862.50 / 3862.50 MiB | +96 MiB; absolute 4096 MiB |
| Minimum sampled free memory | 60% | 35% |
| Generated tokens | 0 | 0 |
| Optimizer steps | 0 | 0 |

The checkpoint is mlx-community/SmolLM2-360M-Instruct, snapshot 5d5e6fee5beb5d4d4d85e2203ef23e1230a9aac7; the complete ten-file snapshot manifest and Apache-2.0 model-card identity are frozen in protocol.json. Runtime packages are preserved in runtime.freeze.txt. Raw stdout, stderr, load result, resource samples, and the run hash manifest are in run-1/.

This result shows that a much smaller MLX checkpoint can initialize in the current host environment. It does not show that the model can generate a useful code patch, pass the independent grader, or learn from rewards. No task or grader was used in this smoke.

## Reproduction

Create an isolated Python 3.12 environment and install the frozen versions in protocol.json (including mlx-lm 0.32.2, mlx 0.32.4, and mlx-metal 0.32.4). With the pinned model snapshot available in the Hugging Face cache, run the following command:

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /path/to/mlx-smoke-venv/bin/python scripts/smoke_local_mlx_model_load_v3.py

The runner verifies protocol, code, package, and model-file hashes before it starts one child process. It exits without retry if a preflight, memory, swap, RSS, or timeout limit fails.
