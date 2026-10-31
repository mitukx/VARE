# vLLM #64830 one-shot Development attempt

## Purpose

Freeze one local Qwen2.5-Coder-14B attempt on the already source-qualified vLLM task. The task scope is limited to the HiSparse and watermarking checks in `VllmConfig._get_v1_model_runner_unsupported_features`; whole-PR correctness and the custom-speculator behavior are excluded.

This was a prospective one-attempt protocol. Exact-head CI passed, then launch preflight returned **NO_GO**. **No model weights were loaded, no prompt was delivered, and no inference or independent grade ran.** A failed preflight is terminal for this frozen protocol; do not retry it.

## Evidence before assignment

- Source qualification: `a1bc5dc255e38ad1741f58399504fdaf357eae1a`; qualification artifact SHA-256 `62536b74d4982a073caaf9ed4a3d68f89c1f9b539bdb7669da912a15cbfc958f`.
- Baseline source: `17550580e932f15e772185847382d3ae4211f5ec`, SHA-256 `e14afda305e9fdb3ceb9ff77f58e71db62da35de287d30490177364a52452738`.
- Independent grader controls: baseline and wrong mutant each fail the two frozen target assertions; upstream-fixed passes both. The focused suite also checks a partial repair and classifies it as a target failure. See `calibration/grader-calibration.json` and `calibration/local-harness-checks-001/`.
- The candidate method runs only in a network-denied worker sandbox. Direct attempts to read the hidden overlay, calibration receipts, and user prompt attachment were denied. The model sandbox Metal smoke passed without loading weights.
- Public pretraining exposure is UNKNOWN. Same-repository broad configuration adjacency is possible; family-level independence is not established.

## Frozen model and limits

- Checkpoint: `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` at `29efdbab55a161237ab1e432a3abaf6c7ae2b477`, local 4-bit MLX snapshot.
- MLX `MLX 0.32.4 / mlx-lm 0.32.2 on Apple Silicon Metal`; native tokenizer chat template and tool schema from the pinned runner.
- One greedy attempt, at most 12 turns, 2,048 generated tokens, 4,096 context tokens, 600 seconds, and 12 GiB process-tree RSS.
- Network, paid compute/APIs, training, Held-Out access, grader feedback, and retries after generation begins are prohibited.
- Model-visible task content is `task/TASK.md` plus the single allowlisted source file and controller-generated tool schema/path boundary. The assignment metadata, tests, qualification-arm logs, and grader output are not model inputs.

## Launch outcome

The exact frozen preflight ran at `2026-10-31T07:05:56Z` after both exact-head CI runs passed. It verified all frozen inputs and model snapshot hashes, but returned **NO_GO** because available memory was `11,035,787,264` bytes (below the 12 GiB minimum) and the estimated working-set-plus-headroom gate failed. Disk, swap, and memory-pressure gates passed. The model was not loaded. The attempt status is **NOT RUN**, and the preflight result is in `attempt-001/launch-preflight.json` with a companion no-run receipt and checksum manifest.

The append-only exposure record now records that the prospective Development assignment failed preflight before model exposure. The task remains unexposed to the model; public pretraining exposure remains UNKNOWN. Any future attempt requires a new protocol and a fresh passing resource gate.

## Reproduction

Verify the frozen protocol and its declared inputs:

```bash
shasum -a 256 -c results/mlsys-fresh-vllm-64830-agent-attempt-v1/protocol.sha256
gzip -dc results/mlsys-fresh-vllm-64830-agent-attempt-v1/runtime/runtime-file-manifest.json.gz | shasum -a 256
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mlsys_vllm_64830_agent_grader_v1.py
python -m ruff check scripts/run_mlsys_vllm_64830_agent_grader_v1.py scripts/run_mlsys_vllm_64830_agent_method_worker_v1.py tests/test_mlsys_vllm_64830_agent_grader_v1.py
python -m py_compile scripts/run_mlsys_vllm_64830_agent_grader_v1.py scripts/run_mlsys_vllm_64830_agent_method_worker_v1.py tests/test_mlsys_vllm_64830_agent_grader_v1.py
```

The exact launch-preflight, one-shot attempt, and independent-grader argv remain frozen in `protocol.json`. The preflight was run exactly once and failed closed; the model-attempt and grader commands were not invoked. Verify the no-run receipt with `(cd results/mlsys-fresh-vllm-64830-agent-attempt-v1/attempt-001 && shasum -a 256 -c SHA256SUMS)`. Do not retry this protocol. If a future separately frozen protocol is considered, it must pass a fresh resource preflight first. The grader scratch path remains under `<TMP>` so the worker can read its extracted candidate method; its durable-copy step is unused.

## Engineering and research claims

The harness adds a distinct target-failure classification when a subset of intended assertions fails, reducing the chance that a partial wrong repair is mislabeled as a grader malfunction. Tests and calibration cover baseline failure, known-fix success, wrong-mutant failure, partial-fix failure, hash/allowlist rejection, invalid syntax, timeout, and sandbox isolation.

This protocol does **not** establish model repair, Coding RL improvement, a policy update, multi-family generalization, pretraining independence, full upstream PR correctness, or algorithmic novelty. A single pass would be a narrow Development feasibility signal only.
