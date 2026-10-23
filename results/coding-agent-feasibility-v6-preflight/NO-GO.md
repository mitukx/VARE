# Qwen3-4B Development study resource gate

Date: 2026-10-21 (Asia/Tokyo)

## Decision

**NO-GO: zero task rollouts and zero policy updates.** The host has Apple M1 Pro, 32 GiB unified memory, and MLX/Metal is available, but free swap is below the preregistered 512 MiB safe-start floor. The gate was not relaxed. The Development task prompts were not sent to the model, and no task candidate was generated.

The machine-readable decision is `resource-gate-no-go.json`. It records 9,963,814,912 bytes available memory and 289,868,349 bytes free swap at the gate check. The latter is 276.5 MiB, 259.5 MiB below the 512 MiB requirement. The prior measurement in the short model smoke test had 281,479,741 bytes free swap after inference, also below the threshold.

## Separate inference smoke test

`qwen3-4b-mlx-preflight.json` is a short, non-task runtime smoke test only. It confirms the pinned 4-bit checkpoint loaded on `Device(gpu, 0)`, generated 32 tokens, and measured 24.68 prompt tokens/s, 52.49 decode tokens/s, 2,409,823,624 bytes MLX peak memory, and 3,066,593,280 bytes process peak RSS. Swap free fell from 1,470,229,381 to 281,479,741 bytes. This result is not an Agent rollout, task success, reward observation, or training result.

The smoke-test record did not preserve the exact input prompt or a command/source hash for its generation call. That historical gap is left explicit; the prompt is not reconstructed. The checkpoint identity is revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, with `model.safetensors` SHA-256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`. Runtime was MLX 0.32.2 / mlx-lm 0.32.2 / Python 3.12.12.

## Study implications

- The four frozen v6 Development tasks remain qualified by separate baseline/fixed-source and public-test calibration evidence. This resource gate did not consume them.
- There are no Agent-generated patches and no reward variance data.
- Phase A is **not evaluated**; no successful repair is claimed.
- SFT and GRPO are **NO-GO** because there are no agent rollouts, verified successes, or nonconstant rewards. No optimizer update or held-out evaluation was performed.
- Do not start inference until a fresh audit on the same pinned model and runtime reports at least 4 GiB available memory and 512 MiB free swap. Then freeze the Development protocol from a clean committed tree and retain the exact prompt/runtime evidence before the first task rollout.

## Reproduction

Run the resource gate from the repository root using the frozen MLX dependency lock and the local pinned model snapshot:

```bash
uv run --no-project --with-requirements requirements/mlsys-agent-mlx.lock.txt \
  python scripts/audit_coding_agent_dev_resource_gate_v1.py \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --output results/coding-agent-feasibility-v6-preflight/resource-gate-no-go.json
```

The recorded gate command returned exit code 2, the documented NO-GO result. `vm_stat` available-page counters and `sysctl vm.swapusage` are used so the gate does not depend on an undeclared `psutil` package.
