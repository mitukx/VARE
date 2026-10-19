# Verified coding-agent baseline v3 — execution preflight

**Outcome: BLOCKED before inference or candidate execution.** This version records the first missing operational gate against the current VARE checkout. It does not change v1 or v2 protocols and does not count as a model, coding-agent, task-success, or grader measurement.

## Why the smoke test did not run

The pinned Qwen2.5-0.5B-Instruct snapshot is locally cached and its protocol-pinned `config.json` and `model.safetensors` hashes match. The checked historical task inputs also match the retained v2 hashes. Python 3.12, `torch` 2.9.1, and `transformers` 4.57.3 are available. The Docker CLI is installed, but its daemon is not running. More importantly, VARE does not yet provide a reviewed launcher that contains untrusted candidate execution while keeping the trusted evaluator and host credentials outside that execution boundary. Starting the model and then grading on the host would violate the repository's executable-evaluation invariant.

The final recorded environment result is [`results/verified-coding-agent-baseline-v3/preflight.json`](../results/verified-coding-agent-baseline-v3/preflight.json). The earlier initial preflight record is retained as `preflight-initial.json`. Both record zero model loads, zero agent starts, zero grader starts, zero candidate-code executions, and no task outcome. The process exit code `3` means the preflight was blocked; it is not a grader failure or agent failure.

## Reproduction

From the repository root:

```bash
python3 scripts/preflight_verified_agent_baseline.py \
  --output results/verified-coding-agent-baseline-v3/preflight-local.json
```

The preflight is read-only except for creating its new evidence file. It will not overwrite an existing file. The task is the previously consumed RVL behavior-policy parity integration control; this protocol is not a fresh held-out task. Required deterministic tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_verified_agent_preflight.py
```

On the recorded machine this completed with **4 passed** under Python 3.12.12.

## Scope and next gate

This adds a reproducible, fail-closed record of prerequisite state. It does not implement candidate isolation, a trusted grader launcher, agent-to-`TrajectoryStore` persistence, interruption recovery in the local-model runner, or an end-to-end smoke test. The smallest next engineering boundary is a reviewed isolation adapter with explicit filesystem, network, credential, process, CPU, memory, and wall-time limits, plus independent grader separation. Only after that adapter is implemented and exercised should the pinned local agent be run; task success must still be measured by the independent grader. No claim of capability, external generalization, or RL improvement follows from this preflight.
