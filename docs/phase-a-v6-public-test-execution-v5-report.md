# Phase A v6 public-test execution v5 calibration

Date: 2026-10-22 UTC
Protocol: `vare.coding-agent.public-test-execution.colima.v5`
Protocol SHA-256: `9b45c4a596e35068abb9104ad31f1956db8773129b5a2f9c95530940d2774be3`
Calibration: [`run-1`](../results/phase-a-v6/public-test-execution-v5/run-1/)

## Question and result

Does staging a candidate workspace from macOS system temporary storage into an owned home-cache directory make the existing no-network Docker public-test runner usable with Colima, while distinguishing infrastructure failures from candidate failures? **Yes for the frozen one-file calibration fixture.** This is an execution-path calibration, not an agent or model success.

The exact frozen image was `vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`; local image ID matched. With `DOCKER_CONTEXT=colima`, a direct bind of a live `/private/var/folders/...` directory failed before container startup with exit 125 and Docker's `invalid mount config ... bind source path does not exist` diagnostic. A directory under `~/.cache/vare-public-test-staging` mounted successfully and the in-container marker matched. Thus the six historical v6 exit-125 results with this recorded daemon diagnostic are Colima bind-launch failures, not test timeouts, candidate assertion failures, or model failures.

The runner now copies an exact candidate workspace snapshot into a private user-owned cache directory, checks regular-file inventory and bytes before and after staging, rejects symlinks/special files and unexpected files, verifies the public-test hash, then mounts the staged workspace read-only into the network-disabled container. The known baseline fixture produced an ordinary assertion failure (exit 1; gradeable false-pass), while the exact known fix passed (exit 0). Docker launch errors, timeouts, and terminated containers are ungradeable with `passed: null`; an ordinary candidate exit 125 without a Docker launch diagnostic remains a test execution error. The fixture also confirmed the host-managed latest-read opaque-handle path can apply one exactly anchored edit without asking the model to echo handles.

## Historical rollout analysis

[`action-analysis-v3`](../results/phase-a-v6/action-analysis-v3/analysis.json) independently verifies the retained 12-rollout raw bundle and its SHA manifest. Across 120 turns there were 94 tool calls: 12 file listings, 28 searches (12 with hits), 39 reads, 6 edit calls, 6 manual public-test calls, and 3 finish calls. There were 8 repeated identical read outputs, 3 repeated identical searches, 19 no-tool retries, and 9 rollouts with no edit. Three rollouts attempted an edit; one edit applied. Five of six edit calls were rejected because required opaque handles were missing. Six serialized public-test records were exit-125 Colima bind errors; one automatically attached event lacks a serialized return code. No public test passed and no independent task succeeded. Stop reasons were six exhausted no-tool retries, four turn-budget exhaustion, one context-budget exhaustion, and one no-tool call after edit. Summed recorded generation usage was 14,802 generated tokens and 341,869 prompt tokens across turns (the latter is not deduplicated context usage).

These historical tasks remain consumed. Their results are not fresh model evidence. The new implicit-handle controller behavior is supported only by deterministic tests and the v4 fixture; it has not been evaluated in a new model rollout.

## Reproduction and integrity

```bash
DOCKER_CONTEXT=colima PYTHONPATH=src python scripts/calibrate_public_test_execution_v5.py \
  --output results/phase-a-v6/public-test-execution-v5/run-1

(cd results/phase-a-v6/public-test-execution-v5/run-1 && shasum -a 256 -c SHA256SUMS)
```

The run records exact protocol and implementation hashes, Python/platform, Docker context, image identity, source and public-test hashes, result classifications, and resource/isolation settings. No model weights were loaded and no optimizer update ran. Protocols v1–v3 and all prior raw calibration bundles remain historical and unchanged; v4 is prospective and explicitly records v3 as its parent. The original Phase A v6 report and raw rollout records were not rewritten.

Focused validation on the current working tree: 37 tests passed across structured editing, handle-schema, public-test classification/staging, and live Colima baseline/fix cases; Ruff passed; mypy passed on the four modified/added Python source entrypoints. The v5 calibration passed and its result SHA manifest verifies.

## Limitations and decision

The calibration uses a synthetic one-file harness fixture. It does not establish repository-task success, fresh Development performance, hidden-grader success by the model, reward variance, or policy improvement. No new inference, SFT, or GRPO was run. Phase A remains **NO-GO** for learning: the 12 historical rollouts yielded zero successes and constant reward, and no fresh task cohort has been frozen and independently calibrated. The next experiment should wait for qualification of unused tasks and then run a new, preregistered development-only agent study; do not replay these four consumed tasks as fresh evidence.
