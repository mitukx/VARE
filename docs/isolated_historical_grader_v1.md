# Isolated historical grader v1

## Outcome

The pinned RVL behavior-policy parity task was graded inside a local Colima container with networking disabled, a read-only root filesystem, read-only candidate/task/harness mounts, no host environment pass-through, no Docker socket, a non-root UID, all Linux capabilities dropped, `no_new_privileges`, 64 process limit, one CPU, 512 MiB memory, 32 MiB temporary storage, and a 90 second wall limit. The image is pinned by digest and runs Python 3.12.15; the host runs Python 3.12.12. A smoke run of the unchanged base completed in 0.688 seconds and returned `passed=false`, as required by the task lock. Its full output and hashes are retained under [`results/isolated-grader-v1/baseline-v3/`](../results/isolated-grader-v1/baseline-v3/). The first launch attempt is retained separately; it found that the grader file is not executable and exited 126. The launcher was fixed to invoke Python explicitly.

One offline CPU agent run used cached `Qwen/Qwen2.5-0.5B-Instruct` snapshot `7ae557604adf67be50417f59c2c2f167def9a775`, seed 101, up to 30 turns, 256 generated tokens per turn, 7,680 total generated tokens, and a 900 second soft wall budget. The valid recorded attempt took 67.462 seconds, generated 213 tokens, made 6 tool calls across 9 recorded turns, made no source changes, and ended with `no_tool_call_retries_exhausted`. The isolated grader returned `passed=false`; the workspace auditor and evidence collector accepted the record as structurally valid and classified it `no_source_change`. Peak process RSS reported by the local runner was 3,869,622,272 bytes. No paid API or external compute was used. The corrected grader record and audit are in [`agent-101-v3-grade/`](../results/isolated-grader-v1/agent-101-v3-grade/) and [`agent-101-v3-audit/`](../results/isolated-grader-v1/agent-101-v3-audit/). This is one failure on a previously consumed historical integration task, not a held-out result or a general coding-capability estimate.

After the review hardening, the pinned unchanged base again graded `passed=false` (0.655 seconds), and the task's predeclared public calibration revision graded `passed=true` (0.718 seconds). These are harness controls, not agent results. Full records are retained under [`review-negative-control-v1/`](../results/isolated-grader-v1/review-negative-control-v1/) and [`review-positive-control-v1/`](../results/isolated-grader-v1/review-positive-control-v1/), and included in `SHA256SUMS`.

The first inference record had a budget-field naming mismatch with the existing evidence collector and was retained as invalid evidence. The runner's field names were corrected and the same frozen smoke configuration was repeated from a fresh workspace. Only the second record is counted as the valid attempt; both records remain in the raw bundle.

## Reproduction

From the repository root, prepare a fresh candidate workspace outside the VARE checkout:

```bash
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/vare-rvl-run.XXXXXX")
python3.12 scripts/prepare_task.py --workspace "$RUN_DIR/VARE-agent-101"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3.12 scripts/run_local_agent.py \
  --workspace "$RUN_DIR/VARE-agent-101" \
  --model-path /path/to/Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --output "$RUN_DIR/agent-101.json" \
  --instruction 'Read src/rvl_systems/hf_backend.py now. Implement the task with a minimal patch and check syntax.' \
  --seed 101 --max-turns 30 --max-new-tokens-per-turn 256 \
  --max-total-new-tokens 7680 --wall-seconds 900
python3.12 scripts/run_isolated_grader.py \
  --workspace "$RUN_DIR/VARE-agent-101" \
  --output-dir "$RUN_DIR/agent-101-grade"
python3.12 scripts/verify_agent_workspace.py \
  --workspace "$RUN_DIR/VARE-agent-101" \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --agent-record "$RUN_DIR/agent-101.json" \
  --grader-record "$RUN_DIR/agent-101-grade/grader-result.json"
python3.12 scripts/record_coding_agent_baseline.py \
  --protocol protocols/verified_coding_agent_baseline_v2.json \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --workspace "$RUN_DIR/VARE-agent-101" \
  --agent-record "$RUN_DIR/agent-101.json" \
  --grader-record "$RUN_DIR/agent-101-grade/grader-result.json" \
  --output-dir "$RUN_DIR/agent-101-audit"
```

The agent is offline and tool-limited to the task's allowlisted source files. Grader execution requires Docker with the pinned image already present and a local context named by `protocols/isolated_historical_grader_v1.json` (currently `colima`). The grader launcher rejects changed task locks, workspace revision mismatch, staged/untracked or out-of-scope changes, non-regular candidate sources, evidence-directory reuse, and candidate/task/output path overlap. It stores stdout, stderr, the structured grader result, source hashes before and after grading, the candidate diff hash, runtime identity, and exit status.

Focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.12 -m pytest -q \
  tests/test_isolated_grader.py tests/test_verified_agent_preflight.py
```

## Security and evidence limits

The container constrains candidate execution's access to the host. This does not make the historical grader resistant to a malicious candidate that deliberately tampers with the grader: the existing evaluator imports candidate modules in its own Python process and the evaluator files are visible inside the container. The run therefore demonstrates bounded host isolation and functional task grading for this smoke; it is not a hostile-code proof or an independently signed grader attestation. Do not count this task as fresh generalization. The run also does not persist turns through VARE's `TrajectoryStore`, test interruption/recovery, compare agent baselines A/B/C, or qualify an independently sampled task cohort.
