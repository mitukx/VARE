# MLX-LM #1904 repository-task isolation calibration v1

**Decision: the narrow train-task function contract passed its pinned baseline/fix and focused isolation checks. This is a task-harness result, not a coding-agent or model-capability result.** The supported contract is deliberately narrower than upstream's complete HTTP test.

## Hypothesis and frozen scope

The task is `mlxlm-1904-tool-arguments` from the frozen Train split in MLSys task deep-qualification v1. The hypothesis was that a clean export of the pinned upstream baseline would fail the dict-valued argument case, while the exact upstream fix would pass it and preserve JSON-string behavior. The [v2 protocol](../benchmarks/mlsys_repo_eval_v1/mlxlm_1904_tool_arguments/protocol-v2.json) fixes the source commits, file/test hashes, workspace policy, image, limits, calibration decision and implementation hashes. It does not alter the frozen source split or its evidence.

The original upstream selector, `test_handle_chat_completions_with_dict_tool_arguments`, starts the HTTP/model-serving fixture and loads MLX. The isolated adapter supports only the directly relevant `process_message_content` behavior: already-decoded object arguments are accepted, while string arguments are still decoded to the same object. It does **not** claim exact equivalence to that full HTTP test. The task prompt and report disclose this limitation.

## Execution and independent grade

The runner verified a clean baseline checkout at `2af5f7ce8620115bf37fbbddf9817f083118df59`, exported only the production Python package and `TASK.md`, and removed upstream Git history and tests. Patch parsing accepts one unified diff for `mlx_lm/server.py`; absolute paths, traversal, renames, file creation/deletion and additional files are rejected. The grader separately copies only `server.py` and a fixed adapter into a temporary subject mount. It AST-selects the pinned function and runs it in VARE's existing isolated candidate callee. Expected outputs remain in the trusted parent process and are never sent to the candidate.

Run 4 used macOS arm64 with Colima Docker 29.5.2/arm64 and the digest-pinned Python 3.12 image. Candidate containers had no network, a read-only root and source-only read-only mount, UID 65534, one CPU, 512 MiB memory cap, 64 PIDs and a five-second call timeout. The trusted grader was in a separate host process and was not mounted into the candidate namespace. The baseline's dict-arguments case failed (`wrong_output`); its string control passed. The fixed candidate passed both cases, and its `server.py` SHA-256 matched the file at pinned fixed commit `caed1943d38ecb58f6e466a72bed98310c9d45a5` exactly.

The fixed-revision source-swap control was rejected. The two candidate calls took 0.18–0.28 s each; the full baseline/fix calibration took 2.06 s. The memory figure is a configured ceiling, not measured peak container usage.

## Security regression results

The separate [red-team record](../results/mlsys-repo-eval-v1/run-4/red-team.json) passed eight focused checks:

- `/grader` was absent and no grader file could be opened; upstream `.git` history was absent from `/subject`.
- A host-only sentinel environment variable was not inherited by the candidate; an outbound network attempt did not connect.
- A subprocess could not write to the read-only source mount. A forged JSON result on candidate stdout did not turn an incorrect return value into a pass.
- Timeout and memory-exhaustion candidates returned `missing_result` and failed closed.
- Unit regressions rejected absolute/traversal/additional-file patches, a symlinked patch, a symlinked workspace, and a symlink at the pinned target.

The red-team probes ran with a 128 MiB cap and 0.4 s timeout for the timeout case. They validate this callee and Docker setup only; they are not a general-purpose sandbox certification. The evaluator input values are delivered to candidate code as function arguments, while expected outputs, grader source, Git history and host credentials are kept outside its namespace.

## Reproduction and retained outcomes

The calibrated source and fixed patch were fetched from the public upstream repository at the locked commits. Re-run from VARE's repository root:

```bash
TMPDIR=/path/shared-with-colima \
PYTHONPATH=src DOCKER_CONTEXT=colima \
python3 scripts/run_mlxlm_1904_calibration.py \
  --source-root /path/to/clean/mlx-lm-baseline \
  --output results/mlsys-repo-eval-v1/new-run/calibration.json

TMPDIR=/path/shared-with-colima \
PYTHONPATH=src DOCKER_CONTEXT=colima \
python3 scripts/red_team_mlxlm_1904.py \
  --source-root /path/to/clean/mlx-lm-baseline \
  --output results/mlsys-repo-eval-v1/new-run/red-team.json
```

Run 1 preserves the initial Colima failure from using a temporary path outside the VM's shared mounts. Run 2 is a successful preliminary replay. Run 3 added a source-swap check. Run 4 is the final protocol-v2 replay and red-team suite. No old record was overwritten. The raw JSON, source patch, protocol, task, grader adapter and SHA-256 manifest are retained under `results/mlsys-repo-eval-v1/`.

## Limits and next gate

No LLM or agent produced a patch. This does not establish agent success, Development pass@k, reward variance, policy update feasibility, or held-out improvement. No Development or Held-out task was executed, and no SFT or GRPO run started. The next gate is to qualify process-isolated graders and affordable execution for all three frozen Development tasks, then freeze a development-only Qwen3-4B MLX attempt protocol. Until that exists, model rollout remains NO-GO.
