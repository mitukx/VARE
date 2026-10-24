# Qwen2.5-3B MLX-LM #1910 parser-debug feasibility run

Execution status: **COMPLETED**. Coding-feasibility outcome: **FAILED** (0 independently correct repairs). The model loaded and both preregistered Development-only attempts completed without a resource abort. This task was already exposed to several earlier model runs, so this is retrospective controller debugging evidence only.

## Frozen hypothesis and scope

The frozen hypothesis was that Qwen2.5-3B-Instruct-4bit could complete an inspect, edit, public-test, revise, finish trajectory on the previously exposed MLX-LM #1910 task after a strict tool-call parser correction. The prospective protocol is `protocol.lock.json` (SHA-256 `06815675714dbca916616ee1b9e93f5ad8d6d5d6d93a09e1d61873c05535c6f7`). It permitted one greedy and one sampled attempt, at most 8 turns and 2,048 generated tokens per attempt, with 180 seconds of agent wall time per attempt. No training or optimizer update was authorized. No held-out task was accessed.

Model identity was `mlx-community/Qwen2.5-3B-Instruct-4bit`, revision `4f83f8f146fdf28b512a06562b671d7af4fab457`; the local checkpoint manifest is SHA-256 `909c41a7304cc77a79dbd334dcad473dc89da3b2202b6a33d209e340197d9d16`. Runtime was Python 3.12.12, MLX 0.32.4, and mlx-lm 0.32.2 on Apple Silicon. The checkpoint was used locally for non-commercial research and evaluation; weights were not redistributed.

The task grader had previously passed baseline-fail, exact-known-fix-pass, three incorrect-mutant, and isolation checks. Each candidate was graded independently after inference. The known fix and grader were absent from the candidate workspace.

## Results

| Arm | Seed | Generated tokens | Agent wall time | Valid edit | Public-test container | Independent grade | Peak sampled RSS |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| Greedy | 23 | 392 | 24.753 s | No | Not started | FAIL | 1,273.438 MiB |
| Sampled | 29 | 487 | 25.032 s | No | Invocation failed before container start | FAIL | 1,929.641 MiB |

Both attempts produced reward 0; their population reward variance was 0. The task-level result was 0/1 and attempt-level success was 0/2. The agent edited no source file and invoked no successful public test. The sampled transcript later claimed a patch and passing tests, but the recorded candidate source hash remained identical to baseline and no edit was applied.

The main failure was malformed tool-call framing. The raw model output repeatedly used `{{<JSON payload>}}` inside `<tool_call>` tags. The parser attempted to parse the extra opening brace and rejected every such call. The sampled attempt also reached the public-test tool once, but its pinned local Docker image ID form was rejected before Docker started. The retained trace exposed both controller defects; the experiment was not repeated after fixing them.

Total runner elapsed time was 59.81 seconds. MLX reported peak allocations of 2,705,178,772 bytes (greedy) and 2,643,460,290 bytes (sampled). Host swap stayed at 3,758.44 MiB from start to finish; free memory was 58% at the first attempt start and 52% at the second. There were no optimizer steps.

## Evidence and reproduction

The raw run is retained under [`run-1/`](run-1/). Its evidence manifest SHA-256 is `fd172f29c4c1bac950bb897eab46acaa9a6006cbf01a16a44538b541ebd4cd3e`; the verifier confirmed all 87 listed files. The agent workspaces contain local Git metadata, so their exact files are also retained in `greedy/workspace.tar` and `sampled/workspace.tar`; the manifest still records each file individually, and the verifier confirmed all 87 entries from the archived layout. The earlier Qwen2.5 #1912 exposure manifest also verified all 51 files. The manifest reader accepts both map and sorted-entry layouts while preserving strict path, length, and digest checks.

Verify the retained run without rerunning inference:

```bash
python3.12 scripts/verify_coding_agent_evidence_manifest_v1.py \
  results/mlsys-repo-task-mlx-lm-1910-qwen25-3b-parser-debug-v1/run-1
```

The follow-up code tests validate recovery of the exact retained brace framing, rejection of extra arguments under the declared tool schema, bounded source reads, and resolution of a local content-addressed image ID to a repository digest that inspects back to the same image. A separate live smoke test passed inside the pinned local Docker image with networking disabled, a read-only container root, and only the candidate workspace mounted. These follow-up checks do not change the frozen 0/2 model result.

This run is not fresh-task capability evidence, not held-out evidence, and not evidence of policy improvement. It does not satisfy the learning GO gate.
