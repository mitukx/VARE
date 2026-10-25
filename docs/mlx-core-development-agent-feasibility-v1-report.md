# MLX core Development agent feasibility v1

**Result: 0/2 independently verified repairs. NO-GO for model learning.** The cached Qwen3-4B policy did not submit a complete edit on either of two newly certified Development tasks. Both attempts ended after the model emitted three identical edit calls at the 192-token per-turn limit; each response was truncated before the tool-call terminator. This measures a failure in the frozen model/tool budget path. It does not show whether the model could solve either task if a complete edit reached the grader.

## Frozen study

The protocol was committed before inference as `4e820c159507802007b87481767da4d59e090516`. Protocol SHA-256: `62f374cc46101649ddbce9a955f6982ab4b9b544b82d604ca3d1287847485e63`.

The study used one greedy attempt (seed 101) on each of two fresh-for-this-run MLX core Development tasks. They had already passed independent source/grader qualification in V5: baseline fail, exact upstream fix pass, and incorrect-mutant fail. They remain Development tasks and add zero families to the held-out inventory.

| Task | Agent trajectory | Outcome |
| --- | --- | --- |
| MLX #4729, negative scalar indexing in `mx.take` | 76.743 s, 695 generated tokens, 7 turns; 4 source-inspection tool calls, then 3 incomplete edit calls at 192 tokens each | No edit applied; empty candidate diff; no candidate build or test |
| MLX #4730, `mx.random.normal` requested output shape | 83.899 s, 730 generated tokens, 8 turns; 5 source-inspection tool calls, then 3 incomplete edit calls at 192 tokens each | No edit applied; empty candidate diff; no candidate build or test |

Each final three-turn sequence used exactly the per-turn token cap and lacked `</tool_call>`. The runner therefore did not parse or apply those edits. The independent grading run validated task, model, prompt, budget, and diff identities, then recorded `NO_CANDIDATE_REPAIR` for both tasks. It did not compile candidate code because neither record contained a patch. The 0/2 result is a failure to deliver a gradeable repair, not a grader rejection of a completed patch.

## Model and task boundary

The only policy was `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, on MLX 0.32.2 / MLX-LM 0.32.2, Python 3.12.12, Apple Silicon Metal. The model file is 2,263,022,417 bytes with SHA-256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`. The complete snapshot inventory and runtime freeze are pinned in the protocol and [preflight freeze](../results/mlx-core-development-agent-feasibility-v1/preflight/pip-freeze.txt).

The agent tools exposed one allowlisted C++ source file per task. It received the task statement, not the known fix, grader, regression-test files, or expected outputs. Candidate execution and grading were kept outside the model tool interface. The upstream material is public, the operator had inspected the fixes during qualification, and model pretraining or task recognition remains unknown.

## Resource and evidence record

Across the two attempts, generated tokens totaled 1,425 and measured process wall time totaled 160.642 seconds. MLX peak active memory was 3,665,624,128 bytes per process; process peak RSS was 1,471,528,960 bytes and 2,627,731,456 bytes. Host spot samples recorded a minimum 5,664,407,552 bytes available, above the 5 GiB floor. Swap usage remained 1,411,842,048 bytes before and after each task, with no observed growth. These were discrete snapshots, not continuous host-memory telemetry. No paid API, network fetch, optimizer update, or model training was used.

The grading invocation first used the agent venv and failed before grading because it lacked the V5 grader's `psutil` dependency. That launch failure is retained. The same frozen grader then ran with the pinned V5 build Python and completed: `NO_CANDIDATE_REPAIR_VERIFIED`, 0/2. This setup error did not start inference or alter the task workspaces.

The task/grader inventory remains 10/15 toward the existing held-out gate. These two Development families do not change that count. There is no positive trajectory to train on, so SFT, GRPO, asynchronous coding-RL comparison, and held-out model inference remain NO-GO. This run establishes no coding-capability improvement, model-learning effect, async benefit, or RSI gain.

The result manifest covers all 13 retained run files and has SHA-256 `3e0d039b7f2162d54fd7613aa576c8c6288b20047a890642fc5ecb4a1a9d4bb1`. Key records:

- Protocol: `62f374cc46101649ddbce9a955f6982ab4b9b544b82d604ca3d1287847485e63`.
- #4729 trajectory: `014e17b6a806ebd5d4046a5c53cc74b457556b0488b7c5fda6f85cc4a1b259a0`.
- #4730 trajectory: `8093a88280ca0ea73f09fffeb9880c5159ab5bc6b23b2ccd9c63c9e9d42e6623`.
- Independent grader cohort summary: `a9f3230eeec111cec7422e6d120aebb7994806d63461827b4cd1874a0f2847a3`.
- V5 task certification audit: `81849c340276d7a2d09511a7853ab5c0d81efd4243fda9f62d23e41823978574`.

The initial broad V1 exposure scan is retained losslessly as `results/mlx-core-development-qualification-v1/exposure-audit.json.gz` to fit the repository's per-blob size limit. Its sidecar manifest records compressed and uncompressed SHA-256 digests; decompressing it reconstructs the original 2,396,541-byte JSON exactly. The frozen V1 protocol uses the corrected V2 exposure audit, whose hash is unchanged.

See the [frozen protocol](../protocols/mlx_core_development_agent_feasibility_v1.json), [raw run bundle](../results/mlx-core-development-agent-feasibility-v1/), and the [V5 source/grader qualification report](mlx-core-development-qualification-v5-report.md).

## Decision and next gate

Do not reuse these two consumed Development tasks for another attempt under a changed token budget. Before another model trial, qualify fresh Development tasks and preregister a tool-output budget that can carry a complete structured edit, with retained attempt records. Continue the independent held-out task certifications separately. Model learning remains NO-GO until the 15-task eligibility gate and positive, independently verified training-signal gates pass.
