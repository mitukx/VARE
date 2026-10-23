# Phase A v6 — Frozen Development Study Result

## Decision

**NO-GO.** The preregistered success gate required independently verified, model-generated repairs on at least two distinct Development tasks and nonconstant sampled rewards on at least one task. The study produced **0/12 verified repairs**, **0/4 tasks with a verified repair**, and zero reward variance in every sampled group. No SFT, GRPO, optimizer update, or held-out evaluation was started.

## Frozen conditions and provenance

- Protocol: `vare-qwen3-4b-coding-agent-development-v6-2026-10-21`, SHA-256 `77ea73f6bd2faf9e8394f6cf285859a1b8dc9f76975a3a33e2791d1c90e4ab94`.
- VARE implementation commit at inference: `d8fc4a4f92d2156d9c7894fffd3c76ec9287c117`; clean source checkout was recorded (`source_repo_dirty: false`).
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` at `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; pinned model files and 4-bit weights were checked before inference. Resource preflight selected MLX Metal GPU (`Device(gpu, 0)`).
- Runtime: Python `3.12.12`, MLX `0.32.2`, MLX-LM `0.32.2`.
- Candidate grading image: `vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`; inspected image ID `sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`. Hidden grader was invoked after the agent process exited; grader files were absent from the candidate subject in all recorded grades.
- Sampling: 4 frozen Development tasks; one greedy rollout (seed 0) and two sampled rollouts (seeds 101 and 202) per task; 12 planned and 12 completed. No held-out task was opened.
- The frozen protocol and task descriptors were not changed. The run-specific orchestrator is separately identified by runner SHA-256 `e9c3dfbdb1cdc89891749fe8a36def419fb2197983222f189f48ebbd78094fdf` and execution supplement SHA-256 `f5bfe01577c71abf4b5a3144c37d4b4ba9a6a8b3ddcf016dceed26c9300bd4da`.

## Results

| Task | Edit attempts | Applied edits | Public test attempts / passes | Verified rewards (greedy, sampled 101, sampled 202) | Independent failure |
|---|---:|---:|---:|---|---|
| `transformers-49315-watermark-repeat-v1` | 0 | 0 | 0 / 0 | [0, 0, 0] | `repeated_ngram_count_mismatch` |
| `transformers-49515-rope-setter-v1` | 2 | 1 | 4 / 0 | [0, 0, 0] | `candidate_exception:KeyError` |
| `transformers-49731-generation-device-type-v1` | 0 | 0 | 3 / 0 | [0, 0, 0] | `structured_output_type_or_value_mismatch` |
| `transformers-49767-janus-background-v1` | 1 | 0 | 0 / 0 | [0, 0, 0] | `janus_background_color_mismatch` |

Aggregate: **94 valid tool calls across 120 tool turns**, 12/12 structured-output parse failures absent, 3 edit attempts, 1 applied edit, 0 public-test passes, 0 independent successes, 0/12 binary rewards. The runner recorded seven public-test attempt events; all six serialized candidate-container results returned code 125 (timeout), and the remaining event has only a controller automatic-test record. No candidate passed the public test suite.

The four task-level grader outcomes were consistent across each task’s three rollouts:

- `transformers-49315-watermark-repeat-v1`: `repeated_ngram_count_mismatch`; no edit was attempted in any rollout.
- `transformers-49515-rope-setter-v1`: `candidate_exception:KeyError`; one edit was applied, while later edit attempts were rejected or not applied.
- `transformers-49731-generation-device-type-v1`: `structured_output_type_or_value_mismatch`; no edit was applied. The candidate output retained a plain `dict` rather than the expected structured output.
- `transformers-49767-janus-background-v1`: `janus_background_color_mismatch`; one edit was attempted but not applied.

## Resource and cost

- Total recorded model-agent wall time: **1,353.03 seconds** across 12 rollouts (sum of per-rollout elapsed time; excludes setup and grader overhead).
- Generated tokens: **14,802**.
- Maximum agent RSS: **2,629,107,712 bytes**.
- Maximum recorded per-turn MLX peak memory: **3,934,537,744 bytes**.
- Median observed decode throughput: **47.30 tokens/s**; median prefill throughput: **351.93 tokens/s**.
- Run preflight: Apple M1 Pro, 32 GiB unified memory, MLX GPU active, available memory 9,011,003,392 bytes, free swap 1,280,636,354 bytes. Preflight passed frozen resource thresholds. The run was not stopped by the memory/swap gates.
- There were **zero optimizer updates**. Cost per verified success is undefined because no success occurred.

## Failure interpretation

The primary failure was action completion, followed by semantic repair quality—not an independent grader outage. Every rollout produced a valid structured response at least once, and the controller/parser recorded no format parse failures. However, the model often kept inspecting/searching without issuing an edit, or stopped on context/turn/no-tool budgets. On the one task with an applied candidate edit, the candidate raised `KeyError`; public tests never passed. Candidate public test execution was also too slow for the frozen 120-second container limit on six invocations. Independent grading itself completed and returned task-specific failures for all 12 candidates, so this is not evidence of grader unavailability.

The six timeout outcomes and the absent edits are retained as negative evidence. This report does not infer that the model family is generally incapable; it shows that this exact frozen agent/model/task configuration did not satisfy the preregistered Phase A gate.

## Reproduction and audit

The exact run is under this directory. `run-manifest.json` records every rollout, task/source revision, model revision, image digest, reward, tool/edit/test counts, wall time, throughput, RSS, and MLX per-turn memory. Each rollout directory retains the raw agent transcript, stdout/stderr, candidate target, candidate diff, launch record, and independent grader result. `runner.snapshot.py` is the exact runner source used during inference and matches the runner SHA in the run manifest. Baseline calibration, frozen protocol copies, preflight, and the execution supplement are also retained. `SHA256SUMS` covers every file in this run directory except itself.

The generated candidate patch for the single applied edit contains two blank lines with spaces; they are preserved verbatim alongside the candidate source. A path-scoped `.gitattributes` rule treats this generated evidence patch like the repository’s other byte-preserved patch artifacts. The run was launched on the original Mac host with the frozen lock environment and source/model caches. The original command used the local source checkout and model snapshot paths; equivalent reproduction requires those pinned revisions and the frozen Docker image to be present. The separate `run-1` record is a preserved pre-inference orchestration setup failure (0 model rollouts); it is not included in these 12 results.

## Next falsifiable hypothesis

Before another fresh model study, change one justified bottleneck: provide a smaller, syntax-preserving public-test invocation that finishes within the frozen test budget, and a controller stop/continue rule that requires an edit proposal after bounded inspection. Then qualify a **new** Development cohort and freeze a new protocol. Do not tune against these consumed four tasks and report them as new validation. Keep SFT/GRPO NO-GO until a fresh study produces distinct verified repairs and nonconstant sampled reward.
