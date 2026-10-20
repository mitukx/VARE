# 7B MLX MLSys Coding Agent: Independent Evaluation

## Status

This is an exploratory, local inference study. It is **not** a broad generalization benchmark and it does not establish a VARE reliability benefit. The 7B development feasibility gate passed, but the fresh matched cohort produced 0/2 verified successes for 7B. A two-task fault-recovery comparison has not been run; this work is therefore submitted as a draft milestone.

## Question and protocol

We tested whether a local MLX coding agent could edit two newly assigned MLSys utility tasks, pass a public smoke test, and then pass a separately executed hidden grader. The candidate received only `task.json` and the subject workspace. The grader reference and generated expected outputs stayed operator-side; candidate execution and grading used the pinned network-disabled ARM64 container. E0 adversarial isolation checks ran before each model measurement.

The frozen cohort is `mlsys-v3-ema-bias` and `mlsys-v3-label-smooth`, pack commit `14291c1f86b8015dba1414a5690ac044573ae365`, manifest SHA-256 `3fb5bdcf8b21e2d1a2768557855d732e35c377273073aa7d894e2910ce2852d2`. It contains two authored exploratory tasks, not upstream issue patches. The v1 development tasks were not counted as fresh. The GQA task was excluded because its mutation condition is not represented in the isolated-call schema.

Both matched arms used one greedy sample per task (`temperature=0`, `do_sample=false`), seed label 101, 4,096 context tokens, 768 generated tokens per turn, 4,096 total generated tokens, 12 tool turns, and 900 seconds/task. The seed label is not a stochastic replicate. Model, tokenizer, task pack, environment and source revisions are pinned in the protocol files. The 4B arm uses Qwen3-4B-Instruct; the 7B arm uses Qwen2.5-Coder-7B. Their model families and tuning differ, so this is not a causal model-size-only experiment.

## Results

| Cohort | Model | Verified | Wilson 95% interval | Generated tokens | Agent wall time | p50 / p95 task latency | Peak MLX memory |
|---|---|---:|---:|---:|---:|---:|---:|
| Development feasibility, previously exposed 5 tasks | Qwen2.5-Coder-7B 4-bit | 2/5 | 0.118–0.769 | 3,155 | 214.3 s total | 45.4 / 55.3 s | 5.25 GB |
| New matched exploratory cohort, 2 tasks | Qwen3-4B 4-bit | 1/2 | 0.095–0.905 | 2,505 | 117.9 s total | 59.0 / 90.5 s | 3.65 GB |
| New matched exploratory cohort, 2 tasks | Qwen2.5-Coder-7B 4-bit | 0/2 | 0.000–0.658 | 2,034 | 99.1 s total | 49.6 / 55.6 s | 5.12 GB |

The 4B verified EMA task used 516 tokens and 27.4 seconds to verified success. Its second task changed source but failed the public test and hidden grader. The 7B runs made no accepted source change on either fresh task: both ended with `malformed_edit` / `malformed_file`; hidden and public checks failed. Tokens per successful task were 2,505 for the 4B arm when charging all attempted work, and undefined for 7B (zero successes). These two observations do not support a model ranking.

On the separate known development cohort, the 7B model passed gradient-accumulation-tail and paired-shuffle-loader. Both had source changes, public test passes and independent hidden-grader passes. The three other tasks failed. This 2/5 result established feasibility only. Its successful trajectories ended with a malformed final `finish` call after the tests had passed; this stop-format issue is retained in the logs.

## Reliability and VARE effect

The isolated runner persisted full transcripts, tool calls, public-test output, hidden-grader records, source diffs and resource samples. Evidence manifests were independently checked by a separate auditor: 1,169 checks for the 4B matched run and 1,161 for 7B passed, with task decisions recomputed from source-change, public-test and hidden-grader records.

No Basic-vs-VARE execution, worker SIGKILL, interrupted trajectory write, delayed grader, restart/resume, duplicate execution/publication or recovery-correctness experiment was run. Therefore no result is reported for recovery success, lost work, restart overhead, reliability overhead or VARE benefit. The measured runner isolation and evidence audit are not substitutes for that experiment.

## Task and trust-boundary checks

- The prior v2 label-smoothing task had an inconsistent contract/grader. The new v3 task uses a clear `epsilon / num_classes` convention and an independent reference. The intentionally incorrect starter and public smoke test were checked before model inference.
- Two earlier 4B attempts failed closed before inference: one due to pack-manifest traversal including Git metadata (the runner stopped before creating an output directory), and one because the starter baseline unexpectedly passed (that run directory is preserved). The pack and protocol were revised and refrozen before the successful 4B/7B measurements.
- The raw 4B/7B summaries currently label the model evidence as “CPU”; the per-agent runtime records correctly identify `apple_metal`, MLX 0.32.2 and the pinned model hashes. Treat the raw backend summary string as a reporting defect, not as a different execution path.
- E0 checks report a hidden canary visibility count of 0/3 and candidate networking disabled. This is evidence for this tested container path, not a general sandbox certification.
- No hidden tests or grader outputs were present in the candidate task roots. The task set is only two authored tasks, has wide task-clustered intervals, and was run once deterministically.

## Historical pilot context

PR #89's 4B pilot reported 2/5 only after a post-run grader correction; its original inference result and the corrected posthoc evaluation are distinct. PR #88's 14B held-out cohort was already consumed and is not treated as fresh. The older PR #84 task pack is used only for development feasibility and is not the confirmation cohort here.

## Reproduction

From the VARE checkout, with the frozen v3 task-pack checkout and local MLX environment available:

```bash
DOCKER_CONTEXT=colima HF_HOME=work/hf-cache PYTHONPATH=src \
  ../venv-mlx/bin/python scripts/run_separated_coding_eval.py \
  --protocol protocols/mlsys_coding_v3_matched_4b_v1.json \
  --pack-root ../VARE-pack-v3 \
  --output work/evidence/v3-matched-4b-v1

DOCKER_CONTEXT=colima HF_HOME=work/hf-cache PYTHONPATH=src \
  ../venv-mlx/bin/python scripts/run_separated_coding_eval.py \
  --protocol protocols/mlsys_coding_v3_matched_7b_v1.json \
  --pack-root ../VARE-pack-v3 \
  --output work/evidence/v3-matched-7b-v1

../venv-mlx/bin/python scripts/build_coding_agent_evidence_manifest_v1.py \
  work/evidence/v3-matched-4b-v1
../venv-mlx/bin/python scripts/audit_coding_agent_evidence_v1.py \
  work/evidence/v3-matched-4b-v1

../venv-mlx/bin/python scripts/build_coding_agent_evidence_manifest_v1.py \
  work/evidence/v3-matched-7b-v1
../venv-mlx/bin/python scripts/audit_coding_agent_evidence_v1.py \
  work/evidence/v3-matched-7b-v1
```

Each run refuses to overwrite an existing output directory. Preserve all attempt directories, including failures, and do not repeat or retune on this held-out cohort.

## Next work

The next protocol must freeze a larger untouched task cohort and a paired fault-injection design before any further model runs. It should test worker termination after tool execution, interrupted trajectory persistence, grader delay/timeout, restart step consistency, duplicate execution/publication detection and recovery of partial valid work. Until then the observed success counts remain exploratory, and no efficiency or reliability gain is claimed.
