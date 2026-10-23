# Qwen3-4B MLSys Development patch feasibility v1

## Frozen setup

This was a no-update, single-turn coding-agent feasibility screen on the three PR #95 Development tasks only. The experiment protocol was committed before inference as `vare.qwen3-4b-mlsys-development-patch-feasibility.v1`. It pins each source/grader revision, prompt hash, model snapshot and sampling seed. The Transformers prompt contains its frozen `pad()` method excerpt to keep memory use bounded; the two MLX-LM prompts contain their full target source files. The candidate received no tests, expected outputs, upstream fixes, or grader results. Each response was accepted only as a raw unified diff, applied to the pinned source, then sent to the task's independent isolated grader.

Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; MLX 0.32.4, MLX-LM 0.32.2, Metal GPU active. Host: Apple M1 Pro MacBook Pro, 32 GiB unified memory. The exact protocol and model file hashes are in [`qwen3_4b_mlsys_development_feasibility_v1.lock.json`](../protocols/qwen3_4b_mlsys_development_feasibility_v1.lock.json).

## Result

The frozen 15-rollout budget completed in 281.77 seconds after a 1.43-second model load. Each task received one greedy attempt and four sampled attempts. **No generated response passed strict patch-format validation, so no candidate patch reached an independent grader.** Greedy success was 0/3 tasks; sampled pass@4 was 0/3; all three sampled reward groups had population variance 0. Each task had four distinct sampled output hashes, but all outputs failed the frozen format contract. Outputs commonly used code fences or explanatory prose. The protocol did not strip wrappers or retry, and those responses remain failures.

| Task | Greedy | Sample successes | pass@4 | Reward variance | Format failures |
|---|---:|---:|---:|---:|---:|
| `transformers-45852-4d-mask` | 0 | 0/4 | 0 | 0.0 | 5/5 |
| `mlxlm-1912-top-p-min-token` | 0 | 0/4 | 0 | 0.0 | 5/5 |
| `mlxlm-1971-mimo-value-scale` | 0 | 0/4 | 0 | 0.0 | 5/5 |

The run processed 44,905 prompt tokens and generated 6,780 tokens. Mean prefill was 331.9 tokens/s (range 275.1–358.2); mean decode was 49.5 tokens/s (range 47.5–53.3). Mean MLX peak allocation was 3.50 GiB, maximum 3.59 GiB. Process peak RSS was 2.39 GB. Available unified memory was 9.89 GB before model load, 8.73 GB after load, and 8.52 GB at the end. Swap use stayed at about 5.22 GB with about 0.96 GB free; no run-time swap growth was observed. Patch-format validation and application handling took 3.91 seconds total; the independent graders were not invoked. No paid API or remote compute was used.

The independent audit reconstructed the three task summaries and verified the 15 raw responses, prompt hashes, implementation commit, model snapshot hashes, and evidence checksums. See [`independent-audit.json`](../results/qwen3-4b-mlsys-development-feasibility-v1/independent-audit.json), its checksum, and the [raw run bundle](../results/qwen3-4b-mlsys-development-feasibility-v1/run-1/).

## Decision and limits

**NO-GO for SFT and GRPO.** This screen produced no verified reward signal. GRPO also has zero reward variance in every sampled group. No policy update, optimizer step, adapter save/reload, or checkpoint rollback was attempted. The nine frozen Held-out tasks were not accessed. They remain below the research target of 15; the Train split still has only three tasks. No statistical improvement claim is possible.

This is a single-turn patch-format screen, not an interactive tool-using agent evaluation or a coding capability estimate. Its failure is specific to the frozen prompt/output contract and these three Development tasks; it does not show whether the proposed code changes were correct. The model has now seen this Development cohort and its own outputs; do not reuse it as a fresh evaluation set. A future attempt needs a new Development cohort and a separately frozen protocol with a declared deterministic diff-extraction contract before any model output. Do not retrofit a parser to these retained responses and count the results as a pass.

## Reproduction

Run the independent audit with:

```bash
PYTHONPATH=src python scripts/audit_qwen3_4b_mlsys_development_feasibility_v1.py \
  --run results/qwen3-4b-mlsys-development-feasibility-v1/run-1
```

The frozen experiment runner, pinned local environment, and source-repository paths are recorded in the protocol and raw preflight record. A reproduction must use the same source commits and model snapshot; a new coding-agent screen requires a new protocol and fresh Development tasks.
