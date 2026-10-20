# MLSys coding-agent run on one 14B CPU snapshot

This report records one frozen local run. It is a live model measurement on nine authored tasks. It is not a success rate, a comparison against a 4B model, or a training result.

## Question

Can one pinned local coding model, through a tool loop that never sees the hidden grader, produce a non-empty source edit that the task's own hidden grader accepts?

The development split was the gate. Those five tasks had already been attempted by an earlier 0.5B harness, so a pass there is feasibility on a consumed split. The held-out split was graded only after that gate passed, once, with the same model, seed, and budgets.

## Frozen choices

Protocols:

- `protocols/mlsys_coding_agent_e2e_dev_v1.json`
- `protocols/mlsys_coding_agent_e2e_heldout_v1.json`
- `protocols/mlsys_coding_agent_recovery_v1.json`

Model: `Qwen/Qwen2.5-Coder-14B-Instruct-GGUF`, revision `d0a692ef765eefbf2fabb130b3cb2e8917e3d225`, file `qwen2.5-coder-14b-instruct-q4_k_m.gguf`, quantization Q4_K_M, 8,988,110,272 bytes, SHA-256 `c1e659736d89ac1065fb495330fb824d94001974a4bfa78e7270e43476a8d940`, license Apache-2.0. Runner `llama-cpp-python==0.3.16`, CPU, `n_gpu_layers=0`, `n_ctx=4096`, `n_threads=4`, `n_batch=128`, temperature 0, top-k 1. Seed 101. Qwen3 thinking tags were not used.

Budgets: 12 turns, 256 new tokens per turn, 3072 new tokens total, 2 retries without a tool call, 2400 seconds per task. The wall budget was raised from 1200 before any pack task because a non-pack probe generated 35 tokens in 15.7 seconds after the weights were resident.

Host: Linux x86_64, 4 CPUs, 15 GiB RAM, a 24 GiB swap file. MLX was not available. A 32B Q4_K_M file is 19,851,335,872 bytes. A resident-page probe held 12 GiB without material swap and spilled about 2 GiB when touching 14 GiB, so the 32B file was not loaded. During the 14B tasks the process resident set was about 14.3–15.4 GiB and swap use stayed under about 1.1 GiB.

Pack manifest SHA-256 `18c5999f823f9a550a561a913023b87f1fc7bf5a86f481d319e4778212d97499`. Development and held-out summaries record git revision `fb7fc40b487e083e41a82c703b0e308bb4d7b338`.

## Harness, before any pack grade

The non-pack probe emitted a single JSON tool call without `<tool_call>` tags. The parser now accepts one bare tool JSON object and still rejects a JSON object that has trailing prose. A second probe then applied `replace_text` and changed `VALUE = 0` to `VALUE = 1` in `/tmp/vare-format-probe`. That file is not a pack task.

Hidden checks stay on the host. The candidate module runs in a user, mount, and network namespace with a chroot. The evaluator `oracle/` directory is not mounted there. Grading copies the candidate workspace, and the evidence tree hash was unchanged after every grade in this run (`integrity_ok` true). This is a process and mount boundary, not a claim that the namespace contains a kernel exploit. GitHub-hosted runners reject `unshare` uid_map writes, so the tests that need that namespace skip there. The measured grades in this report were produced on a host where the namespace call succeeded.

## Development split

Output: `results/mlsys-coding-agent-e2e-dev-v1`. Two of five tasks passed the hidden grader. Both passes finished with a non-empty diff.

| Task | Hidden grade | Stop | New tokens | Seconds | Source diff |
| --- | --- | --- | --- | --- | --- |
| grad-accum-tail | pass | agent finished | 237 | 151.358 | yes |
| paired-shuffle-loader | pass | agent finished | 367 | 169.166 | yes |
| range-sums | fail | no further tool call | 620 | 264.890 | yes, grader reported `IndexError` |
| row-log-softmax | fail | no further tool call | 392 | 170.710 | no |
| warmup-cosine | fail | no further tool call | 862 | 396.739 | no |

The two passes each read the file, applied one exact replacement, ran the public smoke test, and called finish. Public tests are not the hidden grader.

## Held-out split

Output: `results/mlsys-coding-agent-e2e-heldout-v1`. One of four untouched tasks passed. Four tasks do not support a rate.

| Task | Hidden grade | Stop | New tokens | Seconds | Source diff |
| --- | --- | --- | --- | --- | --- |
| aging-scheduler | fail | agent finished | 271 | 131.632 | yes |
| causal-attention | pass | agent finished | 251 | 105.166 | yes |
| dataset-standardize | fail | agent finished | 605 | 239.241 | yes, grader reported `IndentationError` |
| global-grad-clip | fail | turn budget | 920 | 457.690 | no |

`causal-attention` is the first pass on a task that this agent loop had not graded before. `dataset-standardize` failed the hidden correctness check before any speedup comparison. The separated worker adds one RPC to each timed call, so a later speedup failure on this harness would not by itself show that the algorithm is slow. That case did not arise here.

## Recovery

Output: `results/mlsys-coding-agent-recovery-v1`. The predeclared fault is one kill after the first step, before that step is committed, on `grad-accum-tail` only. Baseline discards the workspace and starts again. VARE resumes from the last committed snapshot and keeps the spent token count.

| Arm | Hidden grade | Lost tool calls | Restart wait (s) | New tokens | Seconds |
| --- | --- | --- | --- | --- | --- |
| baseline | pass | 0 | 0.0 | 353 | 169.408 |
| vare | pass | 0 | 2.0 | 349 | 188.658 |

Both arms passed. Lost tool calls are 0 because the interrupted step had not executed a tool. This run does not show that resume preserved completed tool work. The VARE arm spent 2 seconds waiting for an expired lease and finished slower than the baseline restart. The recovery loader was a local fix on top of `fb7fc40`: a protocol without a pack block no longer raises `KeyError`. The agent loop and the task budgets were not changed. The recovery summary's `code_sha256` is `54bdad76088c5ad924e0e0043588bd67b792932e6d9a9de1a3cbd4cbf8210394`.

## What this does not establish

- A coding success rate, a confidence interval, or a 20–30% operating point. The held-out count is 1 of 4.
- That the result would repeat under another seed. Temperature 0 and top-k 1 make another seed the same trajectory, so no second seed was run.
- A gain over a 4B model. No 4B arm was run.
- A gain from VARE resume. The injected fault deleted no tool call.
- Generalization beyond these nine authored tasks, or any post-training update.
- That the user-namespace jail is a security boundary against a hostile candidate.
