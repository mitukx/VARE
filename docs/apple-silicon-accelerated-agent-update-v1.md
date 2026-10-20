# Apple Silicon agent and policy-update feasibility v1

**Status: update smoke passed; held-out capability and three-arm comparison remain NO-GO.** This report records small, frozen feasibility and recovery measurements on a MacBook Pro M1 Pro / 32 GiB. It does not claim a task-success improvement from VARE or from training.

## Questions and prior evidence

The motivating question was whether VARE's existing coding-agent evidence path and durable trajectory store can be exercised with local MLX/Metal inference and whether an actual small-model adapter update fits this machine. PR #89's Qwen3 4B report states 2/5 after a posthoc grader correction, but its raw archive is unavailable for audit and its frozen protocol specifies 10 attempts while configuring five tasks and one seed. Treat that score as unverified author-reported history; see the [protocol deviation audit](4b-mlx-coding-agent-pilot-v6-audit.md). PR #90's exploratory matched cohort recorded 4B 1/2 and 7B 0/2, but it has only two tasks, greedy single samples, and different model families. Those tasks are consumed and are not reused as fresh confirmation here. The 14B CPU cohort from PR #88 is also consumed.

The hypotheses frozen before the new runs were:

1. The pinned Qwen3 4B snapshot executes on Apple Metal and produces the same greedy output as the MLX CPU backend. This local microbenchmark can falsify an assumption of an automatic GPU throughput advantage.
2. A 0.5B 4-bit base can take ten real MLX LoRA optimizer steps from one grader-verified training-split trajectory, save a nonzero adapter, reload it, and remain under conservative memory limits.
3. VARE's existing durable trajectory store can resume after worker SIGKILL following an isolated public-test tool success but before that step is committed. Compare this with restart-from-scratch on the same tool sequence.

## Hardware and inference measurement

The host was an Apple M1 Pro MacBookPro18,3 with 32 GiB unified memory, macOS 27.0.1 (kernel 27.0.0). MLX and MLX-LM were both 0.32.2 under Python 3.12.12; Metal was available and `mx.default_device()` was `Device(gpu, 0)` for the GPU arm. PyTorch MPS availability was also recorded as true in the control Python. The host snapshot is retained at [`apple_silicon_host_snapshot_v1.json`](../results/apple_silicon_host_snapshot_v1.json). Swap use was already high before these runs (about 7.4 GiB used, 1.6 GiB free) and did not change during the inference and update runs; per-run values are in the summaries. The model was `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, Apache-2.0, 4-bit MLX safetensors, group size 64.

The locked benchmark used a 25-token prompt, one eight-token warmup, then three greedy 96-token generations on each backend. All six generations had 96 tokens and the exact same output hash.

| Backend | Median tokens/s | Per-run tokens/s | MLX peak allocation | Process max RSS |
| --- | ---: | --- | ---: | ---: |
| Metal GPU | 55.67 | 54.95, 55.67, 56.01 | 2.41 GB | 2.65 GB (whole process) |
| MLX CPU | 55.79 | 55.79, 56.41, 53.03 | not reported by CPU backend | |

This short batch-one run shows no measurable Metal throughput advantage: the CPU median was 0.22% higher. It does establish that the exact 4B model ran on Metal and stayed within the 6 GB MLX allocation gate. This is not a general throughput comparison and does not compare inference quality.

## MLX LoRA update smoke

The training example was made from the independently graded `mlsys-v1-grad-accum-tail` development trajectory, with its prompt and source patch retained at [`benchmarks/dev_sft_update_smoke_v1/train.jsonl`](../benchmarks/dev_sft_update_smoke_v1/train.jsonl). No held-out task, grader secret, or confirmation label was used. The base was `Qwen/Qwen2.5-0.5B-Instruct`, snapshot `7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0, converted locally to 4-bit MLX weights (group size 64) using pinned MLX-LM 0.32.2.

Frozen settings: SFT causal language modeling, LoRA rank 8, two trainable layers, batch size 1, sequence limit 1024, learning rate `1e-5`, Adam, prompt masking, seed 101, ten optimizer iterations. The terminal training log contains steps 1–10; training loss on this repeated single example fell from 1.281 to 0.807. The resulting 366,592-parameter adapter has 202,752 nonzero LoRA-B parameters. The saved adapter reloaded and produced 32 tokens at 42.75 tokens/s.

Peak MLX allocation was 1.80 GB and process max RSS was 0.92 GB. Free swap was 1.62 GiB at both run boundaries; 227.7 GB disk was free at start. The update completed in 13.71 s after a 2.09 s local conversion. The update feasibility gate passed after a separate audit parsed the MLX progress table and inspected the learned LoRA-B matrices. The original run summary remains unchanged with its initial parser non-pass; the audit documents that parser defect and reconstructs the ten updates from the retained log. See [`independent_audit.json`](../results/mlx-lora-update-smoke-v1/independent_audit.json) and the locked [protocol](../protocols/mlx_lora_update_smoke_v1.lock.json).

This is a one-example training fit and adapter save/reload smoke. It did not retain or reload optimizer state, compare a no-update policy, evaluate held-out task success, test multiple seeds, or run GRPO. Therefore it does **not** establish an SFT benefit, RL result, or model capability gain. A Qwen3 4B update and a 4B task-success gate are not cleared.

## Worker recovery after tool success

The recovery protocol reused VARE's `TrajectoryStore`, `AgentStep`, leases, and hash-chain auditor. It used the already exposed development task `mlsys-v1-grad-accum-tail`, not a fresh task cohort. Each attempt performed an idempotent source write followed by the actual public-test command in the network-disabled, bounded Colima container. In the fault condition the worker received SIGKILL after the test succeeded and the tool-operation ledger was fsynced, but before the second `AgentStep` was committed. Three repetitions were run per arm and condition. A separate audit re-ran the isolated public test and checked every store.

| Condition | Arm | Verified | Tool executions / run | Duplicate operations / run | Median wall to final validation | Median recovery time |
| --- | --- | ---: | --- | --- | ---: | ---: |
| No fault | Basic | 3/3 | 2 | 0 | 0.447 s | 0 s |
| No fault | VARE durable | 3/3 | 2 | 0 | 0.459 s | 0 s |
| SIGKILL | Restart from zero | 3/3 | 4 | 2 | 0.664 s | 0.263 s |
| SIGKILL | VARE resume | 3/3 | 3 | 1 | 3.772 s | 3.350 s |

Both fault arms recovered correctness. VARE avoided redoing the committed source-write operation, but correctly repeated the uncommitted public-test tool. It reduced tool executions from four to three and duplicates from two to one; lease expiry made recovery about 3.1 s slower than the basic restart in this fixture. This is a measured reliability/cost tradeoff, not an efficiency gain. External tools remain at-least-once across an uncommitted boundary; callers must use idempotency or detect duplicates.

The run covers one tool-boundary fault, one development task, one local host, and a fixed 3 s lease. It does not measure model inference interruption, an interrupted SQLite write, delayed/timeout hidden grading, duplicate publication, long-running workloads, or a broad worker fault distribution. The task is too small to estimate reliability rates.

## Research decision

The three-arm comparison (basic agent, basic plus search, VARE verification/candidate selection) was **not run**. The current inventory contains consumed development/evaluation tasks and only a two-task exploratory v3 cohort; no new independent task family set was ready to freeze. Reusing those tasks would not support the requested held-out comparison. New task authoring and grader adversarial review are required before that measurement.

The No-update / SFT / GRPO held-out comparison is also **NO-GO**. The 0.5B result only establishes that a ten-step adapter update can execute and reload. There is no fresh base-model success gate or untouched confirmation task set for this post-training pairing. GRPO was not run and no GRPO implementation was added.

## Reproduction and evidence

The two inference/training protocols are [`mlx_backend_microbenchmark_v1.lock.json`](../protocols/mlx_backend_microbenchmark_v1.lock.json) and [`mlx_lora_update_smoke_v1.lock.json`](../protocols/mlx_lora_update_smoke_v1.lock.json). The recovery protocol and runners are [`agent_tool_recovery_v1.lock.json`](../protocols/agent_tool_recovery_v1.lock.json), [`run_agent_tool_recovery_v1.py`](../scripts/run_agent_tool_recovery_v1.py), and [`audit_agent_tool_recovery_v1.py`](../scripts/audit_agent_tool_recovery_v1.py). The raw run records, SQLite stores, candidate files, adapter, logs and hash manifests live under [`results/`](../results/). Base model snapshots are not committed; their revisions and file hashes are retained in the records and can be fetched without paid APIs. The exact adapter checkpoint is retained in the accompanying evidence bundle and pinned by SHA-256; it is omitted from Git because each raw weights file exceeds this repository’s 1 MiB tracked-blob budget. Pass the extracted `adapters.safetensors` using `--adapter-checkpoint` when rerunning the independent auditor. No converted base-model weights are needed to reproduce the conversion.

The E0 canary checked that `/grader` was absent in candidate containers and that network access failed. This is an integration control for these runs, not a general hostile-code sandbox certification. PR #90 remains a draft with earlier broader benchmark CI failures; this change adds regression fixes for the pack-manifest fixture and removes tests' writes to the host's absolute `/grader` path. CI must be rerun on this head before considering promotion.

## Limitations and next gates

1. Author and independently audit a new task cohort before freezing the Basic / Search / VARE comparison.
2. Measure same-family Qwen 4B task success on that fresh cohort before any 4B update study.
3. Compare no-update and SFT only after a fresh base gate; include untouched held-out tasks and multiple seeds. Treat GRPO as a later gate.
4. Extend recovery to interrupted generation, interrupted persistence, grader delay/timeout, and duplicate publication. Measure lease tuning only under a new frozen protocol.
5. Do not describe this inference, SFT smoke, or deterministic recovery fixture as recursive self-improvement.
