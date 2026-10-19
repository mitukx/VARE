# Benchmark and evidence guide

This page indexes selected system and training-integrity experiments. The report, protocol lock, and retained raw artifacts are authoritative; this summary does not replace them. Results use different workloads and cannot be ranked as if they measured the same outcome.

## Continuous CPU post-training control loop

- **Question:** Can a guarded control loop continue actual candidate optimizer updates after learner interruption while binding worker acknowledgments to updated checkpoint and model state?
- **Workload and hardware:** 12-round continuation campaign, pinned public tiny GPT-2 (102,714 parameters; 2,200 trainable), fresh synthetic one-token tasks, Apple M1 Pro / 32 GiB.
- **Comparison:** Matched minimal-control and guarded modes using the same engine, tasks, evaluation budget, and numerical updates. Guarded throughput was 145.57–176.49 tasks/s across three seeds and 2–8 rollout workers; its measured overhead was 8.80–14.22%.
- **Observed result:** 42/42 declared cases passed; 504 logical rounds and 513 actual AdamW candidate attempts. Eighteen learner SIGKILL/restart cases reproduced uninterrupted model, Adam, RNG, step, and subsequent-update state exactly. Nine uncommitted computations were repeated; no committed update was applied twice.
- **Limit:** Single-host toy-model systems validation on synthetic tasks. This does not measure held-out capability, production LLM throughput, or GPU/distributed behavior.
- **Evidence:** [report, measurements and reproduction](e2e-rl-continuation-v2-report.md), [protocol](../benchmarks/e2e_rl_continuation_v2/protocol.json), [retained release and artifact hashes](../README.md#key-results).
- **Reproduce:** Follow the report's pinned model setup, environment, and fresh output-directory commands. To audit retained bytes and telemetry without training, extract the release archive and run its documented auditor.

## Durable asynchronous trajectories

- **Question:** What throughput and queue-latency tradeoff appears when durable multi-step trajectories execute continuously instead of in matched synchronous cohorts?
- **Workload and hardware:** Synthetic trajectories, three seeds, eight execute workers and two verifiers, single-host CPU; 10,000 trajectories per mode and seed, with actual worker SIGKILL and independent artifact audits.
- **Baseline and result:** Matched synchronous cohorts measured 36.16 ± 2.68 trajectories/s; continuous async measured 54.53 ± 1.85 trajectories/s. Async queue-inclusive end-to-end p95 was 44.04–46.45 s versus 0.240–0.290 s for sync. These are per-run ranges and three observations, not pooled estimates.
- **Limit:** The benchmark consumer acknowledges replay eligibility and performs no optimizer update. Both modes used the same correctness guards, so overhead versus an unprotected system was not measured. No model inference, learner SIGKILL, or distributed scaling is represented.
- **Evidence:** [report and stage latency tables](async-trajectory-runtime.md#measured-cpu-result), [protocol](../benchmarks/trajectory_runtime_v2/protocol.json), [raw-evidence release](https://github.com/mitukx/VARE/releases/tag/trajectory-runtime-cpu-20261017).
- **Reproduce:** The report includes focused checks, Stage A/B commands, per-case audits, source requirements, and instructions for new output directories. Stage C was not run; semantic task qualification remains NO-GO.

## Multi-completion GRPO numerical validation

- **Question:** Do candidate updates from the existing external GRPO trainer match an independent complete-group numerical reference and preserve rollback behavior?
- **Workload and hardware:** Three CPU seeds, TinyStories-1M with 131,200 trainable parameters, synthetic checksum rewards, and synthetic diagnostic evaluation. Apple M1 Pro / 32 GiB; CPU fp32, one Torch thread.
- **Result:** The final v5 protocol completed 36 actual candidate updates (12 per seed), with independent comparisons of loss, pre-clip and clipped gradients, Adam tensors, and updated parameters. Maximum discrepancies stayed below the report's fixed tolerance. Eight deliberately invalid-data/evidence controls per seed were rejected.
- **Baseline and limits:** The reference computes the complete group-mean loss graph independently of the external trainer's per-response accumulation. An unmerged patch to the pinned external trainer was required for the experiment. Rewards and evaluation criteria were synthetic; this is optimizer-integrity evidence, not semantic task success or capability gain. The reported throughput has no unguarded learning baseline.
- **Evidence:** [v5 report and reproduction](grpo-e2e-cpu-report.md), [frozen protocol](../benchmarks/grpo_e2e_cpu_v5/protocol.json), [report-linked raw validation and release assets](grpo-e2e-cpu-report.md#correctness-and-retained-failures).
- **Reproduce:** Use the report's Python/Torch setup, pinned model snapshot, pinned external source and patch, three declared seeds, and new output directories. The read-only artifact audit is documented separately from a new training run.

## Actual-model fault benchmark v1

- **Question:** Do freshness, evidence-integrity, evaluation-lifetime, promotion, and recovery guards reject declared faults in a single-round control loop with an actual neural optimizer update?
- **Workload and hardware:** Pinned tiny GPT-2 (102,714 parameters; 2,200 trainable), fresh synthetic one-token tasks, Apple M1 Pro / 32 GiB.
- **Baseline and result:** The report compares minimal controls with guarded mode under a fixed one-round protocol, including decision and checkpoint-recovery time. It records 78/78 declared case oracles and 63 actual optimizer steps. Guarded mode blocked 15 injected invalid-data/evidence cases admitted by the minimal arm.
- **Limit:** This is not the continuation v2 timing boundary or baseline. It is single-round toy-model systems validation; greedy task success did not improve. Long-running updated-worker synchronization, a next optimizer step after recovery, production throughput, and accelerator/distributed behavior were not measured.
- **Evidence:** [report, latency, faults and reproduction](e2e-rl-real-model-v1-report.md), [protocol](../benchmarks/e2e_rl_real_model_v1/protocol.json), [comparison CSV](../results/e2e-rl-real-model-v1/comparison.csv).

## Other selected system and research records

| Record | Workload and interpretation | Evidence |
| --- | --- | --- |
| Verifier refresh controls | Synthetic visible and within-family shifts. TV-triggered refresh missed the hidden within-family proxy error; periodic refresh removed it under this fixture. No model was trained. | [Report](verifier-refresh-controls.md) |
| Four-parameter control-loop comparison | Earlier synthetic policy simulation: VARE measured 182 versus 300 rollouts/s across 12 paired seeds. A different workload and baseline than actual-model benchmarks; no neural update. | [Report and errata](e2e-rl-system-benchmark-v1.md) · [review corrections](e2e-rl-system-benchmark-v1-errata.md) |
| OpenBookQA GRPO/SFT confirmation | A specific Qwen2.5-0.5B task/reward pairing produced a negative result under its frozen parser. It is not a general verdict on GRPO. | [Report](openbookqa-qwen25-grpo-sft-confirmation-v1-report.md) |
| Candidate task qualification | Latest screen retained a NO-GO before base inference; no model or optimizer was run for that screen. | [Report](research-candidate-screen-2026-10-16.md) |

The [experiment index](experiments.md), [evidence ledger](evidence.md), and [protocol directory](../protocols/) preserve historical, superseded, failed, and negative records. Do not substitute a newer summary for a frozen protocol or original report.

## Cross-cutting measurement limits

- Synthetic trajectory throughput is not model-inference or production-serving throughput.
- Optimizer correctness, diagnostic proxy rewards, and a successful checkpoint save do not establish capability improvement.
- Independent numerical replay is not the same as an independent human review or independently graded task success.
- Correctness under injected faults and single-host CPU conditions does not establish a production failure rate, distributed atomicity, or resilience to hostile code.
- Each report defines its own baseline, clock boundary, sample unit, environment, and audit scope. Read those definitions before comparing measurements.
