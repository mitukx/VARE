# Streamed real-model GRPO integration

**PASS for the retained v3 systems run.** Three seeds each completed four actual updates with the existing RVL GRPO optimizer. The path exercised durable multi-step rollouts, authenticated rewards, policy/verifier freshness, whole-group replay, independent checkpoint decoding, promotion/rejection, and recovery after a lost ingestion acknowledgment. This is a synthetic systems diagnostic; it establishes no semantic task success or model capability gain. The historical v3 run did not validate distribution-shift rejection: its shift reference was derived from the batch under review. Current streaming code requires an independent reference, covered by new positive and negative regression tests; the v3.1 real-model protocol below has not been run.

The implementation reuses `CapabilityLoop`, `RVLGRPOHooks`, `PrioritizedReplay`, the existing `PromotionJournal`, evaluator, and checkpoint API. The new adapter only owns bounded delivery from verified durable groups into the existing learner transaction. A failed or uncertain delivery is quarantined after backend recovery; it is never silently retrained. PR #69's recovery path remains separate and does not demonstrate a GRPO learner process kill.

## Fixed workload and evidence

The retained v3 runs used [`benchmarks/streamed_grpo_cpu_v3/protocol.json`](../benchmarks/streamed_grpo_cpu_v3/protocol.json), SHA-256 `7954ac550a302335bf6c22e6ec009f24eb03d55063ce8441d137c8591d77a9d4`, recovered byte-for-byte from the raw artifact. Those runs use seeds 80011/80012/80013, the public `roneneldan/TinyStories-1M` revision `77f1b168e219585646439073245fe87e56b3023e`, RVL revision `e381c4b3b8c2ad8e5dc4c4b1a33b7922531c3e33` plus the portable backend patch from PR #71, CPU fp32, and one Torch thread on Apple M1 Pro / 32 GiB. The learner updates 131,200 position-embedding and final-layer-normalization parameters. Dependencies and model-file hashes are retained in the protocol and [environment record](../results/streamed-grpo-cpu-v3/environment.json).

After review, the current runner was versioned separately as [`protocol-v3.1.json`](../benchmarks/streamed_grpo_cpu_v3/protocol-v3.1.json), SHA-256 `344a28a6e0b66eec362512fc43b9c42cf65bd9bf45bbb7578746326c96ecbfed`. It names the independent expected training family `train` and fresh seeds 82011/82012/82013. It has not been executed, so all measurements below remain v3 results. A prior tracked file at SHA-256 `1976fce…` was an unrun protocol version; it did not produce the retained runs.

Each seed runs four rounds with four completions per group, two durable agent steps per trajectory, two execute workers, one verifier, backlog 16, and replay capacity 4. The first step records a deterministic lookup observation; the second invokes the real model for three generated tokens. Training reward is a signed text checksum. Independent greedy evaluation uses disjoint prompts and a different synthetic checksum. Both are correctness signals for this fixture, not task-quality graders.

The fault at round two occurs after the model transaction but before durable delivery acknowledgment. The process remains alive: the fixture reloads model, Adam, RNG and module modes, recovers the existing journal owner, quarantines the four uncertain experiences, and then performs the next real optimizer update. This is same-process logical recovery, not learner SIGKILL or power-loss recovery. Per seed the final ledger contains 12 consumed and 4 quarantined trajectories, with no duplicated effective steps or completions.

| Seed | Updates | Whole run, s | Trajectories/s | Generated tokens/s | Peak RSS, MiB | ACK recovery, ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80011 | 4 | 5.323 | 3.006 | 9.018 | 656.52 | 81.55 |
| 80012 | 4 | 5.495 | 2.912 | 8.736 | 650.05 | 87.38 |
| 80013 | 4 | 5.322 | 3.006 | 9.019 | 644.69 | 82.44 |
| Mean ± sample SD | 4 | 5.380 ± 0.099 | 2.975 ± 0.054 | 8.924 ± 0.163 | 650.42 ± 5.92 | 83.79 ± 3.14 |

Throughput counts 16 two-step trajectories and 48 generated response tokens per seed, divided by the recorded four-round wall time. The timing excludes model setup. It is specific to this tiny-model fixture and is not a serving or distributed-scaling estimate.

Recorded stage latency samples use nearest-rank p50/p95. Evaluation includes both incumbent and candidate checkpoint decoding. Verification wall time was not recorded separately, so no verifier latency is claimed.

| Stage | Samples | p50, ms | p95, ms |
| --- | ---: | ---: | ---: |
| Model rollout | 48 | 12.101 | 14.196 |
| Actual optimizer update | 12 | 210.130 | 235.624 |
| Independent checkpoint evaluation | 24 | 102.986 | 194.868 |
| Whole learner round | 12 | 1,314.896 | 1,447.184 |

The round-one promotion gate rejects its deliberately impossible gain for every seed, restoring the incumbent exactly. Independent offline replay of all 12 updates recomputes the existing full-group loss and compares pre-clip and clipped gradients, Adam state, RNG and updated parameters. Maximum absolute discrepancies across the three seeds were:

| Loss | Pre-clip gradient | Clipped gradient | Adam state | Updated parameters |
| ---: | ---: | ---: | ---: | ---: |
| 2.24e-8 | 7.16e-7 | 8.94e-8 | 1.12e-8 | 3.73e-9 |

All are below the frozen `2e-6` tolerance; maximum behavior-logprob alignment error was `1.51e-5` against the `1e-4` limit. The recovery checkpoint is used for the subsequent actual update. These comparisons validate this optimizer path and data boundary; the synthetic evaluation scores do not show learning quality.

The run used one process with asynchronous worker tasks and serialized model access. No matched throughput baseline or protection-overhead control was run for this learner adapter. A separate Stage B systems benchmark compares synchronous and asynchronous trajectory scheduling, but does not include an optimizer. No 50K run, multihost execution, or qualified semantic task was tested.

## Reproduce

Use the Python 3.11/model/RVL setup and exact patch documented in the [GRPO CPU report](grpo-e2e-cpu-report.md). Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1` after fetching the pinned inputs. Outputs must be new directories.

```bash
for seed in 82011 82012 82013; do
  out="/tmp/vare-streamed-grpo-$seed"
  python scripts/validate_streamed_grpo_cpu.py \
    --seed "$seed" --output "$out" \
    --model-dir /path/to/tinystories-1m --rvl-source /path/to/rvl-patched
  cp "$out/SHA256.json" "/tmp/vare-streamed-grpo-$seed-files.json"
  python scripts/audit_streamed_grpo_cpu.py "$out" \
    --model-dir /path/to/tinystories-1m --rvl-source /path/to/rvl-patched \
    --expected-manifest "/tmp/vare-streamed-grpo-$seed-files.json"
done
```

The copied manifest lets the offline auditor detect later file changes; for a fresh v3.1 run it is self-produced, not an independent timing anchor. Published v3 runs use the committed per-file anchors in [`results/streamed-grpo-cpu-v3/cases/`](../results/streamed-grpo-cpu-v3/cases/) and the public [raw-evidence release](https://github.com/mitukx/VARE/releases/tag/streamed-grpo-cpu-20261017). Run the offline auditor against those anchors to reproduce the reported numerical checks without regenerating data. The auditor loads each historical VARE package from its hash-pinned source archive (`audit-source-2b33abf.tar.xz`, `audit-source-5a81dfe.tar.xz`, or [`audit-source-e4155f0.tar.xz`](../benchmarks/streamed_grpo_cpu_v3/audit-source-e4155f0.tar.xz)), so V1–V3 audits do not depend on old Git commits being present in the checkout.

Next system bottleneck: actual learner-process SIGKILL around optimizer/checkpoint publication, followed by cold model/optimizer/RNG restore and a later GRPO update. It must reuse the existing recovery journal and remain separate from the still-NO-GO semantic task study.
