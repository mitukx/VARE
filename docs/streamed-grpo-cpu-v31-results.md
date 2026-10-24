# Streamed GRPO v3.1: executed systems result

**Run date:** 2026-10-24

**Protocol:** [`protocol-v3.1.json`](../benchmarks/streamed_grpo_cpu_v3/protocol-v3.1.json), SHA-256 `7377c2e604252bcc8ce501d440ca98f2ef453c9cf7b4cb3c62c900cfa2ba0962`

**Runner:** [`validate_streamed_grpo_cpu.py`](../scripts/validate_streamed_grpo_cpu.py), SHA-256 `8cd299fa47e2264784fe6108339d838dc92bdc955bf5c605dba7157a03ff02a2`

**VARE source revision:** `327b24d435f01030a760bc86b76aec65cfe9a70e`
**Raw evidence:** [v3.1 evidence prerelease](https://github.com/mitukx/VARE/releases/tag/streamed-grpo-cpu-v31-20261024), with a local copy in the task outputs.

## Result

The frozen v3.1 protocol completed on its three declared seeds. It connected actual inference from `roneneldan/TinyStories-1M` revision `77f1b168e219585646439073245fe87e56b3023e` to RVL's existing GRPO optimizer, authenticated durable trajectories, versioned policy and verifier identities, independent checkpoint decoding, promotion decisions, and recovery after an interrupted delivery acknowledgment. It is a small-model **systems fixture with synthetic checksum rewards**, not a coding-task experiment.

| Seed | Round wall time (s) | Completed trajectories | Consumed / quarantined | Optimizer calls / committed transactions | Peak RSS (GiB) | Recovery after lost ACK (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 82011 | 6.308 | 16 | 12 / 4 | 4 / 3 | 0.792 | 86.6 |
| 82012 | 6.234 | 16 | 12 / 4 | 4 / 3 | 0.774 | 86.6 |
| 82013 | 6.151 | 16 | 12 / 4 | 4 / 3 | 0.782 | 87.6 |
| Mean ± sample SD | 6.231 ± 0.079 | 16 | 12 / 4 | 4 / 3 | 0.783 ± 0.009 | 86.9 ± 0.6 |

Across all seeds, 48 of 48 trajectories completed and received synthetic verification; 36 were admitted and consumed by training, and 12 uncertain trajectory records (three complete groups) were quarantined after the injected lost-ACK boundary. There were 12 optimizer calls, 9 committed candidate transactions, and 3 forced promotion rejections. No duplicate effective trajectory commits were found. The measured useful-consumption rate was 6,934 trajectories/hour (seed range 6,849–7,023); total completion rate was 9,245/hour (9,132–9,364). These rates divide the recorded four-round wall time and exclude model setup. They describe this tiny CPU fixture only.

Each run trained 131,200 parameters on CPU fp32 with one Torch thread. Peak RSS was 0.774–0.792 GiB; measured process CPU use was 111.8–113.1% of one core. Host swap was not recorded. The three seeds each ran four rounds, with four completions per GRPO group, two durable steps per trajectory, two execution workers, and one verifier. The model lock serialized model access.

The offline numerical replay passed for all 12 optimizer calls. Across runs, maximum absolute discrepancies were `4.77e-7` for pre-clip gradients, `2.98e-8` for clipped gradients, `3.73e-9` for updated parameters, `3.73e-9` for Adam state, and `1.87e-8` for loss; all were below the frozen `2e-6` bound. Maximum behavior-logprob alignment error was `5.06e-5`, below `1e-4`. RNG state matched for every replay.

The deliberately interrupted round failed after the model transaction but before the learner received its durable acknowledgment. For each seed, recovery reloaded model, Adam, RNG, and step state from the saved active checkpoint, reconciled the existing promotion journal, quarantined the four uncertain trajectories, and completed a later optimizer transaction. This was an in-process logical restart, **not** learner SIGKILL or power-loss recovery. Promotion transactions show the round-two candidate was already committed before the acknowledgment was lost; quarantine therefore records uncertain input delivery without pretending that the model transaction did not occur.

## Integrity and limits

- The run used the actual pretrained model and optimizer, but the training reward and independent diagnostic score are deterministic text checksums. They do not establish semantic quality.
- All checkpoint evaluations stayed at `0.875` over four diagnostic prompts. Candidate tie promotion was permitted by the frozen protocol, so these committed updates produced no measured evaluation gain.
- Freshness was enforced at zero allowed policy/verifier lag, with one verifier identity. The experiment did **not** inject verifier drift or quantify a stale-rollout rejection rate.
- There was no matched synchronous or naive-asynchronous optimizer arm. The asynchronous workers used serialized model access. No throughput advantage, protection overhead, or comparative queue-latency result is established.
- The only injected runtime fault was the lost learner-ingestion acknowledgment. Worker death, learner SIGKILL at multiple transaction boundaries, and power-loss recovery were not tested.
- The independent auditor recomputed the full training equation from stored inputs and checked the optimizer, RNG, and parameter states. We also independently checked all 79 embedded source-file hashes per seed against Git objects at the recorded VARE revision (237 checks). The audit manifest copied from each run is a post-run integrity anchor, not an independent pre-run chain-of-custody record. The public archive hash now anchors the distributed bytes.
- The protocol did not pin a complete Python dependency lockfile. The execution environment recorded Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, and psutil 7.1.3. The pretrained model files and RVL revision/backend patch are pinned in the protocol.

### Decision

**GO:** the locally feasible path can perform repeated real model inference and real optimizer transactions through authenticated durable asynchronous ingestion, and it can reconcile this specific uncertain-ack boundary without duplicate effective trajectory consumption.

**NO-GO:** no claim of coding capability improvement, useful learning signal, asynchronous throughput improvement, verifier-drift robustness, or process-crash recovery. A matched synchronous/naive/VARE comparison is still necessary before claiming an asynchronous correctness-throughput tradeoff. The prior coding-agent study remains NO-GO and its consumed tasks were not rerun.

This should be read as an integration and recovery feasibility result, not an asynchrony novelty claim. Prior work already studies asynchronous post-training, stale-rollout control, and multi-version consistency: [AReaL](https://arxiv.org/abs/2505.24298), [StaleFlow](https://arxiv.org/abs/2601.12784), [DORA](https://arxiv.org/abs/2604.26256), and [PACE](https://arxiv.org/abs/2609.36830). This experiment does not compare against those systems or demonstrate a new throughput method.

## Provenance and reproduction

The run used Apple M1 Pro / 32 GB, CPU fp32, one Torch thread; TinyStories model revision above; RVL revision `e381c4b3b8c2ad8e5dc4c4b1a33b7922531c3e33` with backend patch SHA-256 `a35179d2ff65b35d8079b0d48a933444cac440debd736d6dc6a7a53a27cfb960`. The report and raw outputs do not change the frozen protocol or runner.

Download and verify the release asset, fetch the exact model files and RVL revision, apply the pinned RVL backend patch, then run:

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

Use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1`. The raw archive is approximately 367 MiB and SHA-256 `5c3714294f417a6fc3f1dbf2897fad8fc98d4d8406064e12128229a36df1abef`. The self-copied run manifest detects post-run file changes but is not an independently signed run-time manifest; the independent numerical replay and the source-to-Git checks above are separate validation steps.
