# End-to-end RL system: actual model backend

**PASS for bounded single-host systems validation; no capability result.** On
2026-10-16, Apple M1 Pro / 32 GiB / macOS 27.0.1 / Python 3.11.15 / CPU fp32,
the campaign completed **78/78 declared cases**, including 24 normal rounds,
54 fault cases and **63 actual AdamW updates**. An offline audit verified 1,025
artifacts and replayed retained evaluation scores from actual checkpoint tensors.
The [raw campaign archive](../results/e2e-rl-real-model-v1/m1-cpu.tar.xz) is the
source of truth. It preserves all 1,025 files, including model/optimizer
checkpoints, raw telemetry and source snapshots, in an 870,756-byte archive.

The earlier [synthetic benchmark](e2e-rl-system-benchmark-v1.md), its protocol and
results are preserved. Its four-parameter policy cannot answer the actual-model
update/reload question. This separate backend uses the existing CapabilityLoop,
LagController, PrioritizedReplay, RewardEvidenceRoot, EvidenceTrustRoot,
TrustedCheckpointStore and PromotionJournal; production code is unchanged.
Existing CI and lifecycle/replay/checkpoint regressions were inspected before
adding only the model benchmark, audit and optional regression job.

## Workload and comparison

[Protocol](../benchmarks/e2e_rl_real_model_v1/protocol.json): public
`sshleifer/tiny-gpt2` checkpoint revision
`5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`, 102,714 transformer parameters,
2,200 trainable parameters with the tied token embedding/output matrix frozen.
This is a toy neural LM, not a representative production or reasoning model.
Each round samples 32 actual one-token A/B actions, verifies the return-A reward,
rescreens replay, performs one REINFORCE/AdamW update and independently scores
both checkpoints on 32 disjoint nonce-tagged prompts in separate processes.
Fresh model and Adam instances then reload the active checkpoint; every model
tensor, Adam state, RNG state and step counter must match exactly.

Seeds 1701/1702/1703 produce new tasks; no retired data or confirmation cohort
was loaded by the benchmark. Every seed/configuration starts from the same base
checkpoint. Minimal and guarded modes use the same engine, grouped replay,
tasks, numerical optimizer update and evaluation budget. The audit confirms
identical update records in every normal pair. Minimal mode disables optional
reward receipts, checkpoint-bound evaluation and durable publication, and allows
lag up to 999. Guarded mode enables these controls with maximum lag zero.
Mandatory engine identity/group/report/lifetime checks remain in both modes:
this is an optional-protection ablation, not a comparison with another RL library.

## Measurements

Values are mean ± sample SD over three seeds. End-to-end timing starts at round
dispatch and includes decision plus fresh checkpoint recovery; incumbent setup,
discarded warmup and pool startup are excluded. Output tokens/s equals tasks/s
because each rollout generates exactly one constrained token. Prompt processing,
64 evaluation forwards and training are included in elapsed time, not counted
as generated output tokens. Mode order alternates by seed.

| Workers | Minimal tasks/s | Guarded tasks/s (= tokens/s) | Guarded wall ms | Wall overhead | Guarded peak RSS MiB | CPU core % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128.97 ± 5.78 | 123.72 ± 2.57 | 258.73 ± 5.41 | +4.13% | 1,442.95 | 105.67 |
| 2 | 147.45 ± 2.07 | 125.28 ± 0.47 | 255.43 ± 0.96 | +17.68% | 1,849.03 | 109.59 |
| 4 | 145.17 ± 0.75 | 134.01 ± 0.19 | 238.79 ± 0.33 | +8.33% | 2,584.98 | 120.72 |
| 8 | 144.82 ± 0.79 | 133.95 ± 4.73 | 239.10 ± 8.44 | +8.21% | 3,602.52 | 122.17 |

Actual maximum simultaneous worker service reached 1/2/4/8; all configured
worker PIDs participated. No end-to-end speedup beyond four workers is established.
RSS is the maximum 20 ms sampled sum across the controller and descendants;
shared pages are counted per process and shorter peaks may be missed. CPU is
sampled mean process CPU time, with 100% representing one core. Warm pool startup
took 1.82/1.96/2.17/2.46 s for 1/2/4/8 rollout workers; the evaluation pool took
1.92 s. The entire measured campaign took 52.04 s including setup and faults.

Guarded four-worker latency, pooled across the three repetitions:

| Stage | Observations | p50 ms | p95 ms |
|---|---:|---:|---:|
| Rollout hook (dispatch/IPC included) | 96 | 1.938 | 8.505 |
| Verification including receipt | 96 | 0.041 | 0.058 |
| Freshness assessment | 288 | 0.002 | 0.006 |
| Replay selection | 3 | 1.580 | 1.747 |
| Training including candidate checkpoint | 3 | 75.881 | 77.274 |
| Independent evaluation, per 32-task report | 6 | 23.267 | 24.573 |
| Promotion hook | 3 | 6.329 | 6.360 |
| Fresh checkpoint recovery | 3 | 88.006 | 88.379 |
| Whole end-to-end round | 3 | 238.935 | 239.013 |

[Comparison CSV](../results/e2e-rl-real-model-v1/comparison.csv) includes all
configurations, stage percentiles, E2E variation, memory and CPU. Raw spans retain
start/end times and worker service/CPU times; evaluation files retain individual
task latencies. Concurrent span sums overlap and are not critical-path time.
Three-round p95 values are descriptive interpolated quantiles, not tail guarantees.

## Fault results and recovery

All nine fault types ran three times in each mode, using two rollout workers.
The 15 invalid-data/evidence cases in minimal mode admitted stale policy/replay,
tampered reward, forged promotion metrics or trailing checkpoint corruption;
guarded mode blocked all 15 at the intended boundary. Normal policy lag and
replay rejection were zero. Injected policy lag one was rejected before replay;
when the verifier advanced after insertion, 32/32 replay entries were rejected
in each guarded run, with no optimizer call. Reward tampering also stopped before
AdamW; forged evaluation and corrupt checkpoint cases restored the incumbent.

Exceptions after actual optimizer steps and cancellation during evaluation
restored exact incumbent state in both modes. Rollout workers really exited with
`os._exit(73)`; publisher processes really exited with code 74 after fsync/replace
and before acknowledgment. The guarded journal fenced ambiguous publication,
then was reopened and reconciled against the actual saved candidate before
clearing its round reservation. Recovery mean ± SD (three guarded observations):

| Failure | Measured recovery boundary | Seconds |
|---|---|---:|
| Rollout worker exit | Detected failure → replacement pool + successful real inference | 2.042 ± 0.076 |
| Evaluation cancellation | Cancellation request → drained work + fresh incumbent reload | 0.118 ± 0.007 |
| Publisher exit | Detected failure → fresh candidate reload + journal reconciliation | 0.095 ± 0.001 |

These are local process failures and exception injections, not power-loss,
multi-host, hostile-code isolation or GPU measurements. Signing keys are
ephemeral same-UID benchmark trust roots; offline audit replays actual scores
rather than claiming to independently authenticate those keys. Multi-member
GRPO groups and tool-trajectory policies are covered by existing tests, not this
one-token workload.

## Research boundary and reproduction

Systems promotion requires nondecreasing held-out mean probability of the
correct constrained token, with no statistical capability gate. Strict greedy
success remained unchanged (9.38–18.75% across seeds). The fixture is deliberately
simple and does not qualify a new learning study. No model capability, reasoning,
pretraining quality, distributed scaling or training-efficiency improvement is
claimed; the repository research no-go remains in force.

From a clean checkout, with Python 3.11 and a new output path:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/e2e-rl-real-model-v1.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
.venv/bin/python scripts/benchmark_e2e_rl_real_model.py --output /tmp/vare-real-e2e-new
.venv/bin/python scripts/audit_e2e_rl_real_model.py /tmp/vare-real-e2e-new
```

The runner downloads only six public files at the pinned model revision and
checks their SHA-256. To replay actual checkpoint scores, pass the downloaded
snapshot path as `--model-dir` to the auditor. Set `VARE_E2E_MODEL_DIR` to that
path and run `python -m pytest tests/test_e2e_rl_real_model.py` for the nine
actual-model regressions. The isolated CPU CI job runs both tests and a small
regression campaign; its timings are not the Mac performance results.

To audit the retained Mac evidence instead of running a new campaign:

```bash
mkdir /tmp/vare-retained-new
tar -xJf results/e2e-rl-real-model-v1/m1-cpu.tar.xz -C /tmp/vare-retained-new
.venv/bin/python scripts/audit_e2e_rl_real_model.py /tmp/vare-retained-new/m1-cpu
```

Local validation: actual-model tests **9 passed**; default suite excluding the
separately run blob-budget and optional math graders **913 passed, 8 skipped**;
blob-budget tests **5 passed**; Ruff and diff checks passed. The skipped eight
model cases ran in the isolated nine-test invocation. Initial incomplete
preflights and environment/partial-checkout test failures are retained under
[pilot archive](../results/e2e-rl-real-model-v1/pilots.tar.xz) and
[validation](../results/e2e-rl-real-model-v1/validation/), excluded from final
performance aggregation. Existing frozen protocols and results were not edited.
