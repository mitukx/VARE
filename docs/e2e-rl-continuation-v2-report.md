# Continuous CPU RL updates and learner recovery

**PASS for the declared single-host systems protocol; no capability claim.**
On 2026-10-16, Apple M1 Pro / 32 GiB / 10 CPU cores / macOS 27.0.1 /
Python 3.11.15, all **42 cases and 504 logical rounds** completed. There were
**513 actual candidate AdamW attempts**: nine uncommitted computations were
repeated after a kill, and no committed update was applied twice. All 18 actual
learner SIGKILL/restart cases matched their uninterrupted reference exactly.
The independent audit verified **6,522 files, 504 candidate tensor states** and
recomputed retained checkpoint evaluation scores.

## Scope and implementation

The [protocol](../benchmarks/e2e_rl_continuation_v2/protocol.json) and executable
sources were committed as `56ba2fefc69a8851f0457646b7031fe8bf2f2c80` before the
main measurement. This addresses two limits of the preserved
[v1 actual-model experiment](e2e-rl-real-model-v1-report.md): subsequent workers
must serve the updated weights, and a restarted learner must execute the next
real optimizer update. It reuses v1's actual REINFORCE/AdamW backend and VARE's
existing engine, lag controller, replay, reward/evaluation evidence, checkpoint
store and promotion journal. Production control-plane code is unchanged.

The public `sshleifer/tiny-gpt2` revision is
`5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`: 102,714 parameters, 2,200 trainable,
CPU fp32, one PyTorch thread per process. Frozen tied embeddings reduce update
cost. Each of 12 rounds uses 32 sampled constrained A/B tokens and 32 disjoint
evaluation prompts. Seeds **2701/2702/2703** create new round-specific nonce
namespaces; no v1 tasks, retired dataset or confirmation cohort is read.
Evaluation runs actual incumbent/candidate checkpoints in separate processes.
Acceptance is nondecreasing mean probability of A, with no statistical claim.
All 12 candidates were accepted in every main trial.

Persistent rollout workers load the immutable active checkpoint on hash change
and acknowledge the checkpoint SHA, actual model tensor fingerprint and step.
The controller checks these before verification and records them in the reward
and training provenance. A negative regression deliberately keeps old worker
weights under a new request: the boundary rejects it, then accepts a real reload.
Every round records the actual optimizer batch and model/Adam/RNG/step state.

Both comparison arms run the same tasks, numerical updates, evaluator budget
and **zero-lag** admission checks. Guarded mode additionally authenticates rewards,
binds evaluation to checkpoint bytes and journals publication. Minimal mode
disables these optional protections. This is an ablation of the same engine,
not a comparison against another RL library. The auditor confirms equality of
all paired numerical update records and active states.

## Measurements

Mean ± sample SD over **three seeds**, each with 12 rounds. Tasks/s counts
training rollouts. Each generates exactly one output token, so output tokens/s
equals tasks/s; prompt processing, training and evaluation consume elapsed time
but do not increase that token count. The normal timing boundary is dispatch
through worker synchronization, rollout, verification, replay, actual update,
independent evaluation, decision and fsynced round cursor. Model/import/pool
startup and discarded warmup are excluded. **This differs from v1, whose round
included a fresh recovery; the tables cannot establish a v1-to-v2 speedup.**

| Workers | Minimal tasks/s | Guarded tasks/s = output tokens/s | Guarded round ms | Paired wall overhead | Guarded peak RSS MiB | CPU core % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 157.75 ± 18.39 | 145.57 ± 10.70 | 220.64 ± 16.61 | +8.80 ± 15.48% | 1,442.25 | 147.07 |
| 2 | 193.57 ± 3.17 | 176.49 ± 3.55 | 181.36 ± 3.61 | +9.70 ± 2.56% | 1,774.94 | 178.14 |
| 4 | 194.97 ± 9.44 | 170.63 ± 3.02 | 187.58 ± 3.34 | +14.22 ± 3.63% | 2,447.33 | 231.25 |
| 8 | 195.33 ± 5.70 | 174.53 ± 4.13 | 183.41 ± 4.28 | +11.91 ± 0.81% | 3,785.34 | 352.06 |

Overhead is the mean of seed-paired guarded/minimal wall-time ratios minus one.
The noisy one-worker pair does not establish a consistently positive cost.
Actual maximum simultaneous worker service reached 1/2/4/8, with every configured
worker PID participating in every case. The small workload establishes no
throughput benefit beyond two workers. RSS is a 20 ms sampled process-tree sum,
including shared pages per process; shorter peaks can be missed. CPU in this
table is recomputed from retained samples as total sampled process CPU divided by
the timestamp span from first to last sample; it includes startup and 100% means
one core. The progress summary's wider sampler-duration value is audited against
this sample-derived value within 1%.

Normal cases require exactly one `round` span for each committed round. A
SIGKILL after durable publication can leave a committed recovered-publication
round without a completed timing span; those fault cases are explicitly excluded
from the warm-throughput comparison, and the auditor does not invent a duration.

Guarded case wall time, including cold startup/shutdown, averaged
10.10/9.54/10.09/10.75 s for 1/2/4/8 workers. The corresponding rollout pool
startup means were 2.04/2.10/2.24/2.88 s; evaluation pool means were
2.11/2.07/2.26/2.18 s. The complete campaign, including cold starts and faults,
took **540.84 s** before final artifact hashing. Warm round time totaled 52.97 s
across the 24 normal cases. Setup therefore materially changes short-run cost.

Guarded four-worker stage latency, pooled over three seeds:

| Stage | Observations | p50 ms | p95 ms |
|---|---:|---:|---:|
| Rollout hook, dispatch/IPC and worker sync included | 1,152 | 3.073 | 10.534 |
| Verification including reward receipt | 1,152 | 0.050 | 0.079 |
| Freshness assessment | 3,489 | 0.0026 | 0.0044 |
| Replay selection | 36 | 1.980 | 2.280 |
| Training including candidate checkpoint | 36 | 77.647 | 92.479 |
| Independent evaluation, each 32-task report | 72 | 30.054 | 37.542 |
| Promotion hook | 36 | 9.839 | 12.593 |
| Whole round including durable cursor | 36 | 183.956 | 207.099 |

[All configuration metrics](../results/e2e-rl-continuation-v2/comparison.csv) and
[stage quantiles and span sums](../results/e2e-rl-continuation-v2/stage-latency.csv)
are derived from raw spans. Concurrent/nested spans overlap; their sums are not
critical-path time. Quantiles are descriptive linear interpolations, not tail
guarantees. Genuine admitted rollouts had policy lag zero. After each first
promotion, a retained real base-policy experience was inserted as a stale
canary: **462/462 logical canaries** were screened before training, with none in
the optimizer batches. For normal trials the intentional replay rejection rate
was **11/(384+11) = 2.785%** in both arms. This is an injected rate, not operational
incident prevalence. Canary lag ranges from one to eleven accepted updates.

## Actual learner death and recovery

At round index 4, a supervisor waits for a fsynced cutpoint marker and SIGSTOP,
then **SIGKILLs the actual learner**, observes exit code -9 and kills its owned
process-group descendants. It verifies zero executing old descendants before
starting a fresh interpreter. Recovery checks pinned sources, checkpoint bytes,
model/Adam/RNG/step and journal state. An unpublished candidate is aborted and
recomputed from the incumbent; an already published candidate is checked,
independently re-evaluated where needed and rolled forward. The next actual
optimizer round must finish and persist a cursor. No mock learner supplies
these measurements.

| SIGKILL cutpoint | Recovery seconds, mean ± SD | Range seconds | Candidate attempts / logical rounds |
|---|---:|---:|---:|
| Immediately after actual AdamW | 6.476 ± 0.290 | 6.250–6.803 | 39 / 36 |
| Candidate saved | 6.601 ± 0.140 | 6.490–6.759 | 39 / 36 |
| Promotion journal prepared | 7.042 ± 0.073 | 6.957–7.091 | 39 / 36 |
| Active pointer published | 6.939 ± 0.305 | 6.705–7.283 | 36 / 36 |
| Journal committed | 6.753 ± 0.168 | 6.565–6.887 | 36 / 36 |
| Round cursor committed | 6.707 ± 0.650 | 6.317–7.457 | 36 / 36 |

Each row has three seeds. Recovery starts at SIGKILL and ends after the first
subsequent completed real optimizer round with a durable cursor, including
interpreter/import/model/pool restart. This is broader than v1's publisher or
worker recovery boundary. Every recovered round's candidate tensors, numerical
update, decision and active model/Adam/RNG/step match the uninterrupted two-worker
reference exactly. All journals end clean, with unique committed round indices
and no duplicated committed candidate. [Recovery CSV](../results/e2e-rl-continuation-v2/recovery.csv).

All main candidates passed the diagnostic promotion gate; v2 does not measure
candidate rejection frequency. The preserved v1 actual-model fault campaign
separately measures reward tampering, forged evaluation, checkpoint corruption,
evaluation cancellation and exact incumbent rollback. Those results are not
relabeled as v2 observations.

## Evidence and reproduction

Raw checkpoints, SQLite state, batches, spans, kill records and pinned sources are
retained as [GitHub release assets](https://github.com/mitukx/VARE/releases/tag/e2e-rl-continuation-v2-20261016).
The [artifact manifest](../results/e2e-rl-continuation-v2/artifact-manifest.json)
records byte counts, SHA-256 and file counts. Large new archives stay outside Git
under the existing blob policy. Diagnostic preflight and both test executions
are retained separately and excluded from headline aggregation.

From a clean checkout with Python 3.11, use a new output path:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/e2e-rl-real-model-v1.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export VARE_E2E_MODEL_DIR=/tmp/vare-tiny-gpt2-5f91d94
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
p = json.loads(Path('benchmarks/e2e_rl_continuation_v2/protocol.json').read_text())
snapshot_download(p['model'], revision=p['revision'],
                  local_dir=os.environ['VARE_E2E_MODEL_DIR'], allow_patterns=list(p['model_files']))
PY
.venv/bin/python scripts/benchmark_e2e_rl_continuation.py --output /tmp/vare-continuation-new --model-dir "$VARE_E2E_MODEL_DIR"
.venv/bin/python scripts/audit_e2e_rl_continuation.py /tmp/vare-continuation-new --model-dir "$VARE_E2E_MODEL_DIR"
.venv/bin/python -m pytest tests/test_e2e_rl_continuation.py
```

The runner refuses an existing output directory and checks model-file hashes.
Reduced budgets receive a distinct `.diagnostic` identity and
`protocol_conformant: false`; the auditor independently checks the budget and
case inventory. Do not change executable sources during a campaign: every
learner startup verifies their pinned hashes.

Audit retained evidence and regenerate tables without new training:

```bash
mkdir /tmp/vare-continuation-retained-new
gh release download e2e-rl-continuation-v2-20261016 --repo mitukx/VARE \
  --pattern m1-continuation-v2.tar.xz --dir /tmp/vare-continuation-retained-new
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('/tmp/vare-continuation-retained-new/m1-continuation-v2.tar.xz')
m = json.loads(Path('results/e2e-rl-continuation-v2/artifact-manifest.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest() == m['files'][p.name]['sha256']
PY
tar -xJf /tmp/vare-continuation-retained-new/m1-continuation-v2.tar.xz -C /tmp/vare-continuation-retained-new
.venv/bin/python scripts/audit_e2e_rl_continuation.py /tmp/vare-continuation-retained-new/continuation-v2-main --model-dir "$VARE_E2E_MODEL_DIR"
.venv/bin/python results/e2e-rl-continuation-v2/derive_tables.py /tmp/vare-continuation-retained-new/continuation-v2-main --output /tmp/vare-continuation-tables-new
```

Omit `--model-dir` for a standard-library byte/record audit; `--check-tensors`
adds PyTorch state verification without model evaluation. Local actual-model
regressions: **11 passed in 104.27 s**, including six kill boundaries, stale
worker weights, source-change refusal and rejection of falsified attempt counts.
The [CPU CI](../.github/workflows/e2e-system-cpu.yml) runs these as a reduced
correctness campaign, not the Mac performance campaign. Validation logs and
the independent audit result are under
[results](../results/e2e-rl-continuation-v2/).

The offline audit also binds a normal round's evaluation pair to the unique
completed learner attempt and its retained `RoundResult`. A tensor audit of
the same retained archive verified 504 candidate states. If a SIGKILL occurs
after durable publication but before that result is written, it instead checks
the pair against the promotion journal's persisted evaluation-metadata digest.
A rehashed `attempt99` pair was accepted by the prior auditor and is now
rejected; this correction re-audits the same 42-case archive and changes no
protocol or model measurements. The focused falsification record is retained
in [audit-review validation](../results/e2e-rl-continuation-v2-audit-review/validation.json).

The audit now also requires a rollout file for every recorded optimizer attempt
and checks progress seed, worker count, and mode against each case's retained
configuration and directory identity. Removing all round rollout files or
relabeling the retained seed while rehashing the archive passed the previous
auditor; both falsification controls now fail closed.

Each round's `training_tasks.json` is an adaptive-curriculum candidate pool;
the existing loop samples from it with replacement. The audit therefore checks
unique pool identities and requires every recorded rollout task's ID, family,
and prompt to match that pool, while preserving the recorded rollout count.
Each optimizer attempt must have one batch whose members match the same
attempt's rollout steps, policy provenance, log probabilities, and rewards.
Per-case and aggregate completed-round counts are derived from committed round
records rather than progress summaries.

A subsequent independent review found three archive-resealing gaps. The auditor
now compares the retained protocol bytes with the separately declared protocol
digest, rejects duplicate `(round, learner_attempt)` identities even when the
duplicate's rollout and batch files are removed, and reconstructs each training
and held-out task cohort from its recorded seed and round. Guarded evaluation
reports must also attest the digest of that round's retained held-out task file.
All five mutation controls (protocol digest, duplicate attempt identity,
evaluation task content in both modes, and guarded evaluation-byte attestation)
were accepted by the preceding auditor and are rejected by the corrected one.
The byte-attestation control preserves the task JSON values while changing only
its serialization, isolating the report-digest check. These are audit-integrity
corrections over the preserved campaign; they add no model, optimizer, or
performance observations.

The later review also closed seven record-linkage gaps. Attempt identities must
follow the configured kill-round schedule, every attempt that reached the
promotion gate must retain its independent evaluation pair, and rollout
log-probabilities must agree with worker values, RVL token records, and
recomputed behavior digests. Timed committed rounds are checked against trace
indices; only recovered-publication rounds, whose learner was killed before it
could close the span, lack a span and are excluded from normal throughput
aggregation. CPU and peak RSS table values now derive from `resources.jsonl`,
while redundant progress summaries are checked against those raw samples.
Every post-promotion attempt must retain exactly one stale-replay screening
event, while the canary record is matched back to its round-zero rollout; the
normal summary must contain exactly one aggregate for each retained worker/mode
pair. Rehashed controls that removed screening evidence, changed the canary
output, or emptied all normal summary groups passed the previous auditor and
are rejected by the current one. The independent canary control is retained
alongside the full actual-model module run and final tensor audit.

A further independent review found that a rehashed accepted promotion could
disagree with its retained evaluation pair, SIGKILL cutpoint markers were
not linked to the killed process, and recovery table counts trusted a
mutable progress flag. The audit now recomputes the v2 nondecreasing-mean
promotion decision from the selected independent pair, binds each cutpoint
marker to the configured round, cutpoint, learner PID/PGID, kill time,
optimizer state and boundary policy, and requires the exact-recovery flag
to agree with committed-round comparisons. Table recovery counts now
derive from those comparisons directly. The three resealed controls were
accepted by the prior auditor and are rejected by the corrected one; the
old table reported 2/3 exact cases for a forged flag while the corrected
table reports the measured 3/3. The full 53-test actual-model module passes, and the raw campaign tables
remain byte-identical to the retained results.

This is a toy neural LM and synthetic one-token workload. Replay is round-local;
general durable replay-backlog/curriculum recovery is not tested. The supervisor
assumes cooperative same-UID POSIX ownership. There is no power-loss, hostile-code,
GPU, multi-host, distributed scalability or production-throughput measurement.
The research admission gate remains unchanged; neither the diagnostic acceptance
rate nor successful actual optimization establishes held-out task capability.
