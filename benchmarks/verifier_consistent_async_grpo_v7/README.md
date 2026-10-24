# Multi-scenario verifier drift and sustained recovery

This v7.3 preregistration tests whether verifier changes can contaminate real GRPO
updates in a delayed asynchronous queue, and whether VARE rejects the stale
group while continuing to train on a fresh group already waiting behind it.
TinyStories-1M sampling and RVL optimizer updates are real. Tasks and rewards
are synthetic; this does not measure coding capability or natural verifier
drift prevalence.

## Preregistration amendments and retained preflight failures

The v7.1 and v7.2 protocol snapshots are preserved in
`protocol-v7.1-failed-preflight.json` and
`protocol-v7.2-failed-preflight.json`, with their amendment records. The v7.1
invocation stopped before model instantiation because the runner looked for a
seed at the wrong JSON path. The v7.2 invocation initialized and checkpointed
the model, then stopped before any completion, verifier call, or optimizer
update because it looked for evaluation prompts at the wrong path. Both failure
inventories remain at `results/verifier_consistent_async_grpo_v7/run-1/` and
`results/verifier_consistent_async_grpo_v7/run-2/`. v7.3 changes only nested workload-field access and adds protocol-shape
validation before model construction; it binds the amendment in
`protocol-amendment-v7.3.json`. The measured run uses a new output directory.

## Frozen primary study

Three drift semantics are preregistered: false acceptance, false rejection, and
reward-ranking inversion, plus a no-drift condition. Each arm uses the same
checkpoint seed, 40 model completions, 40 verifier calls, and four optimizer
commits. Sync processes groups in order. Async first generates old-policy
groups 0 and 1, trains group 1, then samples group 2 from the promoted policy
while group 0 is still buffered. The queue therefore contains both policy
versions and, in drift scenarios, both verifier versions. VARE's store scans
the stale group first, quarantines it, then consumes fresh group 2 in the same
bounded scan. The naive control trains the stale group. The no-drift case
measures the cost of rejecting group 0 when its labels are correct but its
policy is stale.

The primary outcome is the number of committed updates whose training-group
rewards disagree with an independent v1 recomputation. The independent unit is
the purposively selected verifier semantic, not the rollout or repeated update.
The sign comparison is descriptive and cannot establish population-level
superiority. If held-out scores saturate, report consistency and recovery only.
No synthetic score is a coding-capability result.

## Run-3 pilot result: NO-GO

The v7.3 run stopped at its frozen pilot gate on
`false_accept/sync/g-02-false_accept`, whose eight verifier-v1 rewards were all
zero. The first two groups had nonconstant reward vectors and produced two real
optimizer commits; the third group was not trained. Across the successive
policy versions, target accepts were 4/8, 1/8, and 0/8. This is a descriptive
single-run observation, not evidence that the updates caused policy regression.

The partial run generated 24 completions and made 24 verifier calls. Its three
groups were checked against the independently implemented `ground_word`
oracle; checkpoint Adam steps were 0, 1, and 2, and 16 trajectories were
consumed. Peak observed RSS was 905,412,608 bytes. The separate partial-stop
auditor passed and returned `NO_GO_FOR_ASYNC_COMPARISON`. No asynchronous arm,
or crash-recovery case started, so this run provides no VARE comparison or
coding-capability result. The runner's control flow implies five held-out
evaluation passes (baseline plus incumbent and candidate for each of two
promotions), or 60 prompt completions. Their raw outputs and scores were not
retained, so held-out quality cannot be reconstructed from this run. Preserve
the run and do not resample it or relax the stop condition.

The sole started synchronous arm recorded verifier v1 for every candidate;
the protocol's named v0-to-v1 transition was not exercised. The zero
per-group disagreement counts therefore do not demonstrate drift handling.

The original partial-stop audit remains at
[`pilot-stop-audit-v1.json`](pilot-stop-audit-v1.json). The corrected
interpretation is recorded separately at
[`pilot-stop-audit-v2.json`](pilot-stop-audit-v2.json), with its script,
protocol, runner, and raw-run inventory bound in
[`pilot-stop-audit-manifest-v2.json`](pilot-stop-audit-manifest-v2.json). The
raw run remains locally retained at
`results/verifier_consistent_async_grpo_v7/run-3/`; the root inventory has 11
files and SHA-256
`7952c4062e46b302e76df20999391e496106bca33e3d6319dcd475dbc0af12aa`.
Its last `progress.json` snapshot still says `RUNNING`: the exception path
retained that snapshot and recorded the stop reason separately in `failure.json`.
The corrected auditor rejects existing output paths and destinations inside
the retained run, then uses exclusive file creation so a raced path is not
overwritten.

## Reproduction

Use Python 3.12.12 and the package versions in `protocol.json`, the locally
retained TinyStories-1M revision and pinned RVL checkout described there, and
launch the measured runner with `-S`. No network access, new model download,
paid API, or concurrent model is needed.

```sh
python3.12 -S scripts/benchmark_verifier_drift_scenarios.py \
  --output results/verifier_consistent_async_grpo_v7/run-3 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python3.12 -S scripts/audit_verifier_drift_scenarios.py \
  results/verifier_consistent_async_grpo_v7/run-3 \
  --output results/verifier_consistent_async_grpo_v7/audit-run-3.json

python3.12 scripts/audit_verifier_drift_pilot_stop.py \
  results/verifier_consistent_async_grpo_v7/run-3 \
  --protocol benchmarks/verifier_consistent_async_grpo_v7/protocol.json \
  --runner scripts/benchmark_verifier_drift_scenarios.py \
  --output results/verifier_consistent_async_grpo_v7/audit-run-3-partial-reproduction-v2.json
```

The pilot gate was fixed in advance and failed. The first two commands above
describe the preregistered full-run path; neither is permission to resume this
consumed run. The third command re-audits the retained partial output only. Do
not resample or alter prompts, semantics, seeds, or budgets.

## Sustained crash continuation: not run

The continuation was gated on a complete v7 run and passing independent audit.
Those conditions were not met, so no crash fixture or SIGKILL case was created.
The following commands remain only as the frozen conditional plan; they are
not executed by this result.

```sh
python3.12 -S scripts/prepare_sustained_crash_fixture.py \
  --run results/verifier_consistent_async_grpo_v7/run-3 \
  --audit-report results/verifier_consistent_async_grpo_v7/audit-run-3.json \
  --output results/verifier_consistent_async_grpo_v7/crash-fixture-3

python3.12 -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v7/crash-fixture-3 \
  --model-dir ../tinystories-1m-pinned --rvl-source ../rvl-pinned \
  --output results/verifier_consistent_async_grpo_v7/sustained-crash-3 \
  --seed 82106 --timeout 300 \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v4-runner-v2.json

python3.12 -S scripts/audit_sustained_crash_recovery.py \
  --crash-output results/verifier_consistent_async_grpo_v7/sustained-crash-3 \
  --fixture results/verifier_consistent_async_grpo_v7/crash-fixture-3 \
  --source-run results/verifier_consistent_async_grpo_v7/run-3 \
  --source-audit results/verifier_consistent_async_grpo_v7/audit-run-3.json \
  --protocol benchmarks/verifier_consistent_async_grpo_v7/protocol.json \
  --output results/verifier_consistent_async_grpo_v7/audit-crash-3.json
```

The crash auditor compares the complete resumed trainer state against the
third-update reference, including model weights, Adam state, CPU RNG, and module
modes. It also replays the SQLite ledger and checks active promotion identity,
SIGKILL process evidence, optimizer attempts, and once-only trajectory
consumption. The five fault cutpoints share one prefix and one target group;
they are recovery cases, not independent quality replications.

## Scope and related work

Async RL systems such as [AReaL](https://arxiv.org/abs/2505.24298),
[StaleFlow](https://arxiv.org/abs/2601.12784), and
[DORA](https://arxiv.org/abs/2604.26256) study rollout/policy staleness,
coordination, and throughput. This study measures verifier-semantic staleness
inside a multi-policy delayed queue and the cost of rejecting its stale group;
the backend remains serialized on one CPU host, so no parallel throughput
claim is made.

[Maintaining Benchmarks Against Increasingly Capable Agents](https://arxiv.org/abs/2609.34262)
audits unearned passes and reference exposure through Git history and exploit
replay. [SpecBench](https://arxiv.org/abs/2605.21384) studies visible-test and
held-out behavior gaps, while the [Reward Hacking Benchmark](https://arxiv.org/abs/2605.02964)
includes evaluator and metadata shortcuts. These are related benchmark and
reward-integrity concerns. The v7 run stopped before its multi-scenario
asynchronous comparison and recovery extension, so it establishes neither a
new async mechanism nor a novelty claim.

Preserve failed and partial outputs. Every run records raw generations,
verifier versions, per-member labels, queue residence, optimizer replay,
checkpoint hashes, ledger snapshots, resource telemetry, and SHA-256
inventories. Never reuse a consumed output directory.
