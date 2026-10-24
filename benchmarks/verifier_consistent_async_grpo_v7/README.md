# Multi-scenario verifier drift and sustained recovery

This v7.2 preregistration tests whether verifier changes can contaminate real GRPO
updates in a delayed asynchronous queue, and whether VARE rejects the stale
group while continuing to train on a fresh group already waiting behind it.
TinyStories-1M sampling and RVL optimizer updates are real. Tasks and rewards
are synthetic; this does not measure coding capability or natural verifier
drift prevalence.

## Preregistration amendment and retained preflight failure

The v7.1 protocol is preserved at `protocol-v7.1-failed-preflight.json`. Its
first invocation stopped before model instantiation, sampling, or optimizer
work because the runner read the seed from the wrong JSON path. The failure
record and inventory remain at
`results/verifier_consistent_async_grpo_v7/run-1/`; v7.2 changes only the seed
field lookup and binds both protocol hashes in
`protocol-amendment-v7.2.json`. The experiment starts at a new output path.

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

## Reproduction

Use Python 3.12.12 and the package versions in `protocol.json`, the locally
retained TinyStories-1M revision and pinned RVL checkout described there, and
launch the measured runner with `-S`. No network access, new model download,
paid API, or concurrent model is needed.

```sh
python3.12 -S scripts/benchmark_verifier_drift_scenarios.py \
  --output results/verifier_consistent_async_grpo_v7/run-2 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python3.12 -S scripts/audit_verifier_drift_scenarios.py \
  results/verifier_consistent_async_grpo_v7/run-2 \
  --output results/verifier_consistent_async_grpo_v7/audit-run-2.json
```

The pilot gate is fixed in advance. If it fails, retain the partial output and
stop. Do not resample or alter prompts, semantics, seeds, or budgets.

## Sustained crash continuation

Only after the full v7 run and its independent audit pass, prepare a fixture
from the VARE false-accept arm after two committed updates. The retained third
update group and its no-crash checkpoint are the independent continuation
reference. The five actual SIGKILL cutpoints use the existing corrected v6
crash harness; no inference seed or model sampling is repeated. The harness
seed is fixed only for its replay/run identity.

```sh
python3.12 -S scripts/prepare_sustained_crash_fixture.py \
  --run results/verifier_consistent_async_grpo_v7/run-2 \
  --audit-report results/verifier_consistent_async_grpo_v7/audit-run-2.json \
  --output results/verifier_consistent_async_grpo_v7/crash-fixture-2

python3.12 -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v7/crash-fixture-2 \
  --model-dir ../tinystories-1m-pinned --rvl-source ../rvl-pinned \
  --output results/verifier_consistent_async_grpo_v7/sustained-crash-2 \
  --seed 82106 --timeout 300 \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v4-runner-v2.json

python3.12 -S scripts/audit_sustained_crash_recovery.py \
  --crash-output results/verifier_consistent_async_grpo_v7/sustained-crash-2 \
  --fixture results/verifier_consistent_async_grpo_v7/crash-fixture-2 \
  --source-run results/verifier_consistent_async_grpo_v7/run-2 \
  --source-audit results/verifier_consistent_async_grpo_v7/audit-run-2.json \
  --protocol benchmarks/verifier_consistent_async_grpo_v7/protocol.json \
  --output results/verifier_consistent_async_grpo_v7/audit-crash-2.json
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
reward-integrity concerns; this protocol does not claim novelty until the
frozen scenarios and the five sustained crash cases execute and pass an
independent audit.

Preserve failed and partial outputs. Every run records raw generations,
verifier versions, per-member labels, queue residence, optimizer replay,
checkpoint hashes, ledger snapshots, resource telemetry, and SHA-256
inventories. Never reuse a consumed output directory.
