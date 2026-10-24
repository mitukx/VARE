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

The following runner invocation is preserved to document the original
preregistration only. **Do not execute it against the consumed v7.3 tasks or
output directory.** The last command is an offline audit of retained files;
first extract the historical runner below into a separate temporary path.

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
  --runner /tmp/vare-v7.3-runner.py \
  --output results/verifier_consistent_async_grpo_v7/audit-run-3-historical-runner-review-v1.json
```

The v7.3 protocol names runner SHA-256
`d4ca48fd95a8c7e08ef88032e1dca63e7632286748ac2055029d75f34313f067`, which is
the runner committed at `5134460b635a5083943e2856e5d29c2dc1e21c3a`. Extract
that source into a separate temporary file before re-auditing; never pass the
prospective runner as the historical runner:

```sh
git fetch origin refs/pull/127/head
git show 5134460b635a5083943e2856e5d29c2dc1e21c3a:scripts/benchmark_verifier_drift_scenarios.py \
  > /tmp/vare-v7.3-runner.py
shasum -a 256 /tmp/vare-v7.3-runner.py
```

The prospective evaluation-retention runner has a different, separately
recorded source identity. The first two commands above only document the
original full-run and full-audit invocations; neither is authorized for the
consumed v7.3 run. The final command only audits retained files and writes a new
audit record outside the frozen run. The preregistered pilot gate failed; do
not resume, resample, or alter its prompts, semantics, seeds, or budgets.

The offline reproduction returned `PASS_PARTIAL_STOP_AUDIT` with the unchanged
`NO_GO_FOR_ASYNC_COMPARISON` result. Its output is
`results/verifier_consistent_async_grpo_v7/audit-run-3-historical-runner-review-v1.json`;
the v7.3 run directory and original inventory were not changed.

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

## Prospective evaluation-retention correction (v7.4; not run)

The v7.3 runner and its frozen output are unchanged in the recorded source
history. The prospective runner now writes an evaluation-start record before
decoding, then atomically creates a non-overwriting completion record with the
exact prompt, token IDs, output, per-task score/decision, evaluator and verifier
identity, checkpoint hash, source hashes, timestamps, and invocation ID. A
failure record preserves the last stage and all started/completed invocation
IDs. The offline journal auditor checks record hashes and can independently
recompute scores from raw prompts and outputs.

The new source identity is bound in
[`evaluation-retention-correction-v1.json`](evaluation-retention-correction-v1.json).
The v7.3 protocol is still the runner default and its frozen runner hash no
longer matches the prospective source, so it cannot authorize a replay or
resume. Any future run must pass a separate, fresh-task protocol through
`--protocol` and bind the new source hashes. Fault-injection regression tests
cover interruption after a completed evaluation, after a promotion is applied
but before commit, and before arm completion. These are journal-integrity
checks, not fresh model-run evidence.

The prospective arm summary now counts valid quarantined members only in the
actual stale group ID `g-00-<scenario>` and includes the `no_drift` control.
Focused tests cover both that control and a drift scenario, and fail closed if
the expected stale group is absent. This summary correction is prospective;
it does not change or reinterpret the v7.3 run.

## Prospective dense-reward calibration (v1; NO-GO for training)

The frozen calibration in
[`reward-calibration-v1.json`](reward-calibration-v1.json) used four new
calibration prompts and one fixed group of eight outputs per prompt from the
existing TinyStories-1M checkpoint. It did not use held-out evaluation prompts,
make optimizer updates, or resample. The reward changed from generic support
plausibility (v0) to target-conditioned surface compatibility (v1). This was
a bounded reward-semantics calibration, not a coding task or policy-quality
evaluation.

All four v1 groups were nonconstant (4/4), 27 of 32 output strings were unique,
and 17/32 samples had positive reward under each version. The mean absolute
per-sample v0-to-v1 reward shift was 0.1140625, below the preregistered 0.15
minimum. The fraction with shift at least 0.25 was 0.28125, above its 0.25
minimum. Since one required gate failed, the fixed decision is
`NO_GO_REWARD_CALIBRATION`: no GRPO run, async comparison, held-out evaluation,
or coding-capability claim follows. This single four-group pilot does not
estimate the probability that future groups will be nonconstant.

The independent auditor reconstructed all 32 scores and confirmed the same
gate result. Raw groups, sampler progress, summary, and the run's SHA-256
inventory are retained under
`results/verifier_consistent_async_grpo_v7/reward-calibration-v1/`. The
independent audit and parent manifest bind the audit to the run inventory and
frozen protocol. The recorded resource use was 4.18 seconds wall, 2.49 CPU
seconds, 646 MB peak RSS, and no measured swap growth.

## Deterministic verifier-drift mechanism regression (replay only)

`tests/test_authenticated_verifier_drift_regression.py` exercises authenticated
v0, v1, stale, and mixed-version trajectory receipts. It independently
recomputes the authoritative v1 score, rejects stale and mixed groups before
the trainer boundary (zero fake-trainer calls), and admits one fresh v1 group
to a fake trainer (one call). The fresh fixture remains verified and the
trajectory-store audit passes. These fixed replay fixtures test admission and
accounting behavior; they are not fresh on-policy rollouts, an optimizer
update, or evidence of model learning. The regression is independent of the
reward-calibration gate and does not authorize the matched async experiment.

### Combined outcome

The prospective evaluation journal and its three crash-cutpoint tests pass;
the dense-reward calibration independently reconstructs but fails its
preregistered mean-shift gate; and the replay-only drift mechanism regression
passes. Therefore the real-model async comparison remains **NO-GO**. The exact
remaining blocker is that this frozen reward transition changed scores less
than required (0.1141 mean absolute shift versus 0.15); no model training or
async run was started to work around that result.
