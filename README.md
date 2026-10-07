# VARE

VARE is a CPU evaluation control plane for reproducible experiments on agent improvement. It runs independently graded historical software tasks with bounded concurrency, locked protocols, candidate provenance checks, timeout/output handling, and auditable retained records.

Contributors should read [`AGENTS.md`](AGENTS.md) before changing experiments, code, or claims.

## Measured execution evidence

- **24/24 recovery fault cases** passed under a frozen protocol: process death, rollback, lost acknowledgement, late completion, input invalidation and retry exhaustion. The retained exports contain 117 synthetic jobs/126 attempts, plus four historical jobs recovered with all calibrated outcomes preserved. See the [recovery report](docs/recovery-report.md).

- **TRL grader mutation checks:** protocol v2 rejects an unmodeled later normalizer overwrite that passed v1. A separate candidate that doubled the actual loss denominator passed v2, then protocol v3 rejected all 12 branch conditions while retaining the correct extracted normalizers. The fixed-source baseline passes all 12 conditions under v3. See the [overwrite evidence](results/trl-grpo-accumulation-window-normalizer-v1/grader-mutation-v1/summary.json), [denominator evidence](results/trl-grpo-accumulation-window-normalizer-v1/loss-denominator-mutation-v1/summary.json), and [v3 calibration](results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v3/summary.json).

- **RVL generation-setting mutation check:** the v2 grader accepted a candidate that inherited `typical_p=0.72`; v3 rejects it in all six rollout conditions. The pinned fixed source passes all 12 rollout/trainer conditions under v3. See the [mutation evidence](results/rvl-hf-behavior-policy-parity-v1/generation-config-mutation-v1/summary.json) and [v3 calibration](results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v3/summary.json).

- **Local CPU agent pilot:** the corrected v2 tool loop ran one small local model on one pinned task for three formal seeds. It made no source edits and the locked grader rejected all three unchanged checkouts. The original v1 cohort is invalidated because of a tool-history serialization bug. See the [pilot report](docs/local-agent-pilot.md).

- **Promotion-gate input validation:** a frozen CPU regression task found that the pre-fix promotion gate accepted all seven tested reports containing `NaN` or infinity. The fixed gate rejects those inputs and preserves acceptance for a finite positive control. See the [report](docs/promotion-gate-report.md) and [raw result](results/promotion-gate-metrics-v1/summary.json).

- **16/16 synthetic reliability cases** matched their declared outcomes, including incorrect provenance, source mutation, malformed output, timeouts, output flooding and subprocess cleanup.
- **80 historical-source evaluations** preserved the expected pre-fix rejection/fixed acceptance decisions across one-worker and four-worker campaigns.
- **3.4857× median paired speedup** with four workers versus this same runner with one worker, over five pairs on a local macOS arm64 CPU. This is a small local-grader measurement, not a distributed or model-serving result.

- **Freshness validation scales with the completing job:** under a frozen serial 8/16/32-job CPU protocol, commit-time fingerprint checks fell from 64/256/1,024 to 8/16/32. At 32 jobs, median completion time fell from 28.479s to 1.082s on this machine. Full input refresh remains at export. This repeated-fixture coordinator measurement is machine-local, not heterogeneous grader throughput. See the [frozen report](docs/freshness-scaling-report.md).

Start with the [ten-minute walkthrough](docs/walkthrough.md), [technical report](docs/scheduler-report.md), and [execution contract](docs/execution.md). Inspect retained evidence offline:

```bash
python3 scripts/audit_scheduler.py results/cpu-scheduler-v1
python3 -m unittest discover -s tests -p 'test_runner.py' -v
python3 -m unittest discover -s tests -p 'test_durable.py' -v
python3 scripts/audit_recovery.py results/cpu-recovery-v1
```

For a new campaign, use `python3 -m vare run --plan PLAN.json --output NEW_DIRECTORY --workers 4`, then `python3 -m vare audit NEW_DIRECTORY`. The execution contract provides the plan format. The runner needs Python >=3.9, Git and POSIX process groups; it has no third-party Python dependency.

## Persistent evaluation and requirements

`durable-init`, `durable-work`, `durable-export` and `durable-audit` add same-host transactional claims, fenced leases, bounded crash retries, immutable attempt records and freshness invalidation. See the [requirements matrix](docs/requirements.md) and [recovery contract](docs/recovery.md). The installed package and experimental control-loop commands require Python >=3.11; standalone checkout evaluation uses Python >=3.9. All implementation is under `src/vare`, with a checkout bootstrap in `vare/`.

## Current scope

Two real upstream fixes calibrate the task and grading path:

| Task | What the grader exercises | Pre-fix result | Fixed result |
| --- | --- | --- | --- |
| [HF behavior-policy parity](benchmarks/historical/rvl_behavior_policy_parity/TASK.md) | Protocol v3 checks neutral `top_k`, `top_p`, repetition penalty, and pretrained `typical_p` behavior on deterministic CPU fixtures, alongside rollout probabilities and the learner's actual `_sample_objective` path. | Maximum absolute rollout log-probability error `0.7380`; learner error `0.3711`; rejected. | Both errors `0.0`; accepted. |
| [TRL accumulation-window normalizer](benchmarks/historical/trl_grpo_accumulation_scale/TASK.md) | Protocol v3 executes six source-derived arithmetic cases in each of two trainer paths, rejects later normalizer overwrites, and checks that the selected loss denominator directly uses the computed normalizer. | Maximum absolute normalizer error `12`; rejected. | Maximum error `0`; accepted. |

The second task tracks the upstream [TRL issue](https://github.com/huggingface/trl/issues/5619) and [fix](https://github.com/huggingface/trl/pull/6024). The first uses a pinned fix in [Recursive-Verification-Lag](https://github.com/mitukx/Recursive-Verification-Lag).

Raw grader outputs, summaries, protocol snapshots, and SHA-256 manifests are retained under [`results/`](results/). See the [evidence notes](docs/evidence.md) for exact revisions, metrics, and limits. Earlier HF protocol results remain as history; v3 is the current protocol.

The separate [integrity calibration](results/protocol-integrity/cpu-calibration-v4/summary.json) changes the task descriptor, task brief, and grader one at a time for both current task protocols. All six changes are rejected against the unchanged protocol lock. The checked-in Git history is the trust anchor for that lock; this does not detect coordinated edits to the lock itself.

The [local agent pilot](docs/local-agent-pilot.md) is one small-model, one-task negative result: the corrected formal cohort read the task source, but produced no accepted source edits. It does not characterize coding agents generally or establish successful task solving or generalization. The original v1 runs are invalidated and excluded from inference.

The E1 task/evaluator calibrations do not measure a model training or capability improvement. The separate local agent pilot had no successful patch and is reported as a negative result. The TRL grader executes extracted production normalization statements with scalar doubles and checks a direct AST denominator contract; it does not execute the full loss expression or run a trainer/gradient update. The HF grader executes candidate source with local fixtures; it is not an operating-system sandbox for hostile code. Only run it on candidate code you trust.

The restored experimental control plane in `src/vare` includes grouped replay, freshness/curriculum controls, paired promotion gates, workspace-agent campaigns and RVL hooks. These are implementation/regression contracts. No real-model learning, successful local-agent task solution, or GPU result is claimed. The imported L0 raw archive is corrupt; its old accuracy summary is excluded from verified claims (see the evidence notes).

## Reproduce the calibrations

Requirements: Python 3.9 or newer, Git, and network access to fetch the pinned public source revisions. The scripts use only the Python standard library. No model weights, GPU, paid API, or external compute are used for local calibration.

Run each calibration into a new directory outside the repository. The checked-in result directories already exist, so choose a fresh path:

```bash
python3 scripts/calibrate_task.py \
  --output /tmp/vare-rvl-calibration

python3 scripts/calibrate_task.py \
  --task-root benchmarks/historical/trl_grpo_accumulation_scale \
  --output /tmp/vare-trl-calibration

python3 scripts/calibrate_integrity.py \
  --output /tmp/vare-integrity-calibration
```

The calibrator verifies the task, brief, and grader hashes against the protocol lock; fetches the immutable pre-fix and fixed source revisions; grades both; and writes raw JSON, a summary, a source/protocol snapshot, and a manifest. The preregistered outcome requires the pre-fix revision to fail and the known fixed revision to pass.

To prepare and grade a candidate workspace manually:

```bash
python3 scripts/prepare_task.py \
  --workspace /tmp/vare-rvl-candidate
# Make a candidate change in that checkout.
python3 scripts/grade_task.py \
  --workspace /tmp/vare-rvl-candidate
```

For the TRL task, provide `--task-root benchmarks/historical/trl_grpo_accumulation_scale` to both commands. Candidate workspaces must be outside the VARE checkout. The grader and locked task files stay outside the candidate workspace.

## Roadmap

The [roadmap](docs/roadmap.md) tracks the evidence needed before expanding the claims. The next major gap is repeated, comparable agent trajectories on a fixed task set and budget. No such trajectories are currently reported.

## Develop the full package

Use Python >=3.11 in a local virtual environment. The runtime has no third-party dependencies; pytest is only for development tests. Use an editable install from this checkout because task/protocol assets are repository files.

```bash
python -m pip install -e . pytest==8.4.2
python -m pytest
vare --help
```

## License

Apache-2.0. See [LICENSE](LICENSE).
