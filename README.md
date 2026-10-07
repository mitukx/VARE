# VARE

VARE is a CPU evaluation control plane for reproducible experiments on agent improvement. It runs independently graded historical software tasks with bounded concurrency, locked protocols, candidate provenance checks, timeout/output handling, and auditable retained records.

Contributors should read [`AGENTS.md`](AGENTS.md) before changing experiments, code, or claims.

## Measured execution evidence

- **16/16 synthetic reliability cases** matched their declared outcomes, including incorrect provenance, source mutation, malformed output, timeouts, output flooding and subprocess cleanup.
- **80 historical-source evaluations** preserved the expected pre-fix rejection/fixed acceptance decisions across one-worker and four-worker campaigns.
- **3.4857× median paired speedup** with four workers versus this same runner with one worker, over five pairs on a local macOS arm64 CPU. This is a small local-grader measurement, not a distributed or model-serving result.

Start with the [ten-minute walkthrough](docs/walkthrough.md), [technical report](docs/scheduler-report.md), and [execution contract](docs/execution.md). Inspect retained evidence offline:

```bash
python3 scripts/audit_scheduler.py results/cpu-scheduler-v1
python3 -m unittest discover -s tests -v
```

For a new campaign, use `python3 -m vare run --plan PLAN.json --output NEW_DIRECTORY --workers 4`, then `python3 -m vare audit NEW_DIRECTORY`. The execution contract provides the plan format. The runner needs Python >=3.9, Git and POSIX process groups; it has no third-party Python dependency.

## Current scope

Two real upstream fixes calibrate the task and grading path:

| Task | What the grader exercises | Pre-fix result | Fixed result |
| --- | --- | --- | --- |
| [HF behavior-policy parity](benchmarks/historical/rvl_behavior_policy_parity/TASK.md) | Deterministic CPU model/tokenizer fixtures exercise rollout probabilities and the learner's actual `_sample_objective` path across six prompt/temperature conditions. | Maximum absolute rollout log-probability error `0.7380`; learner error `0.3711`; rejected. | Both errors `0.0`; accepted. |
| [TRL accumulation-window normalizer](benchmarks/historical/trl_grpo_accumulation_scale/TASK.md) | The source-derived normalizer branches are executed against scalar fixtures: six arithmetic cases in each of two trainer paths. | Maximum absolute normalizer error `12`; rejected. | Maximum error `0`; accepted. |

The second task tracks the upstream [TRL issue](https://github.com/huggingface/trl/issues/5619) and [fix](https://github.com/huggingface/trl/pull/6024). The first uses a pinned fix in [Recursive-Verification-Lag](https://github.com/mitukx/Recursive-Verification-Lag).

Raw grader outputs, summaries, protocol snapshots, and SHA-256 manifests are retained under [`results/`](results/). See the [evidence notes](docs/evidence.md) for exact revisions, metrics, and limits. The earlier v1 protocol result for the HF task is retained as superseded history; v2 is the current protocol.

The separate [integrity calibration](results/protocol-integrity/cpu-calibration-v1/summary.json) changes the task descriptor, task brief, and grader one at a time for both tasks. All six changes are rejected against the unchanged protocol lock. The checked-in Git history is the trust anchor for that lock; this does not detect coordinated edits to the lock itself.

These are task/evaluator calibration results. They do not measure an agent solving tasks, model training, generalization, or capability improvement. The TRL grader executes extracted production normalization statements with scalar doubles; it does not run a full trainer or gradient update. The HF grader executes candidate source with local fixtures; it is not an operating-system sandbox for hostile code. Only run it on candidate code you trust.

The repository implements evaluation orchestration, not a model-training loop, coding-agent orchestration, curriculum intervention, or promotion system. Those are later research steps, not measured features.

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

## License

Apache-2.0. See [LICENSE](LICENSE).
