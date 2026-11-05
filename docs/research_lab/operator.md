# ResearchGym operator notes

ResearchGym runs small CPU machine-learning repairs. The controller records
every committed experiment. Resume reads that journal and does not execute a
committed experiment id again.

## Isolation, fail closed

Hidden grading launches the worker through `vare.research_lab.isolation`,
which reuses the learning-pipeline `BwrapBackend` probe and argv. That
adapter is not a trusted security sandbox. A passed probe does not certify
the kernel, the host runtime, or a candidate that shares the worker
interpreter. If bubblewrap is missing or the probe is not fully denied, the
grade is `isolation_unverified` and is not a pass. There is no fallback to a
same-user process group.

`run_bounded` is a same-user subprocess in its own process group, with a
wall-clock kill and a CPU rlimit. It is not the hidden-grade boundary and it
is not a trusted OS sandbox. There is no container, no second user, no
chroot, no network namespace, and no hard address-space limit.
`IS_TRUSTED_OS_SANDBOX`, `HARD_ADDRESS_SPACE_LIMIT_APPLIED`, and
`NETWORK_NAMESPACE_APPLIED` are false. `memory_limit_applied` stays false:
`memory_mb` is a recorded budget, and a post-hoc `VmHWM` sample does not stop
an allocation.

On that process-group helper, candidate code can open a socket. The
controller rejects a `shell` or `network` action before launch. That
rejection does not confine code already executing inside a worker that was
started without the bubblewrap gate. A passing grade does not mean a trusted
sandbox prevented network access or memory use. Expected values stay in the
parent. Commands are not taken from the candidate. A timeout is `timeout`,
not a correctness failure.

Do not read a Xeon CPU record as a Metal or M1 measurement.
`counts_as_autonomous_model_success` stays false unless a real model-backed
run is recorded, and this package does not start one.

## Commands

From the repository root, with `PYTHONPATH=src` if the package is not installed:

```bash
python -m vare research list
python -m vare research calibrate --work-root artifacts/research_lab/calibration
python -m vare research run --task TASK_ID --run-id RUN_ID --proposer diagnostic
python -m vare research status --task TASK_ID --run-id RUN_ID
python -m vare research resume --task TASK_ID --run-id RUN_ID
python -m vare research evaluate --task TASK_ID --run-id RUN_ID
python -m vare research report --task TASK_ID --run-id RUN_ID
python -m vare research export --task TASK_ID --run-id RUN_ID
```

`--run-root` defaults to `artifacts/research_lab` (gitignored). `--task` is
required so a run id cannot be paired with the wrong pack.

Proposers:

| Name | What it is |
| --- | --- |
| `diagnostic` | Deterministic policy. It tries one failing repair, then a second repair, using public measurements. Not a model. |
| `first-hypothesis` | Negative control. Submits the first repair even when the public measurement rejects it. |
| `oracle` | Copies `controls/known_good`. Requires `--oracle-control`. Grader calibration only. |
| `external` | No in-process proposer. The operator applies one action at a time with `act`. |

```bash
python -m vare research run --task TASK_ID --run-id RUN_ID --proposer external --max-actions 0
python -m vare research act --task TASK_ID --run-id RUN_ID --action-json action.json
```

`evaluate` re-grades the current workspace and appends `evaluations.jsonl`.
It does not rewrite the journal. If the workspace text differs from the
committed snapshots, the result is `stale_source` and the grade does not run.
`resume` restores the committed snapshots before the next action.

A `CANCEL` file in the run directory stops the controller before the next
experiment and commits a `cancel` record.

## What a passing run is

`success_class=diagnostic_policy` means the deterministic policy produced a
candidate that passed the hidden grader. `success_class=external_agent_assisted`
means an operator or external agent did. `oracle_control` means the known-good
files passed. None of these set a training reward or authorize GRPO.

`counts_as_autonomous_model_success` is always false in this package. No model
runner is invoked.

## Tests

```bash
python -m pytest tests/research_lab -q
```
