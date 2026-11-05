# ResearchGym operator notes

ResearchGym runs small CPU machine-learning repairs. The controller records
every committed experiment. Resume reads that journal and does not execute a
committed experiment id again.

The process boundary is a same-user process group with a wall-clock limit.
It is not a container sandbox, a network namespace, or a second user. Do not
treat it as one. Candidate code is imported only in the child. Expected
values stay in the parent.

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
