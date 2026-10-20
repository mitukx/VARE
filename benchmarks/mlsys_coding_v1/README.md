# MLSys coding pack v1

Authored single-file Python utilities with executable defects. The pack is a
frozen evaluation task set for a coding agent. It is not an upstream
historical commit, and a passing oracle patch is harness calibration, not
model capability.

## Layout

Each task directory contains:

- `task.json`, an `EngineeringTaskSpec` whose `source.local_path` is `subject/`
- `subject/`, the only tree copied into the candidate workspace
- `evaluator/`, hidden checks run with `cwd_base=evaluator`
- `evaluator/oracle/`, the known fix used by the calibration test

`metadata.grader_visibility` is `hidden`. On the isolated-grader branch, the
public repository-grader opt-in refuses any spec that is not explicitly
`public`, so these tasks do not run through that candidate-visible path.

`metadata.task_type` is `correctness` or `performance` and is read from
`task.json`. Graders call `harness_adapter.py`. That module proposes
`vare.separated_candidate_harness.grade_function` and, until the symbol
exists, uses the temporary shim `candidate_call.py`.

Performance tasks (`range_sums`, `dataset_standardize`) print one JSON object
on grader stdout. The object records seven parent-timed samples for a frozen
slow reference and for the candidate, plus the sample standard deviations.
The retained stdout is part of the existing command-result schema. The
protocol reward recomputes

```text
conservative_speedup = baseline_ci95_low / candidate_ci95_high
```

with a Student-t 95% interval at 6 degrees of freedom. Success requires that
ratio to be at least 2. The shaped reward is `min(1, speedup / 4)` when
hidden correctness passed.

## Split

`split.json` freezes five train tasks and four held-out tasks. The manifest
hash covers the split. Do not move a task after a model result exists.

## What this pack does not establish

It does not show that a model can solve the tasks, that hidden files stay
hidden from a candidate process, or that a policy update improves held-out
success. The in-process evaluator used for oracle calibration is a trusted
harness for this repository's own fixtures.
