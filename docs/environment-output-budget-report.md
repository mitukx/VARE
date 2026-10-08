# Environment command output budget

## Failure found

`ExecutableEvaluator` runs repository test and metric commands through `run_command`. Before the fix, `subprocess.run(..., stdout=PIPE, stderr=PIPE)` accumulated each complete stream in the evaluator process, then sliced the byte strings to `max_output_bytes`. A command could therefore emit more than the configured capture limit and still return code 0, causing a truncated result to count as a pass.

## Frozen regression and result

Protocol v4 tests stdout overflow, stderr overflow, a normal command, and on POSIX a child that inherits the command process group. The descriptor pins VARE baseline `410e5321ab762c89ada970de1b41d2aacf99d88a`; the protocol, evaluator, and task brief were frozen before the v4 baseline grade.

- Baseline: both 1 MiB output-flood commands returned code 0 and were treated as passing commands; a child process remained alive long enough to write its marker.
- Fixed candidate: stdout and stderr overflow each return `output_limited=true`, are not treated as passes or timeouts, retain at most 128 bytes per stream, and stop the POSIX process-group child. A normal command below the budget still passes.
- Artifacts: [v4 summary](../results/environment-output-budget-v4/summary.json), [baseline grade](../results/environment-output-budget-v4/baseline.grade.json), [fixed grade](../results/environment-output-budget-v4/fixed.grade.json), [manifest](../results/environment-output-budget-v4/manifest.json), and [protocol snapshot](../results/environment-output-budget-v4/protocol_snapshot/).

The task contract and evaluator are at [`benchmarks/regressions/environment_output_budget_v4/`](../benchmarks/regressions/environment_output_budget_v4/). The code retains output incrementally, caps each stream separately, and kills/reaps the POSIX process group when either stream exceeds the configured limit. `CommandResult.output_limited` distinguishes this from timeout and ordinary command rejection.

## Protocol history

- v1 is retained but invalidated for its process-group-cleanup claim: it checked the descendant marker after the temporary directory holding it had been deleted. Its baseline still exposed accepted over-budget output, but v1 is not the headline calibration.
- v2 correctly calibrated stdout overflow and normal output. It is retained as a narrower result and does not independently test stderr-only overflow.
- v3 is retained as an invalidated harness attempt: its empty marker argument violated the `CommandSpec` contract before a complete baseline grade was produced.
- v4 corrects the fixture and tests stdout and stderr independently. Earlier snapshots and raw outputs are retained under `results/`.

## Limits and reproduction

This is a deterministic CPU regression on one macOS arm64 host. The fixture verifies retained byte counts and process-group cleanup; it does not measure peak RSS. The code's stream buffers are capped at the configured per-stream limit plus fixed-size read chunks, but there is no OS-enforced evaluator memory, CPU, disk, or network quota. A process that deliberately escapes its process group remains outside the runner's containment guarantee. This is not a hostile-code sandbox.

No model weights, GPU, paid API, or external compute are required.
