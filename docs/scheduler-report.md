# CPU evaluation execution: measured reliability and scheduling

## Question and contribution

VARE implements a bounded local evaluation runner that binds accepted scores to locked protocol assets and observed candidate inputs, separates candidate rejection from operational failure, and retains auditable records. The measured question is whether parallel execution reduces wall time while preserving those decisions. All implementation and experiments use the Python standard library, Git and local CPU; there are no model weights, accelerator runs or paid API calls.

This is an original VARE execution implementation, with historical upstream fixes used as calibration inputs. VARE did not author those upstream fixes in this work. The [execution contract](execution.md) documents the implementation and trust boundary.

## Frozen protocol

Commit `2575bd2` includes the runner, regression tests, experiment script and [protocol](../protocols/cpu_scheduler_v1.json), before the first measurement. Each campaign has eight jobs: two tasks × pre-fix/fixed revisions × two copies. Five paired repetitions alternate one-worker and four-worker order. The primary metric is the median of paired serial/parallel wall-time ratios. The acceptance rule requires all 80 outcomes to match the calibrated candidates and all bundles to pass audit; the protocol retains slowdown or null results without retuning. Source checkout and initial protocol snapshots are outside the timed region; validation, subprocess execution, capture and atomic job/journal writes are inside it.

The historical jobs exercise actual HF rollout/learner code with deterministic CPU doubles and extracted TRL loss-normalization statements with scalar fixtures. They do not execute model training. The two tasks and their exact revisions are described in [evidence.md](evidence.md).

## Results

Local environment: macOS 27.0.1 arm64, Python 3.9.6. No third-party Python packages. The baseline is the same instrumented runner with one worker.

| Pair | Order | 1 worker wall (s) | 4 workers wall (s) | Ratio |
| --- | --- | ---: | ---: | ---: |
| 0 | 1 then 4 | 2.8836 | 0.5243 | 5.5004 |
| 1 | 4 then 1 | 1.8546 | 0.5339 | 3.4738 |
| 2 | 1 then 4 | 1.8475 | 0.5300 | 3.4857 |
| 3 | 4 then 1 | 1.8395 | 0.5498 | 3.3459 |
| 4 | 1 then 4 | 1.8561 | 0.5315 | 3.4923 |

The primary median paired speedup is **3.4857×**. Across campaigns, median throughput is approximately **4.31 to 15.05 jobs/s**. All 80 jobs returned the expected valid decisions: 40 pre-fix rejections and 40 fixed acceptances. Queue and execution p50/p95 are retained for every campaign in the [raw summary](../results/cpu-scheduler-v1/summary.json).

Pair 0 has a much slower serial run; it is retained. No warmup was specified, no causal diagnosis of that outlier was measured, and no confidence interval is claimed from five pairs. Alternating order and the preregistered median limit reliance on that outlier, but do not establish performance portability.

## Reliability evidence

All **16/16** preregistered synthetic cases matched their expected classifications. These include a valid pass and candidate rejection, malformed/duplicate/nonfinite JSON, grader crash, incorrect task ID/diff hash, passing JSON with an inconsistent exit code, source mutation during grading, missing source, protected-brief tampering, timeout and output flooding. Two subprocess cases exercise normal child cleanup and an inherited output pipe. Both child processes were observed absent after cleanup. The inherited-pipe case timed out; the ordinary child case passed after cleanup.

The flood case retained at most 1,024 combined output bytes. Other jobs default to a 1 MiB cap and 30-second deadline. [Reliability summary](../results/cpu-scheduler-v1/reliability-summary.json), [raw records](../results/cpu-scheduler-v1/reliability/jobs) and [journal](../results/cpu-scheduler-v1/reliability/ledger.json) preserve the observations. Synthetic classification accuracy is a regression-harness measurement, not robustness against arbitrary attacks.

The regression suite also exercises cancellation with partial evidence, output-directory preservation, input-path validation, bounded concurrency, source symlinks and audit tampering. A failing inherited-pipe regression during development led to explicit stream-EOF completion rather than relying on `Process.wait()` semantics. After measurement, CLI auditing was corrected to return success for a consistent bundle even when it contains expected operational failures; input path checks were also tightened. The measurement snapshots retain the runner actually used. These post-measurement audit/CLI changes were not used to rerun or improve the reported timings.

## Reproduce and inspect

Offline evidence verification (no source fetch):

```bash
python3 scripts/audit_scheduler.py results/cpu-scheduler-v1
python3 -m vare audit results/cpu-scheduler-v1/reliability
python3 -m unittest discover -s tests -v
```

The scheduler auditor reconstructs every pair's throughput, quantiles and speedup from raw campaign records; validates source revisions, task IDs, budgets and outcomes; and checks the headline summary against those calculations. File manifests and journal hashes detect edits relative to the reviewed bundle. They do not digitally authenticate an adversary who replaces the entire bundle and trust anchor.

Full remeasurement (requires network only for pinned public source history):

```bash
python3 scripts/measure_scheduler.py --output /tmp/vare-scheduler-reproduction
python3 scripts/audit_scheduler.py /tmp/vare-scheduler-reproduction
```

Use a fresh output directory. Timing will vary with the machine, filesystem/cache state and background work; outcome and provenance checks should still match.

## What this establishes and the next question

This supplies E0 failure-injection evidence, uses two E1 historical tasks, and provides a narrow E5 local scheduling measurement. It establishes usable CPU evaluation plumbing and a measured improvement against its serial configuration. It does not establish distributed scaling, GPU utilization, inference performance, hostile-code isolation, successful agent task-solving, learning or generalization. A corrected three-run local agent pilot is retained separately; all attempts made no edit, and the original cohort is invalidated due to a harness serialization bug. Broader successful agent trajectories remain open work.
