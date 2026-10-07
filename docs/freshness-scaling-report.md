# Scaling freshness checks during durable completion

## Question

Does the durable coordinator repeat input fingerprint work across all enrolled jobs each time one job completes, and can it restrict commit-time checking to the completing job while preserving use-time and export-time freshness checks?

This is a same-host CPU systems measurement. It does not measure grader throughput, candidate correctness, model behavior, or distributed scale.

## Frozen protocol and source identity

Protocol [`cpu-refresh-scale-v2`](../protocols/cpu_refresh_scale_v2.json) freezes three serial replications at 8, 16 and 32 enrolled job IDs. Each workload reuses one unchanged local Git candidate and locked fixture task. The measured loop claims and commits operational-error terminal records; no grader or model is run. Enrollment and candidate/task construction are outside the timed region.

The primary metric is the number of `durable.inputs()` fingerprint computations during terminal completions. The baseline implementation is bound to commit `68eb136ea0317a05b10948fec9375c8c856e1388`, through the exact committed `src/vare/durable.py` hash. The intervention changes that source file in `43b602b6aac460e7aff837192b13e143c9c6d0d6`. Both arms use the same committed measurement script. The auditor checks those hashes against Git objects and replays every exported bundle.

## Results

| Enrolled jobs | Baseline completion checks | Targeted completion checks | Median completion time before | Median after |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 8 | 1.536 s | 0.265 s |
| 16 | 256 | 16 | 6.939 s | 0.542 s |
| 32 | 1,024 | 32 | 28.479 s | 1.082 s |

At every size, the targeted arm reduced commit-time input checks by exactly the enrolled job count. Every export in both arms still recomputed all enrolled inputs (8, 16 and 32 checks respectively), and all 18 exported bundles passed offline replay. Each measured terminal record was retained once with its expected operational-error classification. No input mutation was introduced in this scaling workload.

Median elapsed time is descriptive for this machine and protocol. At 32 jobs it was 28.479 seconds in the baseline and 1.082 seconds after the change. This is not a portable speedup claim: the timed work is coordinator freshness checking, the workload repeats an identical candidate workspace, and no heterogeneous candidate grading is measured.

The first v1 pilot is retained under [`results/cpu-refresh-scale-v1`](../results/cpu-refresh-scale-v1/). Its own audit identified a mismatch between the protocol's named revision and the measured revision, and its intervention ran with uncommitted source. The pilot is explicitly marked invalid for comparison and none of its timings support the result above. V2 corrected the source and harness identity checks before its baseline was measured.

## Implementation and correctness boundary

`Store.complete()` now refreshes only the job whose claim is being committed. The worker still recomputes the claimed job's inputs immediately before grading. Worker startup and `Store.export()` still refresh all enrolled jobs. This avoids rescanning unrelated workspaces during each completion without caching freshness across a use boundary.

The regression test instruments `inputs()` and checks one targeted check per completion, then verifies explicit full refresh and export still inspect all jobs. Existing tests cover changes after grading/before commit, stale pending jobs, invalidation of completed results and preserved historical attempts. The full suite passes 66 tests.

These checks do not make source hashing atomic with grading or publication. A trusted source writer can still change files immediately after a check; the existing contract remains a same-host snapshot check, not hostile-writer isolation. Public result bundles are internally consistent and hashed, not cryptographically signed against coordinated replacement of the trust anchor.

## Reproduction and audit

From a clean checkout at the recorded commits, using Python and Git:

```bash
python3 scripts/measure_refresh_scale_v2.py --phase baseline --output results/cpu-refresh-reproduction/baseline
python3 scripts/measure_refresh_scale_v2.py --phase targeted --output results/cpu-refresh-reproduction/targeted
python3 scripts/audit_refresh_scale_v2.py results/cpu-refresh-reproduction
```

The baseline command must run with the frozen baseline implementation in the checkout. The targeted command must run at a commit containing the intervention. The auditor verifies the recorded commits, source hashes, measurement-script hash, expected operation counts, every exported job outcome and every bundle manifest. Timing will vary by machine. The experiment uses local CPU, Git and the standard library; no GPU, paid API or external compute is used.
