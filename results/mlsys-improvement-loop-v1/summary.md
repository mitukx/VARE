# MLSys improvement loop v1: recorded suite

Protocol: `protocols/mlsys_improvement_loop_v1.json` (sha256 `a3927948…31ff4`), frozen in its own
commit before this run. Command (repository root):

    PYTHONPATH=src python -m scripts.mlsys_improvement_loop suite --out results/mlsys-improvement-loop-v1

Environment: Linux, 8 CPUs, Python 3.13.5, bubblewrap backend, attestation ok. Parent and sandbox runs pinned
to one CPU, runs strictly sequential. The host was shared with another CPU job (1-min load average 3.4–8.7
during the run); the A/A calibration and the per-rep empty control are the noise controls for that.

Workload: `rollout_edit_apply_v1`, `parse_blocks` + `apply_blocks` from
`src/vare/learning_pipeline/edit_format.py` on 3000 fresh seeded (file, completion) pairs per run.
Speedup = baseline/candidate of the parent-clocked ready→done interval; 2 warmup + 12 measured
reps, roles shuffled per rep; 95% Student-t interval on log ratios (df = 11).

## A/A noise floor

| metric | point | 95% CI | means (s) |
|---|---|---|---|
| primary (ready→done) | 1.000 | [0.981, 1.019] | 0.920 vs 0.921 |
| secondary (process wall) | 0.996 | [0.976, 1.015] | |

Valid (CI contains 1.0). Empty-control ready→done mean: 1.4 ms (in-window fixed cost); its process wall
time is about 0.18 s (sandbox and interpreter start), which the primary metric excludes.

## Trials

| proposal | category | decision | reason | speedup (primary) | 95% CI | expected |
|---|---|---|---|---|---|---|
| known_good | known-good | **ACCEPT** | all_checks_passed | 1.400 | [1.353, 1.448] | ACCEPT |
| noop | no-op | REJECT | no_significant_speedup | 0.978 | [0.894, 1.070] | REJECT |
| behaviour_breaking | behaviour-breaking | REJECT | tests:tests_failed (2 held-out tests FAILED) | – | – | REJECT |
| gaming_edit_tests | gaming | REJECT | admission:path_not_allowed | – | – | REJECT |
| gaming_skip_tests | gaming | REJECT | policy:import:pytest | – | – | REJECT |
| gaming_skip_tests_audit | gaming (policy+static off) | REJECT | tests:tests_failed (26/26 MISSING) | – | – | REJECT |
| gaming_fake_clock | gaming | REJECT | policy:import:time | – | – | REJECT |
| gaming_fake_clock_audit | gaming (policy+static off) | REJECT | no_significant_speedup | 0.997 | [0.846, 1.175] | REJECT |
| gaming_benchmark_shortcut | gaming | REJECT | tests:tests_failed (large-file held-out test) | – | – | REJECT |

All nine decisions match the pre-registered expectations in `proposals.json`.

Details:

- known_good: 26/26 public and held-out tests PASSED through v2 frames; differential digest equal on
  3000 edge-heavy inputs; every timed baseline and candidate output equal to the parent's expectation.
  Means 0.946 s → 0.676 s; seconds saved per 3000 cases 0.270 [0.237, 0.303]. Secondary (process wall,
  including about 0.2 s sandbox start and input generation) 1.272 [1.235, 1.311]. In-process
  diagnostic 1.400.
- noop (comment only): passes every correctness stage; rejected by the speedup rule. Its CI is wider
  than the A/A CI because host load rose during that session.
- behaviour_breaking (first-hit matcher, faster but drops the uniqueness check): FAILED
  `test_relaxed_ambiguous_is_not_found` and `test_relaxed_overlapping_windows_are_ambiguous`; never timed.
- gaming_fake_clock_audit: with the policy and static layers disabled the clock-rebinding, memoizing
  module passes the tests and the differential check, but the parent clock, the runner's pre-bound clock
  and per-run fresh inputs leave no measurable gain. Audit runs can never be accepted.
- Note: the v2 static scan alone does not flag the clock patch (`static.flags` is empty); the loop's
  import allowlist does.

## Files

`aa_calibration.json`, `trial_<id>.json` (all stages, per-rep runs, seeds as sha256 only),
`summary.json`, `ledger.jsonl` (proposal_recorded before evaluation, then decision; UTC timestamps).

## Not shown by this run

- The speedup is for this synthetic input distribution (800–2500 line files, about 60% of blocks
  needing the relaxed path); end-to-end effect on a learning-pipeline run was not measured.
- Proposals are scripted; no automated proposer was run.
- One session, one host; not repeated across machines.
