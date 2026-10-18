# TRL async stale-drop resume cursor v2

## Result

The epoch-aware reproduction confirms a stale-drop cursor defect in pinned TRL
commit `6d6901c5e3963108076131ac4979d344566b13a2`. In three matched CPU runs,
the baseline saved cursor and resumed at group 2, then repeated 144, 145, and
148 already-trained absolute prompt-stream positions. The candidate at
`a85900829d86bcd45b7db0310725f28d3152289b` resumes beyond the stale hole and
repeats one in-flight-tail position in each run. Its saved cursor was 147, 147,
and 151; the resume comparison began at 148, 148, and 147 respectively. These
cursor and replay measurements come from two separate tests within each raw
log, so they are not paired as one checkpoint event.

| Arm | Repeated stream positions | Resume cursor behavior | Runs accepted |
| --- | ---: | --- | ---: |
| Baseline | 144, 145, 148 | Stuck at stale group 2 | 3/3 |
| Candidate | 1, 1, 1 | Past stale group; trained-count state preserved | 3/3 |

V2 supersedes v1's row-identity count. The retained v1
[`run-3-final-runner-replay`](../results/trl-async-stale-resume-v1/run-3-final-runner-replay/)
contains a candidate non-pass: its cursor started near the end of the finite
dataset, and the test mistook row 2's valid next-epoch occurrence for a replay
of the earlier dropped event. V2 tags each completion with the absolute stream
position `dataset_start_index + local_group_id`, so later-epoch visits are
distinct from reprocessing the same stream position. The v1 lock and raw
bundles remain unchanged. They are retained as historical records, but the
current auditor refuses to certify them because their reported runner sources
were not preserved.

The reproduction executes production rollout scheduling, stale filtering,
checkpoint serialization, and resume state loading. It stubs only model
generation and reward scoring; it does not load weights, run an optimizer,
exercise vLLM/GPU, or measure task success. Thus this is a source-level
checkpoint/resume correctness result, not model-learning or throughput
evidence.

The upstream candidate adds stale-dropped IDs to the resolved cursor while
keeping actual trained-group counts separate, restores model version on resume,
and covers AsyncGRPO and AsyncDistillation. Its focused test file passed 9/9.
Two related existing checkpoint tests also passed; ten model-backed tests were
skipped by their optional runtime requirements. Upstream PR [#7786](https://github.com/huggingface/trl/pull/7786)
was auto-closed by the first-contributor process check before code review and
was not merged. Issue [#7274](https://github.com/huggingface/trl/issues/7274)
remains open and unassigned; this report does not claim upstream acceptance.

## Review hardening

Automated review found that the original runner pinned `git HEAD` without
requiring clean TRL source checkouts, and the auditor trusted the summary's
six-run count without validating the actual arm/repetition inventory. The
runner now rejects tracked or untracked source changes and rechecks source
identity around each repetition. The auditor checks the locked source
revisions, exact baseline/candidate repetition pairs, uniqueness, and matching
log names. Additional checks bind dependency versions and the public Gist hash
to the lock, compare the recorded exit code with the pytest result in the raw
log, and verify the runner source against a retained snapshot. New output
bundles embed the exact runner source; snapshots for the two passing v2 runs
are in `results/trl-async-stale-resume-runner-snapshots/`.

The old v1 result summaries report runner hashes `b2c47...` and `8d497...`,
but neither source version was retained. The raw results remain untouched;
the auditor fails closed for those v1 bundles instead of treating a
self-reported runner hash as provenance. The v2 run-1 and run-2 runner sources
are retained and hash-checked.

The v2 environment lock names nine direct packages but does not pin every
transitive distribution, and its protocol has no per-arm timeout. Those frozen
records remain valid only under their original v2 audit contract; they are not
retroactively certified against the stronger environment controls. The current
runner therefore refuses new v1/v2 executions. Protocol v3 adds an exact
56-distribution lock and a 600-second watchdog; its matched rerun is reported
in [v3](trl-async-stale-resume-v3-report.md).

After those changes, the same frozen v2 protocol was rerun against clean pinned
TRL checkouts. Baseline repeated stream positions 144, 144, and 145; candidate
repeated one in-flight-tail position in each of three runs. All six decisions
passed and the updated auditor accepted the bundle. See the additional
[review-follow-up bundle](../results/trl-async-stale-resume-v2/run-2-review-fixes/).
The original run remains retained unchanged. The updated evidence regression
suite passes 16 tests, including resealed runner-hash and pytest-failure
controls. This is automated review correction and CPU reproduction; no
independent human review or upstream acceptance is claimed.

## Historical reproduction

The frozen [v2 protocol](../protocols/trl_async_stale_resume_v2.lock.json),
original [bundle](../results/trl-async-stale-resume-v2/run-1-epoch-aware-replay/),
and clean-source [review rerun](../results/trl-async-stale-resume-v2/run-2-review-fixes/)
remain unchanged. Use the [v3 report](trl-async-stale-resume-v3-report.md) for
the current clean-environment reproduction and command. V1 raw bundles also
remain present; the auditor rejects them because their original runner sources
were not retained.

## Limits and next step

This identifies and tests a small correction candidate; it does not establish
that the external project accepts or merges it. The next contribution step is
to satisfy the upstream first-contributor process and obtain maintainer review.
For VARE's broader research goal, the higher-priority unresolved milestone
remains a qualified real-model repository task with an independently trusted
evaluator and measured held-out task success after an actual model update.
