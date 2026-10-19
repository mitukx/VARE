# TRL stale-resume review regression

Review of the pinned candidate patch found an epoch-count bug when an
unresolved in-flight group sits below already-trained groups. With trained IDs
`{0, 1, 3, 4, 5}` and cursor `2`, the candidate rewinds the cursor to `2` but
saves a trained count of `5`. On resume, IDs 3–5 are replayed and counted again,
so epoch stopping can happen early. The same formula was present in both
AsyncGRPO and AsyncDistillation.

The locked regression patch adds this case to the candidate's existing stale
resume tests. Before the correction, both cases fail because the checkpoint
stores `5` where the expected count is `2`. The follow-up patch counts only
trained IDs below the first unresolved cursor. After applying it, the two
regression cases pass and the complete stale-resume test file passes (11 tests).
The correction is retained as a VARE patch artifact; it has not been accepted
or merged upstream.

| Candidate state | Targeted cases | Full stale-resume test file |
| --- | ---: | ---: |
| Pinned candidate `a859008` | 0/2; exit 1 | Not run |
| Candidate plus correction | 2/2; exit 0 | 11/11; exit 0 |

The protocol, exact patches, raw logs, and manifest are retained in
[`trl_async_stale_resume_replay_suffix_v1.lock.json`](../protocols/trl_async_stale_resume_replay_suffix_v1.lock.json)
and [`review-p1-suffix-epoch-count`](../results/trl-async-stale-resume-v5/review-p1-suffix-epoch-count/).
The original v1–v4 protocol files and raw bundles remain unchanged. The
sanitized-runner v4 follow-up is a separate new bundle, and the v4 report now
records that run and its corrected reproduction recipe.

To reproduce on the locked candidate revision, first create the TRL test
environment used by the v4 stale-resume study, then fetch the candidate from
the contributor fork:

```bash
git clone https://github.com/huggingface/trl.git /tmp/vare-trl-review-source
git -C /tmp/vare-trl-review-source remote add vare https://github.com/mitukx/trl.git
git -C /tmp/vare-trl-review-source fetch vare codex/async-stale-resume-cursor
git -C /tmp/vare-trl-review-source checkout --detach a85900829d86bcd45b7db0310725f28d3152289b
git -C /tmp/vare-trl-review-source apply /path/to/VARE/contributions/trl-async-stale-resume-v1/review-replay-suffix/replay-suffix-regression-test.patch
cd /tmp/vare-trl-review-source
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TRL_EXPERIMENTAL_SILENCE=1 /tmp/vare-trl-v4/bin/python -m pytest -o addopts= -q tests/experimental/test_async_stale_resume.py -k resume_cursor_does_not_count_replayed_suffix
```

The last command is expected to fail both cases. Apply the correction and run
the same file:

```bash
git -C /tmp/vare-trl-review-source apply /path/to/VARE/contributions/trl-async-stale-resume-v1/review-replay-suffix/replay-suffix-epoch-count-fix.patch
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TRL_EXPERIMENTAL_SILENCE=1 /tmp/vare-trl-v4/bin/python -m pytest -o addopts= -q tests/experimental/test_async_stale_resume.py
```

This is a deterministic checkpoint-metadata regression, not a full resumed
training run. It does not execute model rollouts, optimizer updates, independent
task evaluation, or establish task-success improvement. The original v4
comparison remains limited to its locked stale-cursor workload.
