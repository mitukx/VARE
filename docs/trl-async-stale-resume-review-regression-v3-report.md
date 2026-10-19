# TRL stale-resume evidence hardening v3

The review found two evidence-runner defects: ignored repository files could be
present before source capture, and a post-test source-validation exception
could prevent pytest output from being retained. The runner now rejects
tracked, untracked, and ignored contamination before each stage; writes raw
pytest output before post-test validation; records validation failure as a
non-pass stage; and stops before running later stages. Provenance markers are
newline-separated from test output so the bundle auditor can parse them.

The pinned TRL candidate remains commit
`a85900829d86bcd45b7db0310725f28d3152289b`. The accepted v4 replay uses the
same frozen regression and correction patches as v2:

| Stage | Result | Candidate tree | Raw log SHA-256 |
| --- | ---: | --- | --- |
| Candidate plus regression test | 2 failed, exit 1 | `1a6661764365ef38b07a0177711b8fac2d67b44d` | `a090bffee51ab879c7a2fe8753ddb57526fdc23f77f113e03e2375115d974d8a` |
| Retained correction, targeted cases | 2 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `0db334e876ca452b3e667711da57c31feb5d01b000ced0cc3d6d079d9833cae1` |
| Retained correction, full stale-resume file | 11 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `ba8cc79b16faab247ca810ffe8cef5761f391d0750a99d3f81f32881e18063b7` |

The v3 execution is retained as a failed harness attempt: all three source
stages ran, but the auditor rejected the first raw log because its last
provenance marker ran into the first pytest line. V4 fixes that separator,
and its complete bundle passes audit. The v3 record remains unchanged at
[`review-replay-suffix-v3`](../results/trl-async-stale-resume-v7/review-replay-suffix-v3/).
The accepted [v4 protocol](../protocols/trl_async_stale_resume_replay_suffix_v4.lock.json)
and [raw bundle](../results/trl-async-stale-resume-v7/review-replay-suffix-v4/)
retain the full source state, diffs, commands, environment, stage records,
logs, and manifest.
The v3 auditor rejection is retained separately at
[`review-replay-suffix-v3-audit-rejection.log`](../results/trl-async-stale-resume-v7/review-replay-suffix-v3-audit-rejection.log).

This is CPU source-level checkpoint metadata evidence. It does not exercise
model rollouts, optimizer updates, independent task evaluation, or establish a
capability improvement. The correction remains a retained contribution and
has not been accepted upstream.

## Reproduction

Prepare `/tmp/vare-trl-v4/bin/python` as described in the
[stale-resume v4 report](trl-async-stale-resume-v4-report.md), then run from a
clean VARE checkout:

```bash
git clone https://github.com/huggingface/trl.git /tmp/vare-trl-review-source
git -C /tmp/vare-trl-review-source checkout --detach a85900829d86bcd45b7db0310725f28d3152289b
python scripts/reproduce_trl_async_stale_resume_review_v2.py \
  --source-repo /tmp/vare-trl-review-source \
  --protocol protocols/trl_async_stale_resume_replay_suffix_v4.lock.json \
  --output /tmp/vare-replay-suffix-v4-fresh \
  --python /tmp/vare-trl-v4/bin/python
python scripts/reproduce_trl_async_stale_resume_review_v2.py \
  --audit /tmp/vare-replay-suffix-v4-fresh \
  --protocol protocols/trl_async_stale_resume_replay_suffix_v4.lock.json
```
