# TRL stale-resume audit hardening v4

Independent review found two ways to reseal a replay-suffix bundle that the v4
auditor accepted: change each `source_stable` JSON value from boolean `true` to
integer `1`, or replace a manifest artifact with a symlink to identical bytes
outside the bundle. Both pre-fix controls returned `all_checks_passed: true`.
The regression test loads the retained v4 runner to reproduce those accepts,
then verifies that the current auditor rejects them.

The v5 auditor requires exact boolean values for `source_stable` and
`timed_out`, rejects symlinks throughout the bundle, validates normalized
relative manifest paths, and checks that each resolved file stays within the
bundle. The same pinned TRL source-level comparison was rerun with the v5
protocol and passes audit:

| Stage | Result | Candidate tree | Raw log SHA-256 |
| --- | ---: | --- | --- |
| Candidate plus regression test | 2 failed, exit 1 | `1a6661764365ef38b07a0177711b8fac2d67b44d` | `cc30f1d65c82c7d8ab94715535784363bb52d99a5f56d40403ec95e9462f197a` |
| Retained correction, targeted cases | 2 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `ce05e236e01e53ae5495e755396483c0d889968d12122ad072128af8c8b38265` |
| Retained correction, full stale-resume file | 11 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `1c6bb6cff430f26c1393f69e5fcd2ddad8f68e461986f77622426be76c82fe73` |

The [v5 protocol](../protocols/trl_async_stale_resume_replay_suffix_v5.lock.json)
and [accepted raw bundle](../results/trl-async-stale-resume-v7/review-replay-suffix-v5/)
retain the runner, exact patches, stage records, environment, logs, and hashes.
The earlier v4 bundle remains unchanged and passes the stricter current auditor.

This remains deterministic CPU source-level checkpoint metadata evidence. It
does not run model rollouts, optimizer updates, independent task evaluation,
or establish model capability improvement. The correction is not
upstream-accepted.

## Reproduction

Prepare the Python 3.12.12 environment in the
[stale-resume v4 report](trl-async-stale-resume-v4-report.md), then run from
the hardened VARE checkout containing this report. Generation uses the frozen
runner revision required by the v5 protocol; auditing uses the current
checkout:

```bash
set -euo pipefail
VARE_HARDENED_CHECKOUT="$PWD"
git clone https://github.com/mitukx/VARE.git /tmp/vare-v5-runner
git -C /tmp/vare-v5-runner checkout --detach f06d3d5108bc1af89a05c8b08c72fa3585f9b0df
git clone https://github.com/huggingface/trl.git /tmp/vare-trl-review-source
git -C /tmp/vare-trl-review-source checkout --detach a85900829d86bcd45b7db0310725f28d3152289b

(
  cd /tmp/vare-v5-runner
  /tmp/vare-trl-v4/bin/python scripts/reproduce_trl_async_stale_resume_review_v2.py \
    --source-repo /tmp/vare-trl-review-source \
    --protocol protocols/trl_async_stale_resume_replay_suffix_v5.lock.json \
    --output /tmp/vare-replay-suffix-v5-fresh \
    --python /tmp/vare-trl-v4/bin/python
)

/tmp/vare-trl-v4/bin/python "$VARE_HARDENED_CHECKOUT/scripts/reproduce_trl_async_stale_resume_review_v2.py" \
  --audit /tmp/vare-replay-suffix-v5-fresh \
  --protocol "$VARE_HARDENED_CHECKOUT/protocols/trl_async_stale_resume_replay_suffix_v5.lock.json"
cd "$VARE_HARDENED_CHECKOUT"
/tmp/vare-trl-v4/bin/python -m pytest -q tests/test_trl_async_stale_resume_evidence.py
```
