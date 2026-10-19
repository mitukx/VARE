# TRL stale-resume evidence audit hardening v6

An independent review showed that Python's default JSON decoder accepts
duplicate object keys using the last value. A resealed summary could therefore
contain contradictory `all_checks_passed` values, and a stage record could
contain contradictory provenance flags, while the auditor saw only the final
value. The retained v4 auditor accepts both controls. The current auditor now
rejects duplicate keys in every JSON object through a strict object-pairs
decoder; the regression tests verify old-auditor acceptance and current-auditor
rejection for both summary and stage-record attacks.

The new v6 protocol leaves v1-v5 locks and bundles unchanged. A fresh run on
TRL commit `a85900829d86bcd45b7db0310725f28d3152289b` reproduced two expected
baseline failures, then passed the two targeted cases and all eleven stale-
resume tests after applying the retained correction. All source states were
stable, and the final bundle audit passed. It retains twelve manifest entries,
including runner, protocol snapshot, three stage records and logs, source
patches, environment, and the duplicate-key regression-test output.

| Stage | Result | Candidate tree | Raw log SHA-256 |
| --- | ---: | --- | --- |
| Candidate plus regression test | 2 failed, exit 1 | `1a6661764365ef38b07a0177711b8fac2d67b44d` | `5a5711a0f9bb8b801b89decd2038fa6e4fc5795b530ca3cdd95c229f7be82ff3` |
| Retained correction, targeted cases | 2 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `6308d66149979777b61a167773157b09c723ebc2ab91d7090b8209ff37b4f969` |
| Retained correction, full test file | 11 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `ffe1e844e15d6185ceca42fb815895380a6d62b53c12a8e1d1e3c552f0a38f82` |

The duplicate-key negative control passes two cases against the frozen auditor,
and the current auditor rejects both malformed bundles. The protocol pins
runner SHA-256 `8d6038fd7fa66ac6890d472bedcae08c92ff1d24d5eb2d7a96df2634febee0e2`.
The control-test log SHA-256 is
`99db33d5c94c6f7da021687c7391181685cc33299c5435b1c861e2db3be3ec87`.
See the [v6 protocol](../protocols/trl_async_stale_resume_replay_suffix_v6.lock.json),
[raw bundle](../results/trl-async-stale-resume-v7/review-replay-suffix-v6/),
and [regression test](../tests/test_trl_async_stale_resume_evidence.py).

This is CPU source-level checkpoint-metadata and evidence-auditor validation.
It does not execute model rollouts or optimizer updates, establish independent
task success, or demonstrate a capability improvement.

## Reproduction

Prepare the locked Python environment as described in the
[stale-resume v4 report](trl-async-stale-resume-v4-report.md), then run from
this VARE revision:

```bash
set -euo pipefail
git clone https://github.com/huggingface/trl.git /tmp/vare-trl-v6-source
git -C /tmp/vare-trl-v6-source remote add vare https://github.com/mitukx/trl.git
git -C /tmp/vare-trl-v6-source fetch vare codex/async-stale-resume-cursor
git -C /tmp/vare-trl-v6-source checkout --detach a85900829d86bcd45b7db0310725f28d3152289b
/tmp/vare-trl-v4/bin/python scripts/reproduce_trl_async_stale_resume_review_v2.py \
  --source-repo /tmp/vare-trl-v6-source \
  --protocol protocols/trl_async_stale_resume_replay_suffix_v6.lock.json \
  --output /tmp/vare-replay-suffix-v6-fresh \
  --python /tmp/vare-trl-v4/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/vare-trl-v4/bin/python -m pytest -q \
  tests/test_trl_async_stale_resume_evidence.py::test_replay_suffix_audit_rejects_duplicate_json_keys
/tmp/vare-trl-v4/bin/python scripts/audit_trl_async_stale_resume_review_v2.py \
  /tmp/vare-replay-suffix-v6-fresh \
  --protocol protocols/trl_async_stale_resume_replay_suffix_v6.lock.json
```
