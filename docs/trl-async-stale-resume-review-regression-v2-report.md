# TRL stale-resume review regression v2

The pinned AsyncGRPO/AsyncDistillation candidate rewinds its rollout cursor to
the first unresolved group but counts later trained groups in the saved epoch
counter. With trained IDs `{0, 1, 3, 4, 5}` and cursor `2`, the checkpoint
records `5` rather than `2`; replaying IDs 3–5 after resume can count their
training signals twice.

The v2 protocol reruns the candidate regression and retained correction on
TRL commit `a85900829d86bcd45b7db0310725f28d3152289b`. The candidate fails both
targeted cases. The correction passes both targeted cases and all 11 tests in
the stale-resume file. The run used CPU on an Apple M1 Pro with Python 3.12.12,
PyTorch 2.9.1, Transformers 5.20.1, and the dependency/platform lock inherited
from protocol v4.

| Stage | Result | Candidate tree | Raw log SHA-256 |
| --- | ---: | --- | --- |
| Candidate plus regression test | 2 failed, exit 1 | `1a6661764365ef38b07a0177711b8fac2d67b44d` | `24b995c57c1a89810609c57febda4f706297e24421b2ad27f572d9e502dcccbc` |
| Correction, targeted cases | 2 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `946da250c103e0bd792514745f068a639a30956b43ba84f4d10f70455ade8b3e` |
| Correction, complete test file | 11 passed, exit 0 | `5719aadfebf9ec559e98edae6076e1d1a29e5096` | `d55e4b3093af180ed064be719ac92ef4855430a3c16b5533dd4866011542b6da` |

The retained v2 bundle includes the runner and frozen protocol, exact source
diffs, candidate revision/tree identities, test and patch hashes, stage JSON,
raw logs, and a manifest. Its audit passes, and the regression suite checks
that every stage JSON digest matches its raw log. The earlier review bundle
stored under `results/trl-async-stale-resume-v5/` has a baseline metadata
digest that disagrees with its raw log. That frozen bundle is preserved
unchanged and is not used for the v2 result; the v2 baseline stage record is
bound to its matching raw log digest.

An earlier run completed the same three test stages but was rejected because
the initial v2 auditor compared the summary environment against a nonexistent
top-level field. Its raw logs and rejection record are retained under
[`review-p1-suffix-epoch-count-provenance-v2-rerun`](../results/trl-async-stale-resume-v6/review-p1-suffix-epoch-count-provenance-v2-rerun/);
only the final bundle above is accepted.

This is a deterministic CPU checkpoint-metadata regression. It does not run
model rollouts, optimizer updates, independent task evaluation, or establish
model capability improvement. The correction is retained as a VARE patch and
has not been accepted upstream.

## Reproduction

Prepare `/tmp/vare-trl-v4/bin/python` using the environment recipe in the
[v4 report](trl-async-stale-resume-v4-report.md). This protocol pins its
runner and helper to VARE commit `b456452257a5490d61ff5da92fd818dba38e672e`;
use that retained source revision for a fresh reproduction so the recorded
runner hashes match the frozen protocol:

```bash
set -e
git clone https://github.com/mitukx/VARE.git /tmp/vare-review-v2-source
git -C /tmp/vare-review-v2-source fetch origin codex/trl-async-stale-resume-cursor
git -C /tmp/vare-review-v2-source worktree add --detach /tmp/vare-review-v2-runner b456452257a5490d61ff5da92fd818dba38e672e
git -C /tmp/vare-review-v2-source worktree add --detach /tmp/vare-review-v2-hardened 557dffd25e45f43b98c09b2740ae6a1239f0dff6
git clone https://github.com/huggingface/trl.git /tmp/vare-trl-review-source
git -C /tmp/vare-trl-review-source remote add vare https://github.com/mitukx/trl.git
git -C /tmp/vare-trl-review-source fetch vare codex/async-stale-resume-cursor
git -C /tmp/vare-trl-review-source checkout --detach a85900829d86bcd45b7db0310725f28d3152289b
cd /tmp/vare-review-v2-runner
/tmp/vare-trl-v4/bin/python scripts/reproduce_trl_async_stale_resume_review_v2.py \
  --source-repo /tmp/vare-trl-review-source \
  --output /tmp/vare-replay-suffix-v2-fresh
cd /tmp/vare-review-v2-hardened
/tmp/vare-trl-v4/bin/python scripts/audit_trl_async_stale_resume_review_v2.py \
  /tmp/vare-replay-suffix-v2-fresh
/tmp/vare-trl-v4/bin/python -m pytest -q tests/test_trl_async_stale_resume_evidence.py
```

The pinned v2 runner is used only to generate the bundle. The strict audit
entrypoint and evidence tests run from the hardened checkout, which rejects
contradictory raw markers and malformed outcome counts. Do not run the audit
from the frozen runner checkout.

The authoritative result bundle is
[`review-p1-suffix-epoch-count-provenance-v2-final`](../results/trl-async-stale-resume-v6/review-p1-suffix-epoch-count-provenance-v2-final/).
