# TRL async stale-drop resume cursor v4

V4 closes the remaining execution-provenance gap from review: the runner now
records and audits the operating system, OS release, architecture, Mac model,
processor, installed memory, and CPU-only device. The protocol preserves the
full 56-distribution Python lock and per-arm 600-second timeout from v3. The
frozen v1-v3 protocols and bundles are unchanged.

| Arm | Repeated absolute stream positions | Accepted | Timed out |
| --- | ---: | ---: | ---: |
| Pinned TRL baseline | 144, 144, 144 | 3/3 | 0/3 |
| Candidate | 1, 1, 0 | 3/3 | 0/3 |

The baseline resumes at stale group 2 and replays later trained positions. The
candidate advances its cursor past stale drops; two repetitions replay one
in-flight tail position and one has no repeated position. All six decisions
pass the locked acceptance rules. The run used Python 3.12.12, CPU, Apple M1
Pro (MacBookPro18,3), 32 GiB, macOS 27.0.1 / Darwin release 27.0.0, and the
locked 56-package inventory. The protocol rejects a different platform or
package inventory. The raw
bundle contains all six logs, the exact runner, summary, and manifest at
[`run-1-platform-locked`](../results/trl-async-stale-resume-v4/run-1-platform-locked/).

A runner-hardening rerun clears inherited `PYTHONHOME`, `PYTHONPATH`,
`PYTHONUSERBASE`, and `PYTHONEXECUTABLE` before each pytest arm and disables the
user site directory. On the same locked platform and dependency inventory, all
six decisions passed again without timeout; baseline duplicate counts were
145, 145, and 147, and candidate counts were 1, 1, and 1. The retained runner
and raw logs are in
[`run-2-python-env-sanitized`](../results/trl-async-stale-resume-v4/run-2-python-env-sanitized/).

This is source-level checkpoint/resume evidence. The reproduction uses the
production rollout scheduling, stale filtering, checkpoint-save, and resume
paths, while stubbing generation and scoring. It does not load model weights,
perform an optimizer update, measure task success or throughput, or establish
an upstream-accepted fix. TRL PR [#7786](https://github.com/huggingface/trl/pull/7786)
was closed before maintainer code review.

The v4 protocol's original runner is retained at VARE commit
`54ff88c368d98c66111489d7b23046df62112d59`. A fresh reproduction must use
that revision; the current reproducer has since changed and is not approved by
the frozen v4 protocol. Fetch the PR branch to make this historical commit
available, then run the locked commands from its checkout on the M1 Pro /
macOS environment:

```bash
set -e
git clone https://github.com/mitukx/VARE.git /tmp/vare-v4-source
git -C /tmp/vare-v4-source fetch origin codex/trl-async-stale-resume-cursor
git -C /tmp/vare-v4-source worktree add --detach /tmp/vare-v4-runner 54ff88c368d98c66111489d7b23046df62112d59
git -C /tmp/vare-v4-source worktree add --detach /tmp/vare-v4-hardened f001a387b0472481c87d3d9372ddcc9c66035663
cd /tmp/vare-v4-runner
python3.12 -m venv /tmp/vare-trl-v4
/tmp/vare-trl-v4/bin/python -m pip install -r requirements/trl-async-stale-resume-v3.txt
git clone https://github.com/huggingface/trl.git /tmp/vare-trl-v4-source
git -C /tmp/vare-trl-v4-source worktree add --detach /tmp/vare-trl-v4-base 6d6901c5e3963108076131ac4979d344566b13a2
git -C /tmp/vare-trl-v4-source remote add vare https://github.com/mitukx/trl.git
git -C /tmp/vare-trl-v4-source fetch vare codex/async-stale-resume-cursor
git -C /tmp/vare-trl-v4-source worktree add --detach /tmp/vare-trl-v4-candidate a85900829d86bcd45b7db0310725f28d3152289b
/tmp/vare-trl-v4/bin/python scripts/reproduce_trl_async_stale_resume_v1.py \
  --protocol trl-async-stale-resume-v4 \
  --python /tmp/vare-trl-v4/bin/python \
  --baseline-source /tmp/vare-trl-v4-base \
  --candidate-source /tmp/vare-trl-v4-candidate \
  --output /tmp/trl-async-stale-resume-v4-fresh
cd /tmp/vare-v4-hardened
/tmp/vare-trl-v4/bin/python scripts/audit_trl_async_stale_resume_v1.py /tmp/trl-async-stale-resume-v4-fresh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/vare-trl-v4/bin/python -m pytest -o addopts= -q tests/test_trl_async_stale_resume_evidence.py
```

The frozen v4 runner generates the evidence, then the hardened checkout audits
it and runs the regression suite. `set -e` keeps any nonzero reproduction or
audit status from being masked by a later successful command.

The [v4 protocol](../protocols/trl_async_stale_resume_v4.lock.json) binds the
platform and points to the already frozen full dependency lock. V4 records no
model-learning or capability result. The runner defaults to v4 and rejects
new reproductions under v1-v3; those versions remain available for historical
bundle audits only.
