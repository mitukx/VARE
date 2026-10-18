# TRL async stale-drop resume cursor v3

The v3 rerun preserves the epoch-aware comparison while closing two evidence
runner gaps found in review: its clean Python 3.12.12 environment pins all 56
installed distributions, and every pytest arm has a protocol-locked 600-second
timeout. A timeout keeps partial output and is always a failed arm. The frozen
v1 and v2 protocols and result bundles remain unchanged; the current runner
requires v3 for new executions because earlier protocols lack these controls.

| Arm | Repeated stream positions | Exit 0 | Timed out |
| --- | --- | ---: | ---: |
| Pinned TRL baseline | 145, 145, 145 | 3/3 | 0/3 |
| Candidate | 1, 1, 1 | 3/3 | 0/3 |

The baseline resumes at stale group 2 and replays the later trained stream;
the candidate advances past that stale hole and repeats one in-flight tail
position. All six decisions pass the locked criteria. The 56-package inventory,
six raw logs, exact runner source, summary, and SHA-256 manifest are retained in
the [result bundle](../results/trl-async-stale-resume-v3/run-1-full-lock-timeout/)
and audited by the runner. `pip check` found no broken requirements. The
20-test focused evidence suite passes, including dirty-checkout, tampered
transitive dependency, and timeout failure controls.

This remains source-level checkpoint/resume evidence. Generation and scoring
are stubbed; no model weights, optimizer, task-success evaluation, throughput,
or upstream acceptance are demonstrated. TRL PR [#7786](https://github.com/huggingface/trl/pull/7786)
was closed before maintainer code review.

Reproduce from the VARE root on Python 3.12.12:

```bash
python3.12 -m venv /tmp/vare-trl-v3
/tmp/vare-trl-v3/bin/python -m pip install -r requirements/trl-async-stale-resume-v3.txt
git clone https://github.com/huggingface/trl.git /tmp/vare-trl-v3-source
git -C /tmp/vare-trl-v3-source worktree add --detach /tmp/vare-trl-v3-base 6d6901c5e3963108076131ac4979d344566b13a2
git -C /tmp/vare-trl-v3-source remote add vare https://github.com/mitukx/trl.git
git -C /tmp/vare-trl-v3-source fetch vare codex/async-stale-resume-cursor
git -C /tmp/vare-trl-v3-source worktree add --detach /tmp/vare-trl-v3-candidate a85900829d86bcd45b7db0310725f28d3152289b
python scripts/reproduce_trl_async_stale_resume_v1.py \
  --protocol trl-async-stale-resume-v3 \
  --python /tmp/vare-trl-v3/bin/python \
  --baseline-source /tmp/vare-trl-v3-base \
  --candidate-source /tmp/vare-trl-v3-candidate \
  --output /tmp/trl-async-stale-resume-v3
python scripts/audit_trl_async_stale_resume_v1.py /tmp/trl-async-stale-resume-v3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts= -q tests/test_trl_async_stale_resume_evidence.py
```

The immutable [v3 protocol](../protocols/trl_async_stale_resume_v3.lock.json)
and full [resolved dependency lock](../requirements/trl-async-stale-resume-v3.txt)
bind the execution environment and watchdog. Earlier v2 raw results remain
historical evidence with only their original direct dependencies pinned; they
are not retroactively presented as having the v3 environment guarantee.
