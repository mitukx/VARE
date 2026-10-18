# TRL async stale-drop resume cursor v3

The v3 rerun preserves the epoch-aware comparison while closing two evidence
runner gaps found in review: its clean Python 3.12.12 environment pins all 56
installed distributions, and every pytest arm has a protocol-locked 600-second
timeout. A timeout keeps partial output and is always a failed arm. The frozen
v1–v3 protocols and result bundles remain unchanged as historical evidence;
the current runner requires v4 for new executions because earlier protocols do
not bind the execution platform.

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

V3 is retained as historical evidence. The current runner rejects new v3 runs
because that frozen protocol does not bind the execution platform. To repeat
the comparison under the supported protocol, follow the
[v4 reproduction instructions](trl-async-stale-resume-v4-report.md); v4 uses
the same pinned TRL revisions and full dependency lock while adding a platform
check. Audit the retained v3 result with:

```bash
python scripts/audit_trl_async_stale_resume_v1.py results/trl-async-stale-resume-v3/run-1-full-lock-timeout
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts= -q tests/test_trl_async_stale_resume_evidence.py
```

The immutable [v3 protocol](../protocols/trl_async_stale_resume_v3.lock.json)
and full [resolved dependency lock](../requirements/trl-async-stale-resume-v3.txt)
bind the execution environment and watchdog. Earlier v2 raw results remain
historical evidence with only their original direct dependencies pinned; they
are not retroactively presented as having the v3 environment guarantee.
