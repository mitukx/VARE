# Synthetic benchmark v1: review corrections

The frozen [original report](e2e-rl-system-benchmark-v1.md), protocol and raw
results are unchanged. Independent automated review of PR #68 identified:

- The report's “7/7” benchmark-test count is incorrect. At measured revision
  `19f67abf` and retained-results revision `d6514866`, the referenced test file
  contained six test functions. Treat that historical count as **6**, not 7.
  New regression cases added later do not retroactively validate the old count.
- Reduced CLI runs could carry the locked protocol ID despite changing the
  seed/round/fault budget. The current CLI and direct campaign entry point now
  reject nonconformant campaigns before writing evidence. The original report's
  reduced-smoke suggestion is superseded; private fixture tests remain bounded
  code-path checks and are not campaign evidence.

The retained full campaign used the declared locked budget. These corrections
do not establish new performance or model-capability results.
