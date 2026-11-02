#!/usr/bin/env python3
"""Apply protocol amendment 001: evaluate the probe gate on the v2 probe output.

The frozen analysis is run unchanged first and retained. This script reads its
output, re-evaluates only the probe-dependent part of the kv_correctness gate
with the frozen thresholds on the v2 probe, and recomputes the decision with
the frozen decision order. Nothing else is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--frozen-analysis', type=Path, required=True)
    parser.add_argument('--probe-v2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    frozen = json.loads(args.frozen_analysis.read_text())
    probe = json.loads(args.probe_v2.read_text())
    rules = protocol['decision_rule']['thresholds']
    probe_ok = bool(probe['probes']) and all(
        p.get('token_prefix_exact') and p['restored_vs_full']['argmax_equal']
        and p['restored_vs_full']['top5_overlap'] >= rules['probe_min_top5_overlap'] for p in probe['probes'])
    stale_detected = all(p.get('changed_tool_response_detected_by_exact_check', True) for p in probe['probes'])
    audit_violations = sum(run.get('audit_violations', 0) for run in frozen['runs'])
    gates = dict(frozen['gates'])
    gates['kv_correctness'] = audit_violations == 0 and probe_ok and stale_detected
    if not gates['kv_correctness'] or not gates['retention_bounded']:
        decision = 'NO_GO_V10_KEEP_V9_DEFAULT'
    elif not gates['action_equivalence'] or not gates['reliability'] or not gates['efficiency']:
        decision = 'KEEP_V9_DEFAULT_V10_EXPLICIT_OPT_IN'
    elif not gates['memory']:
        decision = 'KEEP_EXPLICIT_FALLBACK_V10_OPT_IN_UNDER_MEMORY_HEADROOM'
    else:
        decision = 'PROMOTE_V10_DEFAULT_RETAIN_EXPLICIT_V9_FALLBACK'
    out = {'amendment': 'amendment-001', 'frozen_decision': frozen['decision'], 'frozen_gates': frozen['gates'],
           'probe_v2_ok': probe_ok, 'probe_v2_stale_prefix_detected': stale_detected,
           'audit_violations': audit_violations, 'gates': gates, 'decision': decision}
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
