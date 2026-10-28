#!/usr/bin/env python3
"""Post-hoc (not preregistered) check of the frozen action-equivalence gate.

The frozen analysis represents a rejected turn by its raw controller output.
On slot-budget exhaustion the controller's raw output embeds the generation
protocol id ("vare.schema-slots.v9" vs "...v10"), so identical model behaviour
compares unequal. This script compares parsed actions and parse statuses, with
rejected turns represented by parse status and generated token ids only. It
does not change the frozen decision; it is reported beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalized(path: Path) -> list[list[object]]:
    record = json.loads((path / 'trajectory.json').read_text())
    return [[t['parse_status'], t['action'], t['stage_generated_token_ids']]
            for t in record['turns'] if t.get('status') == 'completed']


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-analysis', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_analysis.read_text())
    pairs = []
    for pair in frozen['pairs']:
        a, b = normalized(args.runs / pair['v9']), normalized(args.runs / pair['v10'])
        pairs.append({'v9': pair['v9'], 'v10': pair['v10'], 'frozen_actions_equal': pair.get('actions_equal'),
                      'normalized_equal': a == b,
                      'rejected_turns': sum(1 for item in a if item[0] != 'complete_tool_call')})
    out = {'note': 'post-hoc; does not alter the frozen decision', 'pairs': pairs,
           'all_normalized_equal': all(p['normalized_equal'] for p in pairs)}
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'all_normalized_equal': out['all_normalized_equal'], 'pairs': len(pairs)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
