#!/usr/bin/env python3
"""Grade one benchmark trajectory's final source with grader-only checks.

Runs in a process separate from the model worker. Hidden checks are never
shown to the model. A trajectory is correct only when the final source parses,
all candidate-visible public tests pass, and all grader-only tests pass.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from vare.synthetic_function_tests_v1 import run_function_tests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--hidden-tests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.trajectory.read_text(encoding='utf-8'))
    fixtures = json.loads(args.fixtures.read_text(encoding='utf-8'))
    hidden = json.loads(args.hidden_tests.read_text(encoding='utf-8'))
    variant = next(item for item in fixtures['variants'] if item['variant_id'] == record['variant'])
    source = record.get('final_source')
    result: dict[str, object] = {
        'grader': 'vare.synthetic-two-bug-repair.grader.v1',
        'variant': record['variant'],
        'hidden_tests_sha256': hashlib.sha256(hidden[record['variant']].encode()).hexdigest(),
    }
    if not isinstance(source, str):
        result.update({'correct': False, 'reason': 'final_source_missing'})
    else:
        result['final_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        result['source_changed'] = source != variant['source']
        try:
            ast.parse(source)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        public = run_function_tests(source, variant['public_tests'])
        grader_only = run_function_tests(source, hidden[record['variant']])
        result.update({'syntax_valid': syntax_ok, 'public': public, 'grader_only': grader_only,
                       'correct': bool(syntax_ok and public['passed'] and grader_only['passed'])})
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=1)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
