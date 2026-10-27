#!/usr/bin/env python3
"""Retrospectively classify tool-call parsing in a completed local-agent run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

from run_local_agent import normalize_tool_call_format, parse_tool_call_blocks, tools_for


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts' / 'run_local_agent.py'
BLOCK_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_arm(evidence: Path, arm: str, manifest: dict) -> dict:
    raw_path = evidence / arm / 'raw-turns.jsonl'
    relative = f'{arm}/raw-turns.jsonl'
    actual_digest = sha256(raw_path)
    expected = manifest['files'][relative]
    if actual_digest != expected['sha256'] or raw_path.stat().st_size != expected['bytes']:
        raise ValueError(f'raw trace does not match frozen evidence manifest: {relative}')

    tools = tools_for('unified_diff', public_tests=True)
    tool_names = {tool['function']['name'] for tool in tools}
    raw_turns = generated_tokens = recovered_turns = recovered_blocks = 0
    accepted_names: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    complete_blocks = 0
    for line in raw_path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        raw = row['model_output']
        normalized, recovered = normalize_tool_call_format(raw, tool_names)
        raw_turns += 1
        generated_tokens += int(row.get('generated_tokens', 0))
        recovered_turns += int(recovered)
        raw_blocks = list(BLOCK_PATTERN.finditer(raw))
        normalized_blocks = list(BLOCK_PATTERN.finditer(normalized))
        recovered_blocks += sum(
            before.group(0) != after.group(0)
            for before, after in zip(raw_blocks, normalized_blocks)
        )
        blocks, parsed, failures = parse_tool_call_blocks(normalized, tools)
        complete_blocks += len(blocks)
        accepted_names.update(name for _, name, _ in parsed)
        rejected.update(error.split(':', 1)[0] for _, error in failures)
    return {
        'raw_turns': raw_turns,
        'generated_tokens': generated_tokens,
        'raw_trace_sha256': actual_digest,
        'turns_with_format_recovery': recovered_turns,
        'format_recovered_blocks': recovered_blocks,
        'complete_tool_call_blocks': complete_blocks,
        'schema_valid_calls': dict(sorted(accepted_names.items())),
        'rejected_blocks_by_error': dict(sorted(rejected.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Frozen run directory containing evidence-manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New append-only report path')
    args = parser.parse_args()
    evidence = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f'refusing to overwrite analysis report: {output}')
    manifest_path = evidence / 'evidence-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    arms = {arm: analyze_arm(evidence, arm, manifest) for arm in ('greedy', 'sampled')}
    result = {
        'schema_version': 1,
        'analysis_id': 'vare.qwen25-3b-agent-v3.retrospective-tool-format-v1',
        'status': 'COMPLETED',
        'method': 'Offline normalization and schema parsing of retained raw turns; no inference, tool execution, edit, or grading.',
        'frozen_run': {
            'protocol_id': 'vare.real-coding-agent-feasibility.mlxlm-1912-qwen25-3b-v3',
            'protocol_sha256': 'ce83fb4d65ec760013299367d1704d3849b9a7ead5ce5866318f01e786add73c',
            'manifest_sha256': sha256(manifest_path),
        },
        'analysis_implementation': {
            'analyzer_sha256': sha256(Path(__file__).resolve()),
            'runner_sha256': sha256(RUNNER),
        },
        'arms': arms,
        'claim_limit': 'Retrospective controller-format analysis only. The three accepted calls were source-inspection tools; no patch call was recovered, and the original 0/2 independent grading result is unchanged.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(output)
    print(f'analysis_sha256={sha256(output)}')
    print(f'schema_valid_calls={sum(sum(arm["schema_valid_calls"].values()) for arm in arms.values())}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
