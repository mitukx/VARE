#!/usr/bin/env python3
"""Freeze the v9/v10 multi-turn protocol: bind file, model, and runtime identities, then hash it.

Must run before any model inference for this protocol. Writes protocol.json and
protocol.sha256 exclusively; it refuses to replace an existing frozen protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN_FILES = (
    'scripts/run_v10_multiturn_worker_v1.py', 'scripts/run_v10_multiturn_campaign_v1.py',
    'scripts/grade_v10_multiturn_v1.py', 'scripts/analyze_v10_multiturn_v1.py',
    'scripts/preflight_v10_multiturn_tokenizer_v1.py', 'scripts/freeze_v10_multiturn_protocol_v1.py',
    'scripts/run_local_agent.py', 'src/vare/synthetic_function_tests_v1.py', 'src/vare/schema_slot_protocol.py',
    'src/vare/agent_edit_v2.py', 'src/vare/source_navigation_v1.py', 'src/vare/source_navigation_v2.py',
    'tests/test_v10_multiturn_cache_safety_v1.py',
)
EVIDENCE_FILES = ('fixtures.json', 'grader/hidden_tests.json', 'grader/qualification_sources.json',
                  'tokenizer-preflight.json', 'protocol-template.json')
MODEL_FILES = ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    protocol = json.loads((evidence / 'protocol-template.json').read_text(encoding='utf-8'))
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit('refusing to freeze with tracked modifications')
    preflight = json.loads((evidence / 'tokenizer-preflight.json').read_text(encoding='utf-8'))
    if not preflight['all_prefixes_exact'] or preflight['context_headroom_tokens'] <= 0:
        raise SystemExit('tokenizer preflight failed; not freezing')
    model = args.model.resolve()
    versions = subprocess.run(
        [str(args.python), '-c', 'import sys,importlib.metadata as m;print(sys.version.split()[0]);'
         '[print(m.version(p)) for p in ("mlx","mlx-lm","transformers","psutil")]'],
        capture_output=True, text=True, check=True).stdout.split()
    protocol.update({
        'status': 'FROZEN_BEFORE_MODEL_INFERENCE',
        'implementation_commit': head,
        'model': {**protocol['model'], 'snapshot_path': str(model), 'revision': model.name,
                  'file_sha256': {name: digest(model / name) for name in MODEL_FILES}},
        'runtime': {'python_executable': str(args.python), 'python': versions[0], 'mlx': versions[1],
                    'mlx_lm': versions[2], 'transformers': versions[3], 'psutil': versions[4]},
        'tokenizer_preflight': {key: preflight[key] for key in
                                ('all_prefixes_exact', 'max_scripted_stage_prompt_tokens', 'context_headroom_tokens')},
    })
    protocol['implementation']['file_sha256'] = {
        **{name: digest(REPO / name) for name in FROZEN_FILES},
        **{str((evidence / name).relative_to(REPO)): digest(evidence / name) for name in EVIDENCE_FILES},
    }
    target = evidence / 'protocol.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, indent=1, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sha = digest(target)
    with (evidence / 'protocol.sha256').open('x', encoding='utf-8') as stream:
        stream.write(f'{sha}  protocol.json\n')
    print(sha)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
