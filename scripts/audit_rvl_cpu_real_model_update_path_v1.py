#!/usr/bin/env python3
"""Record the v1 update-path run's artifact-retention failure without repairing it."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/rvl_cpu_real_model_update_path_v1.lock.json'
RUNNER = ROOT / 'scripts/run_rvl_cpu_real_model_update_path_v1.py'
ADAPTER = ROOT / 'src/vare/integrations/rvl_grpo.py'
BUNDLE = ROOT / 'results/rvl-cpu-real-model-update-path-v1/run-1'
RVL = Path('/Users/user/Documents/Codex/2026-10-08/https-github-com-mitukx-vare-https/work/rvl-cpu-update-smoke')

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reward(text: str, answer: str) -> float:
    match = re.match(r'^\s*([A-D])(?=$|[\s).,:;])', text)
    return float(bool(match and match.group(1) == answer))

protocol = json.loads(PROTOCOL.read_text())
progress_path = BUNDLE / 'progress.json'
progress = json.loads(progress_path.read_text())
checks = {
    'protocol_hash_matches_frozen_lock': progress['protocol_sha256'] == sha(PROTOCOL),
    'runner_hash_matches_frozen_lock': sha(RUNNER) == protocol['runner_sha256'] == progress['runner_sha256'],
    'adapter_hash_matches_frozen_lock': sha(ADAPTER) == protocol['vare_adapter_sha256'] == progress['vare_adapter_sha256'],
    'pinned_rvl_sources_match': {
        p.relative_to(RVL).as_posix(): sha(p) for p in sorted((RVL / 'src/rvl_systems').rglob('*.py'))
    } == protocol['rvl_source_files_sha256'],
    'summary_json_retained': (BUNDLE / 'summary.json').is_file(),
    'progress_has_all_response_groups': len(progress.get('responses', [])) == 5,
}
replayed = []
for group in progress.get('responses', []):
    expected = next(row for row in protocol['task']['groups'] if row['prompt_id'] == group['prompt_id'])
    values = [reward(row['text'], expected['answer']) for row in group['responses']]
    replayed.append({'prompt_id': group['prompt_id'], 'rewards_replayed': values, 'match': values == group['rewards']})
checks['all_retained_reward_labels_replay'] = all(row['match'] for row in replayed)
checks['selected_first_nonconstant_group_was_reached'] = bool(replayed and len(set(replayed[-1]['rewards_replayed'])) > 1 and all(len(set(row['rewards_replayed'])) == 1 for row in replayed[:-1]))
result = {
    'protocol_id': protocol['protocol_id'],
    'protocol_sha256': sha(PROTOCOL),
    'progress_sha256': sha(progress_path),
    'checks': checks,
    'groups': replayed,
    'evidence_status': 'incomplete_artifact_retention',
    'reason': 'The frozen v1 runner printed its terminal result but did not write summary.json or finalize progress.json. Training metrics and the round-trip record are therefore not preserved as a machine-readable bundle.',
    'limits': ['Do not interpret this record as a validated model update result.', 'The protocol cohort remains consumed and excluded from future use.'],
}
(BUNDLE / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2, sort_keys=True))
