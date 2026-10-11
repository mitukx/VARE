"""Summarize the two frozen public records; never execute transcript commands."""
import collections, hashlib, json, pathlib, re
root = pathlib.Path(__file__).resolve().parent
protocol = json.loads((root / 'protocol.json').read_text())
rows = []
for index in protocol['record_indices']:
    data = (root / f'record-{index}.json').read_bytes()
    row = json.loads(data)
    messages = json.loads(row['messages'])
    if isinstance(messages, str):
        messages = json.loads(messages)
    nested_keys = set()
    def walk(value):
        if isinstance(value, dict):
            nested_keys.update(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(messages)
    roles = dict(collections.Counter(m['role'] for m in messages))
    # These are explicit edit actions, not inferred environment mutations.
    edits = [m['action'] for m in messages if m.get('role') == 'assistant'
             and isinstance(m.get('action'), str)
             and m['action'].startswith('str_replace_editor str_replace ')]
    rows.append({
        'file_row_index': index, 'record_sha256': hashlib.sha256(data).hexdigest(),
        'instance_id': row['instance_id'], 'traj_id': row['traj_id'],
        'model': row['model'], 'reported_resolved': row['resolved'],
        'top_level_keys': sorted(row), 'nested_message_keys': sorted(nested_keys),
        'message_count': len(messages), 'roles': roles,
        'explicit_edit_paths': [x.split()[2] for x in edits],
        'patch_paths': re.findall(r'^diff --git a/(.*?) b/', row['patch'], re.M),
        'token_probability_metadata_present': any(
            k in nested_keys or k in row for k in
            ['logprobs', 'token_logprobs', 'behavior_logprobs', 'logits']),
        'token_ids_recorded': any(k in nested_keys or k in row for k in
                                  ['token_ids', 'input_ids', 'output_ids']),
        'scope': 'This record only; recorded metadata is not a recovered sampling law.'
    })
result = {
    'protocol_sha256': hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest(),
    'rows': rows, 'reuse_admitted': False,
    'decision_basis': [
        'Neither record provides a recoverable behavior sampling law, residual categorical access, or an audited independent target/group lineage.',
        'The recorded patch paths disagree with the explicit edited project files in both specimens. This reproduces an already reported issue, not a new discovery.',
        'No environment replay, transport, learner update or task-success evaluation was executed.'
    ],
    'known_issue': 'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/discussions/3',
    'decision': 'STOP this public-donor route; keep the broader research hypothesis unvalidated.'
}
(root / 'observations.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'records': len(rows), 'reuse_admitted': False,
                  'decision': result['decision']}, indent=2))
