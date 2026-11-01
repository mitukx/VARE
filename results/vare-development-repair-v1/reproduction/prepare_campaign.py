from pathlib import Path
import ast
import datetime
import hashlib
import importlib.metadata
import json
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "work/VARE"
ASSETS = ROOT / "benchmarks/vare_development_repair_v1"
sys.path.insert(0, str(ROOT))
from scripts import grade_vare_development_candidate_v1 as g

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')

def transplant(baseline, fixed, owner, method):
    bt, ft = ast.parse(baseline), ast.parse(fixed)
    b, f = g.select(bt, owner, method), g.select(ft, owner, method)
    # Preserve every non-target byte. AST line positions are one-based.
    lines, fresh = baseline.splitlines(keepends=True), fixed.splitlines(keepends=True)
    return ''.join(lines[:b.lineno - 1] + fresh[f.lineno - 1:f.end_lineno] + lines[b.end_lineno:])

prompts = {
    'navigation-overflow': 'Repair SourceNavigatorV1._bounded in src/vare/source_navigation_v1.py. Search indexes up to 128 valid source paths. With many long paths, the indexed_paths metadata can exhaust the serialized JSON response budget, removing every useful match and yielding output_limit_too_small. Preserve useful ranked matches when the auxiliary inventory is too large. Every response must still fit max_output_bytes and report an accurate output_bytes when present. Keep existing invalid-query and no-match behavior. Change only the body of _bounded, preserving its signature and every other definition.',
    'navigation-retry': 'Repair SourceNavigatorV1.search in src/vare/source_navigation_v1.py. A source file can temporarily exceed the index limits or be unreadable. After that file is repaired, subsequent searches currently keep returning the old build error for the rest of the session. Make valid later searches recover, retaining bounded rebuilds, current error behavior while the source remains invalid, accurate index_rebuilt metadata, and failure for an empty allowlist. Change only the body of search, preserving its signature and every other definition.',
    'manifest-nonempty': 'Repair manifest_entries in scripts/verify_coding_agent_evidence_manifest_v1.py. A manifest that omits files, or supplies an empty object or empty array, currently verifies zero files and can be reported as completed. Require a nonempty files inventory. Invalid inventory must return a nonempty errors list with no entries. Continue to support nonempty path-map and array formats, retaining duplicate-path and malformed-array checks. Change only the body of manifest_entries, preserving its signature and every other definition.',
}
for task in prompts:
    prompts[task] += ' Do not add imports, IO, reflection, nested definitions, or changes to module state. Hidden behavioral checks run only after the attempt. This is Development feasibility, not held-out evaluation.'

qualification = BASE / 'outputs/task-qualification'
qualification.mkdir()
inputs = qualification / 'inputs'
inputs.mkdir()
rows = []
for task, (asset, owner, method) in g.TASKS.items():
    baseline = (ASSETS / asset).read_text()
    fixed_path = ROOT / ('src/vare/source_navigation_v1.py' if owner else 'scripts/verify_coding_agent_evidence_manifest_v1.py')
    fix = transplant(baseline, fixed_path.read_text(), owner, method)
    if task == 'navigation-overflow':
        mutant = fix.replace('results = result.get("results")', 'result["results"] = []\n        results = result.get("results")')
    elif task == 'navigation-retry':
        mutant = fix.replace('self._last_build_error is not None or self._is_stale()', 'self._last_build_error is not None and self._is_stale()')
    else:
        mutant = fix.replace("manifest.get('files')", "manifest.get('files', {'placeholder': {}})")
    for variant, text in [('baseline', baseline), ('known-fix', fix), ('incorrect-mutant', mutant)]:
        path = inputs / f'{task}-{variant}.py'
        path.write_text(text)
        rows.append({'task': task, 'variant': variant, 'path': str(path), 'sha256': sha(path)})

source_names = [
    'scripts/run_local_agent.py', 'scripts/run_vare_development_campaign_v1.py',
    'scripts/development_campaign_model_entry_v1.py', 'scripts/grade_vare_development_candidate_v1.py',
    'src/vare/agent_edit_v2.py', 'src/vare/agent_edit_format.py', 'src/vare/agent_task_contract.py',
    'src/vare/source_navigation_v1.py', 'benchmarks/vare_development_repair_v1/navigation-baseline.py.txt',
    'benchmarks/vare_development_repair_v1/manifest-baseline.py.txt',
]
frozen = {n: sha(ROOT / n) for n in source_names}
write(ASSETS / 'qualification-protocol.json', {
    'id': 'vare-development-qualification-v1', 'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_hashes': frozen, 'controls': rows, 'expectations': {'baseline': False, 'known-fix': True, 'incorrect-mutant': False},
    'acceptance': 'All three baseline/fix/mutant controls have matching source hashes, are admitted, complete within 30 seconds and meet the fixed expected success values. Admission interference probes must pass. No protocol modification or task retry after outcome.',
    'scope': 'Three fresh VARE Development bug contracts, two source families; investigator-authored separate behavioral oracle. Same-host calibration is not external independent review or model-unseen certification.',
})

model_roots = [
    ('qwen25-3b', 'mlx-community/Qwen2.5-3B-Instruct-4bit', '4f83f8f146fdf28b512a06562b671d7af4fab457'),
    ('qwen3-4b', 'mlx-community/Qwen3-4B-Instruct-2507-4bit', '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'),
]
cases = []
for label, repo, snapshot in model_roots:
    model_path = Path('/Users/user/.cache/huggingface/hub') / ('models--' + repo.replace('/', '--')) / 'snapshots' / snapshot
    hashes = {p.name: sha(p) for p in sorted(model_path.iterdir()) if p.is_file()}
    for task, (asset, owner, method) in g.TASKS.items():
        source_path = 'src/vare/source_navigation_v1.py' if owner else 'scripts/verify_coding_agent_evidence_manifest_v1.py'
        cases.append({'id': f'{label}-{task}', 'task': task, 'baseline_asset': asset, 'source_path': source_path,
                      'upstream_revision': '6d95bd9196645ec7ecf7ff5d0c9ec997b24dcab1' if owner else '58ecd9c9727a1544a5a08c9c250f1f36dd9a5f8d',
                      'prompt': prompts[task], 'instruction': f'Repair {method} in {source_path}. Apply a minimal source edit and then call finish.',
                      'seed': 101, 'model_path': str(model_path), 'model_repository': repo, 'model_files_sha256': hashes})
write(ASSETS / 'campaign-protocol.json', {
    'id': 'vare-development-agent-campaign-v1', 'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'hypothesis': 'At least one cached local checkpoint can produce an admitted verifier-correct repair on these fresh Development contracts using the materially different navigation interface and a 1024-token turn budget.',
    'cheaper_alternative': 'Unit regressions and baseline/fix/mutant qualification first; they can establish the defect and oracle but cannot establish model tool use or model-generated repair.',
    'primary_metric': 'Verified task success per model/task; one greedy attempt each. Report admitted patches, no-edit outcomes, tool transport, token budgets, wall time and memory. No significance or population accuracy claim for six cells.',
    'cases': cases, 'source_hashes': frozen, 'model_python': '/Users/user/.cache/vare-mlx-venv/bin/python',
    'resources': {'minimum_available_memory_bytes': 6 * 1024**3, 'minimum_free_swap_bytes': 512 * 1024**2,
                  'rss_limit_bytes': 5 * 1024**3, 'metal_allocation_limit_bytes': 6 * 1024**3,
                  'swap_growth_limit_bytes': 256 * 1024**2, 'model_wall_limit_s': 660, 'maximum_paid_spend': 0,
                  'max_turns': 12, 'max_tokens_per_turn': 1024, 'max_tokens_per_attempt': 4096, 'max_context_tokens': 12288},
    'fault_case': 'qwen25-3b-manifest-nonempty',
    'fault_experiment': 'One actual SIGKILL after durable model rollout before grading; recovery must bind every pending artifact, grade the retained candidate once, and retain exactly one model invocation. No within-generation recovery or speedup claim.',
    'stopping': 'Shared resource/inference failure stops remaining cells. No outcome-driven retries, prompt edits, or grader changes. Always grade completed unchanged candidates as failures.',
    'decision': 'Positive repairs authorize only consideration of a separately frozen update-cost smoke. SFT/GRPO and promotion remain NO-GO without the independent eligible 15-family set and training gates.',
    'claim_boundary': 'Development on three narrow VARE module contracts, two related task families, public future-code exposure unknown. No independent external review, held-out gain, trained-policy gain, or general repository-agent success claim.',
})
print(json.dumps({'campaign_protocol': sha(ASSETS/'campaign-protocol.json'), 'qualification_protocol': sha(ASSETS/'qualification-protocol.json'), 'controls': len(rows), 'cases': len(cases)}, indent=2))
