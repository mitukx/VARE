"""Read-only qualification capture: repository ledger and public metadata, never task rows or weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '467ff55dbf6accf0a59e1f0c7f5741ac53ce21de'
MODELS = ['HuggingFaceTB/SmolLM3-3B', 'Qwen/Qwen3-0.6B']
DATASETS = ['birdsql/bird-critic-1.0-sqlite', 'bigcode/bigcodebench', 'allenai/IFBench_test',
            'thanhdath/CoRE-DB-SQL-R1']
REPOS = ['bird-bench/BIRD-CRITIC-1', 'bigcode-project/bigcodebench',
         'allenai/IFBench', 'uiuc-kang-lab/ReViSQL', 'taoyds/test-suite-sql-eval']
PAPERS = ['2609.18135v1', '2603.01409v1', '2609.03955v1', '2605.02944v1',
          '2605.30478v1', '2402.15833v1', '2609.34064v1', '2507.02833v1']
LEDGER = ['AGENTS.md', 'README.md', 'docs/roadmap.md', 'docs/current-gaps.md',
          'docs/experiments.md', 'docs/research-candidate-qualification-2026-10-15.md',
          'docs/stage0-research-gate-2026-10-11.md', 'docs/next-study-decision-2026-10-09.md',
          'docs/cpu-code-repair-feasibility-v1-report.md',
          'docs/cpu-qwen-ticket-tool-feasibility-v1-report.md']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'schema':'vare.research-qualification-metadata.v1', 'repository_revision':REVISION,
                'started_utc':datetime.now(timezone.utc).isoformat(), 'requests':[], 'ledger':{},
                'models':[], 'datasets':[], 'repositories':[], 'papers':[],
                'scope':'Metadata/source review only. No model inference, optimizer, dataset payload or grader execution.'}

    def fetch(name, url, retain=True):
        record = {'name':name, 'url':url, 'started_utc':datetime.now(timezone.utc).isoformat()}
        began = time.perf_counter()
        try:
            request = urllib.request.Request(url, headers={'User-Agent':'VARE-research-qualification/1.0'})
            with urllib.request.urlopen(request, timeout=25) as response:
                raw = response.read(524289)
                if len(raw) > 524288:
                    raise ValueError('metadata response exceeds 512 KiB bound')
                record.update(status=response.status, response_bytes=len(raw), response_sha256=digest(raw))
                if retain:
                    (output/name).write_bytes(raw)
                    record['retained_file'] = name
                return raw
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            record['error'] = {'type':type(error).__name__, 'message':str(error)}
            return None
        finally:
            record['retrieval_wall_s'] = time.perf_counter()-began
            manifest['requests'].append(record)
            write(output/'capture.json', manifest)

    for name in LEDGER:
        raw = subprocess.check_output(['git','show',REVISION+':'+name], cwd=ROOT)
        manifest['ledger'][name] = {'sha256':digest(raw), 'bytes':len(raw)}
    manifest['collector_sha256'] = digest(Path(__file__).read_bytes())
    inventory = {}
    for term in [*MODELS, 'BIRD-CRITIC', 'bird-critic-1.0-sqlite', 'BigCodeBench', 'IFBench']:
        result = subprocess.run(['git','grep','-n','-i','-I','-F',term,REVISION,'--',
                                 'docs','protocols','configs','requirements','scripts','benchmarks'],
                                cwd=ROOT,capture_output=True,text=True,check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        inventory[term] = sorted({':'.join(line.split(':')[1:3]) for line in result.stdout.splitlines()})
    manifest['tracked_metadata_occurrences'] = inventory
    for index, model in enumerate(MODELS):
        raw = fetch(f'model-{index}.json', 'https://huggingface.co/api/models/'+model)
        if raw is None:
            continue
        data = json.loads(raw)
        entry = {'id':data['id'],'revision':data['sha'],'gated':data.get('gated'),
                 'license':data.get('cardData',{}).get('license'),
                 'parameters':data.get('safetensors',{}).get('total')}
        manifest['models'].append(entry)
        fetch(f'model-{index}-config.json',f"https://huggingface.co/{model}/resolve/{entry['revision']}/config.json")
    for index, dataset in enumerate(DATASETS):
        raw = fetch(f'dataset-{index}.json', 'https://huggingface.co/api/datasets/'+dataset)
        if raw is None:
            continue
        data = json.loads(raw)
        manifest['datasets'].append({'id':data['id'],'revision':data['sha'],'gated':data.get('gated'),
                                     'license':data.get('cardData',{}).get('license'),
                                     'files':[item['rfilename'] for item in data.get('siblings',[])]})
    for index, repo in enumerate(REPOS):
        branch = 'master' if repo=='taoyds/test-suite-sql-eval' else 'main'
        raw = fetch(f'repository-{index}.json', 'https://api.github.com/repos/'+repo+'/commits/'+branch)
        if raw is None:
            continue
        data = json.loads(raw)
        sha = data['sha']
        manifest['repositories'].append({'id':repo, 'revision':sha, 'commit_date':data['commit']['committer']['date']})
        fetch(f'repository-{index}-tree.json', f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1')
        # Inspect only public evaluator source, never dataset or example-output files.
        paths = {0:['evaluation/src/single_instance_eval_sqlite.py', 'evaluation/src/sqlite_test_utils.py',
                    'evaluation/src/sqlite_utils.py'],
                 1:['bigcodebench/evaluate.py'], 2:['run_eval.py'],
                 3:['README.md'], 4:['README.md', 'evaluation.py']}[index]
        for number, path in enumerate(paths):
            fetch(f'repository-{index}-source-{number}.txt', f'https://raw.githubusercontent.com/{repo}/{sha}/{path}')
    for paper in PAPERS:
        # Only bibliography and a response digest are retained, not paper text.
        raw = fetch('paper-'+paper, 'https://arxiv.org/abs/'+paper, retain=False)
        if raw is not None:
            title = re.search(rb'<meta\s+name="citation_title"\s+content="([^"]+)"', raw)
            manifest['papers'].append({'id':paper,'title':title[1].decode() if title else None})
        time.sleep(0.2)
    manifest['capture_wall_s'] = time.perf_counter()-started
    manifest['model_inference_calls'] = manifest['optimizer_calls'] = manifest['task_payload_downloads'] = 0
    write(output/'capture.json', manifest)
    write(output/'SHA256SUMS.json', {p.name:digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({'requests':len(manifest['requests']), 'errors':sum('error' in r for r in manifest['requests']),
                      'metadata_capture_s':manifest['capture_wall_s'], 'model_inference_calls':0}, sort_keys=True))


def verify(output):
    hashes = json.loads((output/'SHA256SUMS.json').read_text())
    if {p.name for p in output.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'} != set(hashes):
        raise ValueError('metadata artifact inventory differs')
    for name, expected in hashes.items():
        if digest((output/name).read_bytes()) != expected:
            raise ValueError(f'metadata artifact checksum differs: {name}')
    manifest = json.loads((output/'capture.json').read_text())
    if manifest.get('repository_revision') != REVISION:
        raise ValueError('capture repository revision differs from the pinned source revision')
    if not manifest['model_inference_calls']==manifest['optimizer_calls']==manifest['task_payload_downloads']==0:
        raise ValueError('capture is not metadata-only')
    for record in manifest['requests']:
        if 'retained_file' in record:
            raw = (output/record['retained_file']).read_bytes()
            if len(raw)!=record['response_bytes'] or digest(raw)!=record['response_sha256']:
                raise ValueError(f'request body differs: {record["retained_file"]}')
    print(json.dumps({'metadata_integrity':'PASS', 'files':len(hashes),
                      'failed_fetches_retained':sum('error' in r for r in manifest['requests'])}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--output', type=Path)
    choice.add_argument('--verify', type=Path)
    args = parser.parse_args()
    verify(args.verify) if args.verify else capture(args.output)
