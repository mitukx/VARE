"""Measure cached-repair delegation independently of installed source imports."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import qualify_flagship_development_v1 as qualification  # noqa: E402

DELEGATE = b'''import marshal
from pathlib import Path
namespace = {"__name__": "cached_upstream_gptq"}
cache = Path("/usr/local/lib/python3.12/site-packages/mlx_lm/quant/__pycache__/gptq.cpython-312.pyc")
exec(marshal.loads(cache.read_bytes()[16:]), namespace)
gptq_quantize = namespace["gptq_quantize"]
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.protocol.read_bytes()).hexdigest() != args.protocol_sha256:
        raise ValueError('compiled-repair protocol anchor mismatch')
    protocol = json.loads(args.protocol.read_text())
    for name, expected in protocol['inputs_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('experiment input drift: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'delegate.py.txt').write_bytes(DELEGATE)
    task_protocol = json.loads((ROOT / protocol['qualification_protocol']).read_text())
    task = next(task for task in task_protocol['tasks'] if task['id'] == 'gptq-groups')
    calls = []
    for case in task_protocol['gptq_cases']:
        path = ROOT / protocol['qualification_run'] / 'gptq-groups' / (case['id']+'-reference.json')
        value = json.loads(json.loads(path.read_text())['stdout'])
        calls.append(qualification.grading.FunctionCall(case['id'], 'adapter', 'run', [value['request']], value['expected']))
    results = {}
    for name, image in protocol['cache_comparison_images'].items():
        report = qualification.grade_source(DELEGATE, (ROOT / task['adapter']).read_bytes(), calls,
                                            {**task_protocol, 'image': image})
        results[name] = asdict(report)
        (args.output / (name+'.json')).write_text(json.dumps(results[name], indent=2)+'\n')
        print(json.dumps({'image': name, 'passed': report.passed,
                          'cases_passed': sum(case.passed for case in report.cases)}), flush=True)
    accepted = results['original']['passed'] and results['source-only']['passed'] and not results['portable']['passed']
    (args.output / 'summary.json').write_text(json.dumps({
        'protocol_sha256': args.protocol_sha256, 'compiled_repair_leakage_reproduced_and_closed': accepted,
        'results': results, 'model_runs': 0, 'optimizer_steps': 0,
    }, indent=2)+'\n')
    return 0 if accepted else 2


if __name__ == '__main__':
    raise SystemExit(main())
