"""Build-time removal of installed upstream repair assets and stale metadata."""
import base64
import csv
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'quant/gptq.py': '81d7df02f5de9ec8cb79390145f34724ee549a1a7cafadaac3edaaa7bc2cb318',
    'tokenizer_utils.py': 'a6c3b727ae30989c5d65d4c1e81b8616cd3e960f02cf6e9cd5474c14926fe49f',
}


def sanitize(site, expected=EXPECTED):
    root = site / 'mlx_lm'
    inventory = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*.py'))}
    for name, digest in expected.items():
        if inventory.get(name) != digest:
            raise ValueError('installed baseline mismatch: ' + name)
    records = list(site.glob('mlx_lm-*.dist-info/RECORD'))
    if len(records) != 1:
        raise ValueError('expected one installation record')
    with records[0].open(newline='') as stream:
        rows = list(csv.reader(stream))
    for name in expected:
        if sum(row[0] == 'mlx_lm/' + name for row in rows) != 1:
            raise ValueError('missing or duplicate installation entry')
    bytecode = list(root.rglob('*.pyc'))
    for path in bytecode:
        path.unlink()
    for row in rows:
        if row[0] in {'mlx_lm/' + name for name in expected}:
            content = (site / row[0]).read_bytes()
            row[1] = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
            row[2] = str(len(content))
    rows = [row for row in rows if not row[0].endswith('.pyc') or (site / row[0]).exists()]
    with records[0].open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*.py'))}
    if after != inventory or list(root.rglob('*.pyc')):
        raise ValueError('source preservation or bytecode cleanup failed')
    return {'removed_bytecode_files': len(bytecode), 'python_sources_preserved': len(after),
            'baseline_source_hashes': expected, 'record_sha256': hashlib.sha256(records[0].read_bytes()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(sanitize(Path('/usr/local/lib/python3.12/site-packages')), sort_keys=True))
