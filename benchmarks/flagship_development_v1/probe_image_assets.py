"""Operator-only image probe; never installed in a candidate's filesystem."""
import base64
import csv
import hashlib
import json
from pathlib import Path

site = Path('/usr/local/lib/python3.12/site-packages')
root = site / 'mlx_lm'
with next(site.glob('mlx_lm-*.dist-info/RECORD')).open(newline='') as stream:
    records = {row[0]: row[1:] for row in csv.reader(stream)}
assets = {}
for name in ['quant/gptq.py', 'tokenizer_utils.py']:
    data = (root / name).read_bytes()
    stored, size = records['mlx_lm/' + name]
    encoded = stored.split('=', 1)[1]
    assets[name] = {
        'actual_sha256': hashlib.sha256(data).hexdigest(),
        'record_sha256': base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).hex(),
        'actual_bytes': len(data), 'record_bytes': int(size),
    }
print(json.dumps({
    'assets': assets,
    'bytecode_files': [str(path.relative_to(root)) for path in root.rglob('*.pyc')],
    'python_source_inventory': {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*.py'))
    },
}, sort_keys=True))
