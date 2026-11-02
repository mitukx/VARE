FROM vare-mlx-cpu@sha256:a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7
# Build context contains only these two immutable upstream baseline modules.
# Candidate code must not obtain an upstream repair from site-packages.
COPY gptq.py /usr/local/lib/python3.12/site-packages/mlx_lm/quant/gptq.py
COPY tokenizer_utils.py /usr/local/lib/python3.12/site-packages/mlx_lm/tokenizer_utils.py
RUN python - <<'PY'
from pathlib import Path
import hashlib
root = Path('/usr/local/lib/python3.12/site-packages/mlx_lm')
expected = {
    'quant/gptq.py': '81d7df02f5de9ec8cb79390145f34724ee549a1a7cafadaac3edaaa7bc2cb318',
    'tokenizer_utils.py': 'a6c3b727ae30989c5d65d4c1e81b8616cd3e960f02cf6e9cd5474c14926fe49f',
}
for name, digest in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise ValueError('baseline source hash mismatch: ' + name)
for path in root.rglob('*.pyc'):
    path.unlink()
PY
