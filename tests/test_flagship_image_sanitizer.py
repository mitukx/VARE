"""A source-only rewrite must not retain compiled repairs or their hashes."""
import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'benchmarks/flagship_development_v1/sanitize_installed_assets.py'
spec = importlib.util.spec_from_file_location('sanitizer', SOURCE)
sanitizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanitizer)


@pytest.mark.parametrize('wrong_source', [False, True])
def test_installed_repair_assets_are_removed(tmp_path, wrong_source):
    package = tmp_path / 'mlx_lm'
    cache = package / 'quant/__pycache__'
    cache.mkdir(parents=True)
    module = package / 'quant/gptq.py'
    module.write_bytes(b'baseline\n')
    bytecode = cache / 'gptq.cpython-312.pyc'
    bytecode.write_bytes(b'compiled upstream fix')
    record = tmp_path / 'mlx_lm-1.dist-info/RECORD'
    record.parent.mkdir()
    record.write_text('mlx_lm/quant/gptq.py,sha256=forbidden-fixed-hash,999\nmlx_lm/quant/__pycache__/gptq.cpython-312.pyc,,\n')
    expected = {'quant/gptq.py': hashlib.sha256(module.read_bytes()).hexdigest()}
    if wrong_source:
        expected['quant/gptq.py'] = '0' * 64
        with pytest.raises(ValueError, match='baseline mismatch'):
            sanitizer.sanitize(tmp_path, expected)
        assert bytecode.exists()
    else:
        result = sanitizer.sanitize(tmp_path, expected)
        assert result['removed_bytecode_files'] == 1
        assert module.read_bytes() == b'baseline\n'
        assert not bytecode.exists()
        with record.open(newline='') as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 1
        assert 'forbidden-fixed-hash' not in record.read_text()
        assert rows[0][2] == str(len(module.read_bytes()))
