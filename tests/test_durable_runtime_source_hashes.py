"""Supplement the frozen malformed-asset matrix with valid BLOB drift."""
import pytest

from benchmarks.scheduler.fixtures import make_fixture
from vare.durable import RuntimeMismatchError, Store


@pytest.mark.parametrize('name', ['__init__.py', '__main__.py', 'runner.py', 'durable.py', '_grader_guard.py'])
def test_each_source_hash_is_checked_for_valid_blob_assets(tmp_path, name):
    root, job = make_fixture(tmp_path, 'valid')
    store = Store.create(tmp_path / 'state.db', [job], root)
    with store.connection(True) as db:
        original = db.execute('SELECT content FROM assets WHERE name=?', ('vare/' + name,)).fetchone()[0]
        assert isinstance(original, bytes)
        db.execute('UPDATE assets SET content=? WHERE name=?', (original + b'\n# enrolled source drift\n', 'vare/' + name))
        assert db.execute('SELECT typeof(content) FROM assets WHERE name=?', ('vare/' + name,)).fetchone()[0] == 'blob'
    before = store.jobs()
    with pytest.raises(RuntimeMismatchError, match='differs from enrolled source') as failure:
        store.claim('changed')
    assert name in str(failure.value)
    assert store.jobs() == before
    with store.connection() as db:
        assert db.execute('SELECT count(*) FROM attempts').fetchone()[0] == 0
