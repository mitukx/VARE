import mlx.core as mx
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import RotatingKVCache

def assert_mask(max_size, keep, history, chunk):
    cache = RotatingKVCache(max_size=max_size, keep=keep)
    kv = mx.zeros((1, 1, history, 4))
    cache.update_and_fetch(kv, kv)
    mask = cache.make_mask(chunk)
    rows = mask.tolist()
    offset = min(max_size - 1, history)
    assert len(rows) == chunk
    assert all(len(row) == offset + chunk for row in rows)
    for row_index, row in enumerate(rows):
        query_index = offset + row_index
        expected = [
            (key_index < keep) or
            (query_index - (max_size - keep) < key_index <= query_index)
            for key_index in range(offset + chunk)
        ]
        assert row == expected
        assert sum(row) == max_size

for case in [(8, 2, 20, 5), (16, 4, 48, 7), (12, 3, 27, 2)]:
    assert_mask(*case)

plain = RotatingKVCache(max_size=8, keep=0)
kv = mx.zeros((1, 1, 20, 4))
plain.update_and_fetch(kv, kv)
assert mx.array_equal(plain.make_mask(5), create_causal_mask(5, 7, window_size=8)).item()
