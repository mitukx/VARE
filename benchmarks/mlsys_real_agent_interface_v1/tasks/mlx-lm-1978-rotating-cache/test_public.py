import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

def check(max_size, keep, history, chunk):
    cache = RotatingKVCache(max_size=max_size, keep=keep)
    kv = mx.zeros((1, 1, history, 4))
    cache.update_and_fetch(kv, kv)
    actual = cache.make_mask(chunk).tolist()
    offset = min(max_size - 1, history)
    key_count = offset + chunk
    expected = []
    for row in range(chunk):
        query = offset + row
        visible = []
        for key in range(key_count):
            retained = key < keep
            recent = query - (max_size - keep) < key <= query
            visible.append(retained or recent)
        expected.append(visible)
    assert actual == expected, (actual, expected)

check(max_size=8, keep=2, history=20, chunk=5)
