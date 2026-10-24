# Task: rotating cache attention after wrap

In `mlx_lm/models/cache.py`, fix the attention mask produced by `RotatingKVCache` when `keep > 0` and a multi-token prefill arrives after the cache has wrapped. The cache must retain its prefix positions while using the remaining capacity for the most recent tokens. A query must never attend to a future token. Preserve behavior for caches with no retained prefix.

Only edit `mlx_lm/models/cache.py`. The public test is a small behavioral check; the independent grader also checks different cache sizes and chunk lengths.
