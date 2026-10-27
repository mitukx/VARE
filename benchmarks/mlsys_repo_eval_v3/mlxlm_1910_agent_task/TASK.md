Fix Qwen3 Coder argument parsing when a parameter schema has no top-level type. Preserve existing behavior for typed parameters and keep malformed input from crashing parsing.

Editable source: qwen3_coder.py. The public regression is test_public.py.
