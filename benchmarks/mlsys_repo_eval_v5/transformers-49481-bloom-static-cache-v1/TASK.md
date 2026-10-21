# BLOOM static-cache generation regression

Generation with a static cache should work when callers omit an attention mask for an unpadded input. A small local BLOOM configuration currently fails on this path. Make the smallest compatible change so generation completes and preserves the input prefix and requested output length. Keep unrelated generation behavior unchanged.

The local source checkout and public smoke test are available in the workspace. Do not use network access, pretrained checkpoints, or Git history.
