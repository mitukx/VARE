# Phi3 long-context generation consistency

For a small Phi3 model using LongRoPE, cached generation should produce the same next-token logits as a full-prefix forward pass at every generated step, including when the configured sliding window is enabled. The current behavior diverges after crossing the original context boundary. Correct the cache behavior while preserving ordinary generation semantics.

The local source checkout and public smoke test are available in the workspace. Do not use network access, pretrained checkpoints, or Git history.
