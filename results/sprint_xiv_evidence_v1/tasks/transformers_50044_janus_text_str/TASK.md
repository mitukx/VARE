# Task: preserve a single Janus text prompt

`JanusProcessor.__call__` accepts either one prompt string or a list of prompt
strings. A single string must be treated as one prompt in both text generation
and image generation. Keep list inputs working as before.

The source file is `src/transformers/models/janus/processing_janus.py`. Run the
provided public smoke test after editing it. The external grader is hidden.

## Reproduce

The regression is reachable without loading model weights: pass a single string
to the processor while selecting `generation_mode="image"` or text generation
with the default system prompt enabled. The forwarded prompt must remain a
one-element batch; a multi-prompt list must remain a multi-element batch.
