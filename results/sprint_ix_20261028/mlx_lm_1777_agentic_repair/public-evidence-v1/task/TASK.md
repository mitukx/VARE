# Development task: logits processor history during chunked prefill

When MLX-LM generates from a prompt in multiple prefill chunks, a logits processor must receive the same complete prompt-token history that it would receive if the prompt were processed in one chunk. The first processor call currently may omit tokens consumed by earlier prefill chunks. Later processor calls must continue from the complete prompt history and append generated tokens in order.

Inspect the candidate-visible generation implementation and make the smallest correct repair in `mlx_lm/generate.py`. Preserve KV-cache behavior, prefill chunking, input-embedding behavior, and the existing generated-token history contract.

The workspace exposes only the editable source file through source tools. The fixed public smoke checks are available only through the isolated test tool. Do not modify the tests. After a source edit, run the public smoke checks and finish with a concise explanation.

Task family: MLX-LM generation history (Development; one independent bug family).
