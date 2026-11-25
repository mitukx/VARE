# Task: stabilize support-constrained chunked log-probabilities

At the pinned Hugging Face TRL baseline, the PyTorch chunked fallback in `trl/kernels/chunked_logprob.py` can produce numerically incorrect support-constrained log-probabilities and gradients in edge cases. Repair the implementation so excluded logits do not corrupt support normalization and chunk transitions preserve the support result.

Only edit `trl/kernels/chunked_logprob.py`. Do not add dependencies. The evaluation checks the selected value and gradient behavior under bounded CPU inputs.

This is a source-qualification task. It has no Train allocation, and no model or coding agent is authorized by this task file.
