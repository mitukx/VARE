# Task

Repair the SuperPoint forward method so its list comprehensions do not reuse last_hidden_state as a comprehension target. Preserve the outer encoder output for both keypoint and descriptor decoding; keep both decoder comprehensions aligned with the same per-image hidden-state tensor.
