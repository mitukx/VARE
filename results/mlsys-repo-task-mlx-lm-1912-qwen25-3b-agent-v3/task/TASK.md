Fix `mlx_lm.sample_utils.apply_top_p` so nucleus filtering retains at least the
most likely token for every row, including very small positive thresholds and
low precision log probabilities. Preserve the existing public function
signature, output shape, and retained log probabilities. Only edit
`mlx_lm/sample_utils.py`.
