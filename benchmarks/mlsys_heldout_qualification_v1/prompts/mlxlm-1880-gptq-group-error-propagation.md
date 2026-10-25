# Task: correct GPTQ error propagation across groups

`gptq_quantize` partitions weight columns into groups and defers the update of
later groups. The current implementation uses an incorrect error-buffer index
and applies an in-group update past the current group boundary. Correct the
group-local error bookkeeping and restrict the immediate update to the current
group, leaving the deferred update responsible for later columns.

Keep the change limited to `mlx_lm/quant/gptq.py`. Do not change tests or
dependency metadata.
