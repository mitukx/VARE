# Transformers MXFP4 destination dtype task

In `src/transformers/integrations/mxfp4.py`, repair MXFP4 conversion so the
result for `gate_up_proj` and `down_proj` has the dtype of the already-created
destination parameter. Preserve the decoded values and tensor layout.

The evaluator exercises both projections with destination dtypes `float32`,
`bfloat16`, `float16`, and `float64`. Keep the change within this source file.

The task is scored by a restricted patch-contract grader followed by CPU
behavioral checks. It is not a general repository-repair metric and does not
test model loading, CUDA, GPU kernels, or end-to-end inference.
