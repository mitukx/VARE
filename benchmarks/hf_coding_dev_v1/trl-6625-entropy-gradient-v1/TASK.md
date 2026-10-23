# trl-6625-entropy-gradient-v1

The chunked log-probability autograd function also returns entropy. Backpropagating through entropy alone should match a full-matrix reference, but currently drops part of the gradient. Restore correct gradient propagation while retaining chunked computation and existing log-probability behavior.

Editable file: `trl/trainer/utils.py`.

This candidate task is not frozen for model inference. See `task.json` for source, license, execution and limitation provenance.
