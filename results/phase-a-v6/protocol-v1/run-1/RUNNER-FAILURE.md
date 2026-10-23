# Run 1 — pre-inference orchestration failure

This directory records a setup attempt for frozen protocol v1. The resource audit returned GO, the existing calibration manifests verified, the exact model and image identities matched, and the first pinned baseline was independently graded as failing as expected.

The orchestration runner then raised `KeyError: 'maximum_mlx_peak_memory_bytes'` while constructing the first agent command. The frozen protocol field is `maximum_mlx_peak_bytes_per_turn`. This exception occurred before the `run_local_agent.py` subprocess was launched. No coding-agent prompt was sent, no model was loaded by the agent runner, and no rollout or reward was produced. The partial run manifest retains `rollouts_completed: 0` and an empty rollout list.

This is an orchestration defect, not a model result. The runner was corrected before any inference. The corrected invocation is recorded separately under run 2; run 1 remains preserved and is not counted as a rollout.
