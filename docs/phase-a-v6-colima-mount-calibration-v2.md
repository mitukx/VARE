# Phase A v6 Colima mount calibration v2

The current runner calibration was executed against the frozen arm64 image with `DOCKER_CONTEXT=colima`. It used the default cache-backed candidate staging directory, without a `TMPDIR` override. The deliberately broken fixture was rejected as `assertion_failure` (exit 1), and the known fix passed (exit 0). The report binds the run to the exact source commit and implementation hashes and records the pinned image digest and image ID.

This validates the synthetic candidate execution path and Colima bind mount only. It is not an OSS task repair, a model rollout, or evidence of policy improvement. The raw report and stdout/stderr files are in [`phase-a-v6-colima-mount-calibration-v2`](../results/phase-a-v6-colima-mount-calibration-v2/); its `SHA256SUMS` verifies the retained files.

A second append-only calibration is retained under [`run-2`](../results/phase-a-v6-colima-mount-calibration-v2/run-2/). It ran after the final #106 stack update at source commit `9cb1cb0705ec61fe00d5da8c0950fe1eb29acec3`, with a clean tree. It reproduced the same baseline rejection and known-fix acceptance.
