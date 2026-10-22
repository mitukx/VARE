# MLX-LM #1912 isolated calibration run 1

- Result: passed.
- Implementation commit: `e9d59ade60b1f7e764847f03bf8524ac9ab9f178`.
- Baseline: 6/12 cases failed; aggregate failed as required.
- Known fix: 12/12 cases passed and the target SHA matched the pinned fixed source.
- Red team: visibility/output probe passed; timeout and memory probes failed closed; source swap was rejected.
- Image: `localhost:5000/vare/mlx-cpu-task@sha256:847edcf1b2b15d00914674e89931f9d38c4aa8a0f1d763cbb96c1359a0647388`.
- Wall time: 3.94 seconds. Peak memory was not measured.
- This is a one-task CPU calibration, not an Agent or model result.
