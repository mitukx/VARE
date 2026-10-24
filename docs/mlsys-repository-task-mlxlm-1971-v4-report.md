# MLX-LM #1971 isolated grader v4 calibration

## Why v4 exists

Independent review found that v2/v3 improved grader semantics but left the public grade_workspace() API bound to the v1 single-scale, cache-free grader. V4 changes that public entry point to dispatch through the current scale-diverse, incremental-cache grader. V1, v2, and v3 protocols and their historical calibration records remain preserved.

## Frozen inputs

- Protocol: vare.mlsys-repository-task.mlxlm-1971.v4
- Protocol SHA-256: ddf00153d3cf08ee3ab6dcf978b03e4ed70ece0fd764dd32d1d44044bc022e61
- Implementation commit: 23f71a0c6923f6e677ed8abc0d4ed186ae047e89
- Protocol parent v3 SHA-256 is recorded in protocol-v4.json.
- Baseline revision/hash: 5f3c79d6bb1864a85d5343818192bd061b161e3a / c110f8deed605277ae070359dd6d2ba0c700981101240c54bcd049e8d60e4847
- Fixed revision/hash: 8fda5e976a2da412d9f8119361edefd405381cf9 / a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512
- Pinned image: localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe (linux/arm64, CPU-only MLX)
- Candidate input scales: 0.25, 0.5, 0.8.
- The calibration runner invoked the public grade_workspace() API, not only the versioned internal grader.

## Reproduction

Use a new output directory; this path is intentionally outside the repository so a fresh checkout can reproduce without colliding with retained run-1:

    DOCKER_CONTEXT=colima python scripts/run_mlxlm_1971_calibration_v4.py --source-repo /tmp/vare-mlx-lm-full-v4 --image 'localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe' --output /tmp/mlxlm-1971-v4-run

## Results

| Control | Result |
| --- | --- |
| Pinned broken baseline | Rejected on all three scales |
| Pinned known fix | Accepted on all three scales |
| Constant 0.5 scaling mutant | Rejected at 0.25 and 0.8; accepted only at 0.5 |
| Scaling after KV-cache update mutant | Rejected on all three scales |
| Fixed-source swap | Rejected |
| Grader visibility, Git history, environment secret, network, stdout forgery and subprocess write probes | Blocked or hidden as expected |
| Timeout and memory exhaustion probes | Terminated and rejected |

Overall calibration: PASS. Raw result is results/mlxlm-1971-isolated-v4/run-1/calibration.json, SHA-256 5722fde84bc72da0f69445524857c453c35e1414c8cd99e9d70f6f3b83239805, recorded in adjacent SHA256SUMS.

## Validation and limitations

- V4 focused tests verify protocol and implementation locks, v1 evidence preservation, reject controls, and that the public grade_workspace() API dispatches to v4.
- V2 and v3 remain historical versions and fail closed against the changed live implementation; their original source commits and evidence were not rewritten.
- This is a narrow deterministic synthetic-model calibration using a CPU-only container. It is not an LLM-generated repair, Apple Metal validation, or a capability improvement result.
