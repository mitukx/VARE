# MLX-LM #1971 isolated grader v2 calibration

## Hypothesis

The v1 grader's single `value_scale=0.5` case can accept a constant-scale implementation, and its cache-free forward comparison does not prove that scaling occurs before KV-cache insertion. V2 varies the scale and compares both full-forward and incremental cache state from the pinned fixed implementation.

## Frozen inputs

- Protocol: `vare.mlsys-repository-task.mlxlm-1971.v2`
- Protocol SHA-256: `86852e5991244d7a0716faae24b45a22fc14933315e6b84ab3a92e904590db6b`
- Implementation commit: `4c0ae92128f1bea4098a18e130294f562c95c6ff`
- Repository: `ml-explore/mlx-lm`
- Baseline commit / target SHA-256: `5f3c79d6bb1864a85d5343818192bd061b161e3a` / `c110f8deed605277ae070359dd6d2ba0c700981101240c54bcd049e8d60e4847`
- Fixed commit / target SHA-256: `8fda5e976a2da412d9f8119361edefd405381cf9` / `a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512`
- Candidate image: `localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe` (`linux/arm64`, CPU-only MLX)
- Colima: macOS Virtualization.Framework, aarch64, virtiofs
- Input scales: `0.25`, `0.5`, `0.8`; all other synthetic model inputs and seed are inherited from v1.

## Reproduction

```bash
DOCKER_CONTEXT=colima python scripts/run_mlxlm_1971_calibration_v2.py \
  --source-repo /tmp/vare-mlx-lm-full-v4 \
  --image 'localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe' \
  --output /tmp/mlxlm-1971-v2-run
```

First check out the exact locked implementation revision `4c0ae92128f1bea4098a18e130294f562c95c6ff`; later grader changes intentionally make this historical v2 implementation lock fail closed. The source mirror contained both exact upstream Git commits. The runner checked its implementation lock and clean worktree before it executed. The fresh output path preserves the retained run-1 evidence.

## Results

| Control | Result |
| --- | --- |
| Pinned broken baseline | Rejected on all 3 scales |
| Pinned known fix | Accepted on all 3 scales |
| Constant `0.5` scaling mutant | Rejected on `0.25` and `0.8`; the `0.5` case alone passes, as expected |
| Scaling after KV-cache update mutant | Rejected on all 3 scales |
| Fixed-source swap in place of baseline | Rejected by pinned source verification |
| Grader visibility, Git history, and environment secret probes | No protected data visible |
| Network, stdout result forgery, and subprocess write probes | Blocked |
| Timeout and memory exhaustion probes | Terminated and rejected |

Overall calibration: **PASS**. Raw details and per-case timings are in `results/mlxlm-1971-isolated-v2/run-1/calibration.json`; its SHA-256 is `fa81750e5f20bfba41528259502ab9f58f916521cdb90503836da4a7e00f8a4e` and is recorded in the adjacent `SHA256SUMS` file.

## Validation and preservation

- Focused v2 tests: 7 passed.
- Ruff on all changed Python files: passed.
- `git diff --cached --check`: passed.
- v2 protocol, lock, all 11 implementation inputs, and v1 protocol/output hashes were independently checked; all matched.
- V1 protocol and historical calibration evidence were not changed.

## Limits

This is a narrow grader calibration using a small deterministic synthetic model in a pinned CPU-only container. It is not an LLM-generated repair, a Coding-Agent success, a Metal validation, or a model capability result. The task covers only the pinned MLX-LM function contract; it does not establish a general-purpose safe repository grader.
