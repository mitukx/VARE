# MLX-LM #1971 isolated grader v3 calibration

## Why v3 exists

The v2 calibration was valid for its pinned implementation commit, but PR #98 integration updated `src/vare/isolated_candidate.py` to reject overlapping subject and staging paths. V2 remains unchanged as historical evidence. V3 binds the current main-derived staging implementation and reruns the complete calibration before current-source use.

## Frozen inputs

- Protocol: `vare.mlsys-repository-task.mlxlm-1971.v3`
- Protocol SHA-256: `08913c959ad5565d1e3899fc448ca6196eac49fddbb8bdf0c069a725326ea23b`
- Implementation commit: `3cbc0551dc235dbe883c2269bd6f41442cf43bce`
- Main merge incorporated: `38dcaa8e4048af2e637046afd5ee60370901ff7f` (#98 merge commit)
- Baseline/fixed upstream commits and target hashes are identical to v2: baseline `5f3c79d6bb1864a85d5343818192bd061b161e3a` / `c110f8deed605277ae070359dd6d2ba0c700981101240c54bcd049e8d60e4847`; fix `8fda5e976a2da412d9f8119361edefd405381cf9` / `a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512`.
- Candidate image: `localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe` (`linux/arm64`, CPU-only MLX).
- Colima: macOS Virtualization.Framework, aarch64, virtiofs.
- Scales: `0.25`, `0.5`, and `0.8`; expected tensors were generated from the exact fixed source using the pinned image and retained unchanged from v2.

## Reproduction

```bash
DOCKER_CONTEXT=colima python scripts/run_mlxlm_1971_calibration_v3.py \
  --source-repo /tmp/vare-mlx-lm-full-v4 \
  --image 'localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe' \
  --output results/mlxlm-1971-isolated-v3/run-1
```

The runner verified a clean implementation worktree, the v3 protocol lock, all locked source files, exact upstream commits, and the source bytes before execution.

## Results

| Control | Result |
| --- | --- |
| Pinned broken baseline | Rejected on all 3 scales |
| Pinned known fix | Accepted on all 3 scales |
| Constant `0.5` scaling mutant | Rejected on `0.25` and `0.8`; accepted only on `0.5` |
| Scaling after KV-cache update mutant | Rejected on all 3 scales |
| Fixed-source swap in place of baseline | Rejected |
| Grader visibility, Git history, and environment secret probes | No protected data visible |
| Network, stdout result forgery, subprocess write probes | Blocked |
| Timeout and memory exhaustion probes | Terminated and rejected |

Overall v3 calibration: **PASS**. Full per-case data is retained in `results/mlxlm-1971-isolated-v3/run-1/calibration.json` (SHA-256 `96296955784a975dbd07f3eaaf9f9d8b41e893296d8b05cd4d4dbe51a3aeca54`, recorded in adjacent `SHA256SUMS`).

## Validation and preservation

- V2 and v3 focused grader tests passed on the integrated branch.
- Colima isolated-candidate and v1/v3 #1971 grader regression suites passed with `DOCKER_CONTEXT=colima`.
- Ruff passed on the v3 module, runner, and tests.
- V1 source/protocol/output hashes and v2 protocol bytes remain unchanged; v2's old shared-runtime hash is now expected to fail closed at current HEAD. Historical v2 raw evidence is retained without reinterpretation.

## Limits

This is a deterministic synthetic-model grader calibration in a pinned CPU-only container. It is not a Coding-Agent repair, Metal validation, or model capability result. The grader supports only the pinned MLX-LM function contract.
