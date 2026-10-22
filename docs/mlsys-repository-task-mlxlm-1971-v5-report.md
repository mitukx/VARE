# MLX-LM #1971 isolated grader v5 calibration

## Purpose

The v4 calibration runner treated any candidate rejection as a successful negative control. V5 requires the intended per-case semantic outcomes: the broken baseline and the post-cache mutant must return `wrong_output` at all three scales; the known fix must pass all scales; the constant-0.5 mutant must fail at 0.25 and 0.8 while passing at 0.5. Crashes, timeouts, missing results, and malformed output cannot satisfy the calibration gate.

V5 leaves v1-v4 protocols, locks, reports, and raw runs unchanged. It uses the public `grade_workspace()` entrypoint, which verifies and delegates to the locked v4 grader implementation.

## Frozen identities

- Protocol: `vare.mlsys-repository-task.mlxlm-1971.v5`
- Protocol SHA-256: `c3cfa38319bf400178e1e877f2db4e01791eecb06413819d282b3a683bf4a601`
- Parent v4 protocol SHA-256: `ddf00153d3cf08ee3ab6dcf978b03e4ed70ece0fd764dd32d1d44044bc022e61`
- Implementation commit: `05360189fd3ad51d1169e681f97a1a70aa27d07f`
- Baseline source SHA-256: `c110f8deed605277ae070359dd6d2ba0c700981101240c54bcd049e8d60e4847`
- Fixed source SHA-256: `a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512`
- Colima image: `localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe` (`linux/arm64`, CPU-only MLX)

## Calibration

Executed with `DOCKER_CONTEXT=colima` using the pinned baseline and fixed source revisions. All expected semantic decisions were observed:

| Candidate | 0.25 | 0.5 | 0.8 |
| --- | --- | --- | --- |
| Broken baseline | `wrong_output` | `wrong_output` | `wrong_output` |
| Exact known fix | pass | pass | pass |
| Constant-0.5 mutant | `wrong_output` | pass | `wrong_output` |
| Scale-after-cache mutant | `wrong_output` | `wrong_output` | `wrong_output` |

Source-swap rejection and the isolated grader red-team suite also passed. The runner recorded `semantic_negative_controls_match=true` and `calibration_passed=true`.

- Raw record: `results/mlxlm-1971-isolated-v5/run-1/calibration.json`
- Raw record SHA-256: `9b2264eddb5a5526d20ad7678fd6dfb39423895124124081c26fb7bbe59f186a`
- Integrity check: `cd results/mlxlm-1971-isolated-v5/run-1 && shasum -a 256 -c SHA256SUMS`
- Executed command: `DOCKER_CONTEXT=colima python scripts/run_mlxlm_1971_calibration_v5.py --source-repo /tmp/vare-mlx-lm-full-v4 --image 'localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe' --output /tmp/mlxlm-1971-v5-run`
- Reproduction: prepare a source mirror containing the exact baseline and fixed commits, then use the same command with a fresh output path such as `/tmp/mlxlm-1971-v5-reproduction`.

## Scope and limitations

This is deterministic synthetic-model grader calibration in a CPU-only container. It is not an LLM-generated repair, a pretrained-model result, Apple Metal validation, or evidence of model capability improvement. The v2 report now points to a fresh output path and identifies its historical implementation commit; its protocol and evidence are unchanged.
