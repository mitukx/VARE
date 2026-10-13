# Held-out exact-integer base gate v1 result

Protocol `heldout-exact-integer-base-gate-v1` is stopped. The retained run is an operator-reported Apple Silicon mlx execution of code revision `5d36336ad0272d59ca815646fcb2f812849be1c5`. `environment.json` records Python 3.12.12 and `mlx-lm` 0.32.0. It does not record a git SHA; the revision above is the operator report.

## Decision

| Field | Value |
| --- | --- |
| Model | `mlx-community/SmolLM2-360M-Instruct` |
| Pack SHA-256 | `189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc` |
| Tasks passed | 0/64 |
| Decision | `stop_zero` |
| Wilson 95% | [0, 0.05662405979280534] |
| Samples | 256, seeds `20261013` through `20261268` |
| Final-line parser matches | 3 |
| Graded-correct samples | 0 |

The parser count uses the frozen rule `^####\s+(-?\d+)\s*$` on the last non-empty line. It is a description of the retained file. It does not replace `pass@4`, and it does not add a second grader. Most samples missed that final line. The three lines that matched were still not the expression value. v1 is not rescored with any other pattern.

## Files

Directory: `results/heldout-exact-integer-base-gate-v1/m1-mlx-5d36336/`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `environment.json` | 68 | `44359b7dc3db5b7584cd4b25df3087750471aeb0585c094cc9ab093310d2a64e` |
| `samples.jsonl` | 71970 | `27ecc58ca592f85d8fd8c4817166d3c6750ba32dd30a7154fb27702983a5530d` |
| `summary.json` | 522 | `c7c3d1840ae5cd0fad5d76fd96a55d08e7318810982e7e53056de66ffb05404e` |
| `manifest.json` | 1290 | `349d93ca5f3863d3e3c6bb91ba6370bb5b9eb316e90bdd9d5f2724e1b5a10b5e` |

Each file is below 1,048,576 bytes, so it is tracked. No blob-allowlist row is added. Model weights are not in the tree.

Do not retune the v1 prompt, parser, sample count, or threshold, and do not start an update from this result.
