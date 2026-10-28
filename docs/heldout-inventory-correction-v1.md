# Prospective held-out inventory correction v1

Append-only correction of the prospective source/grader-certified held-out count. No protocol, result, ledger, exposure record, or historical evidence file is edited. The prior 9/15 claim lives on PR #142 (`codex/sprint-iv-evidence-public-v1` at `91d3d19a4c9e10ffacea0f6d2c5f48f453ea4c9c`), not on main `f3ab90a36e8724441b6b9555255f8f75a0a4d529`.

## Decision

| Field | Value |
| --- | --- |
| Excluded from the prospective count | mlx-lm #1880 (`mlxlm-1880-gptq-group-error-propagation`) |
| Identity with Development `gptq-groups` | Confirmed |
| Prospective source/grader-certified count | 8/15 |
| 15-task gate | NOT MET |
| Fully model-eligible held-out total | `UNKNOWN_NOT_ESTABLISHED` |

#1880 and `gptq-groups` share upstream PR 1880, baseline `e5962529e5614ce00f14bdd39fe5fc6e410ca2b0`, fix `d4d38d96485c6546a920b3c8431e455c7aa60980`, target `mlx_lm/quant/gptq.py`, and the baseline-target, fixed-target, and grader/upstream-test hashes `81d7df02…`, `bd4c0845…`, and `f7fcf463…`. The Development task then received two `mlx-community/Mistral-7B-Instruct-v0.3-4bit` attempts (seeds 101 and 211) recorded in `results/flagship-cached-mistral-teacher-v1/summary.json` on PR #122. That summary first appears in commit `37b31519818b5dfbf245075cc541d6c8a38b4099` (committer 2026-10-24T15:12:17+09:00). Readiness-v2, created the next day, still reports `matches_for_1877_and_1880 = 0`.

The historical readiness certificate is retained. This record removes #1880 only from the prospective untouched count, on top of the #1926 removal already recorded by exposure-disposition append v1. The eight remaining task ids are listed in the JSON.

A VARE-level finding of no exposure match is not evidence that a pretrained model never saw the task. External pretraining exposure stays UNKNOWN.

## Where this check disagrees with the 2026-10-28 re-derivation

The identity and the exposure stand. Two attributions do not.

- Commit `9688af422a3f03ffac0f9a4f314f19bb5acf58d9` (2026-10-24T14:24:25+09:00) freezes the teacher protocol before model load and records a qualification summary with `model_inference = NOT_RUN`. It does not contain the teacher attempt summary. The two model attempts are in `37b31519` only.
- The sentence that the two tasks become consumed at first model access is in `protocols/flagship_cached_mistral_teacher_v1.json`, not in `protocols/flagship_development_qualification_v1.json`.

## Open items

These stay `OPEN_UNVERIFIED`. None is a disqualification of any of the remaining eight tasks.

- Cited readiness sha256 `cde44890…` does not match the committed blob or sidecar `ead67871…`.
- #1926 fix revisions differ between readiness (`3051e26b…`) and the Sprint IV disposition (`ebba55f5…`).
- #49412 is held-out with fix `01269bc9…` in readiness and development with fix `527725ca…` in the PR #102 task metadata. The PR #139 #49731 task records `external_upstream_revision` equal to that readiness fix.
- #49147, #1938, and #1979 share baseline revisions with other splits. Whether those runs, or the #1913 padding theme, touched the held-out targets was not opened.

## Files

`results/heldout-inventory-correction-v1/heldout-inventory-correction-v1.json` and its `.sha256` sidecar. The JSON cites the superseded PR #142 records by path, branch, and hash.

This record does not establish model capability, task success, or a policy update.
