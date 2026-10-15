# RVL GRPO token-ID integrity gate v1

Status: frozen before implementation.

VARE base commit: `99a9d08` (PR #53 merged). Pinned RVL trainer revision: `c7e646b043cb56e5ea3c2623bb8a61e065451f72`.

## Question

Can malformed replay token IDs reach the actual VARE-to-RVL GRPO adapter and be silently changed by the pinned trainer before forming the gradient target?

The pinned trainer's `_metadata()` currently applies `int(x)` to prompt and response token IDs. VARE's `_to_verified_generation()` passes `rvl_generation.metadata` through without checking token-ID types. A float ID can therefore be truncated while the stored behavior log-probability remains paired with it; booleans also coerce to integers.

## Frozen cases

Exercise `RVLGRPOHooks.train_candidate()` with malformed metadata and a trainer spy. Cases:

| Field | Value | Expected |
| --- | --- | --- |
| `prompt_token_ids` | `[1.9]` | Reject before trainer call |
| `response_token_ids` | `[101.9]` | Reject before trainer call |
| `response_token_ids` | `[True]` | Reject before trainer call |
| `response_token_ids` | `[-1]` | Reject before trainer call |

Baseline corroboration separately passes the same adapter output to the pinned trainer's static `_metadata()` and records the coerced IDs. Valid nonnegative integer fixtures remain accepted.

## Acceptance

1. Preserve baseline source hashes and coercion outputs before editing.
2. Validate token IDs at the VARE adapter boundary before generation factories or trainer state mutation.
3. Add regressions that run through `train_candidate()` and prove the trainer is not called for malformed IDs.
4. Existing GRPO, replay, and trajectory tests pass; run the full default suite, Ruff, mypy, and blob-budget checks.

This E0 contract demonstrates a malformed-input optimization-target mismatch. It does not claim that ordinary model sampling emits malformed token IDs or establish any model-quality change.
